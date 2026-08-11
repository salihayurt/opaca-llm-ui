"""
Vector storage for indexed documents.

A thin, typed wrapper over Qdrant. It deals in vectors only -- embedding a
chunk or a query happens a layer up -- which keeps this module testable
against an in-memory Qdrant instance with no server, no network and no API
key.

Three things it exists to get right, all of them defects in the FAISS-based
implementation it replaces (`vector_storage.py`, see docs/rag_design.md):

1. **Session isolation.** The old store was a module-level singleton holding
   one index, so every user's documents shared it and one session's uploads
   were retrievable from another's. Collections here are per session, and a
   search cannot reach outside the session it was issued for.

2. **Persistence and deletion.** A `faiss.IndexFlatL2` lives in the process
   and dies with it, and removal was left as a TODO. Qdrant persists, and
   documents can be deleted individually or with their session.

3. **Stable identity.** Point ids are derived from the file and chunk index,
   so re-indexing a file overwrites its chunks instead of appending a second
   copy of every one of them.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from .chunking import Chunk

logger = logging.getLogger(__name__)

COLLECTION_PREFIX = "sage-session"

# Namespace for deterministic point ids. Fixed, because changing it would
# orphan every previously indexed chunk rather than overwriting it.
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


@dataclass(frozen=True)
class StoredChunk:
    """A chunk as it comes back from a search, with everything a citation needs."""

    text: str
    file_id: str
    filename: str
    chunk_index: int
    location: str | None
    score: float


def collection_name(session_id: str) -> str:
    """Map a session id to a collection name.

    Session ids come from the client, so they are sanitised rather than
    trusted: an id containing a slash or a quote would otherwise reach
    Qdrant's HTTP path unescaped.
    """
    if not session_id:
        raise ValueError("session_id must not be empty")
    return f"{COLLECTION_PREFIX}-{_UNSAFE.sub('_', session_id)}"


def point_id(file_id: str, chunk_index: int) -> str:
    """A stable id for one chunk of one file.

    Deterministic so that re-indexing a document replaces its chunks. With
    random ids, uploading the same file twice would double every chunk, and
    the duplicates would then compete with each other for slots in the
    retrieved context.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{file_id}:{chunk_index}"))


class DocumentStore:
    """Per-session vector storage for document chunks."""

    def __init__(self, client: AsyncQdrantClient, vector_size: int):
        self.client = client
        self.vector_size = vector_size

    # -- collection lifecycle ------------------------------------------------

    async def ensure_collection(self, session_id: str) -> str:
        """Create the session's collection if it does not exist yet.

        Payload indexes are created alongside it. Without them Qdrant filters
        by scanning, which is fine at our scale but degrades as a session
        accumulates documents -- and the filters here run on every query.
        """
        name = collection_name(session_id)
        if await self.client.collection_exists(name):
            return name

        await self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        for field in ("file_id", "active"):
            await self.client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD
                if field == "file_id"
                else models.PayloadSchemaType.BOOL,
            )
        return name

    async def delete_session(self, session_id: str) -> None:
        """Drop a session's collection.

        Called from session teardown. Without it, collections outlive the
        sessions that own them and accumulate with no owner and no expiry.
        """
        name = collection_name(session_id)
        if await self.client.collection_exists(name):
            await self.client.delete_collection(name)

    # -- writing -------------------------------------------------------------

    async def upsert_document(
        self,
        session_id: str,
        file_id: str,
        filename: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        *,
        active: bool = True,
    ) -> int:
        """Store a document's chunks, replacing any previous version of it.

        Returns the number of chunks written.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(chunks)} chunks but {len(vectors)} vectors -- "
                "these must correspond one to one"
            )
        if not chunks:
            return 0

        for vector in vectors:
            if len(vector) != self.vector_size:
                raise ValueError(
                    f"vector has {len(vector)} dimensions, collection expects "
                    f"{self.vector_size}. The embedding model likely changed; "
                    "the collection must be rebuilt rather than mixed."
                )

        name = await self.ensure_collection(session_id)

        # Chunk count can shrink when a file is re-indexed with different
        # settings. Deterministic ids overwrite the chunks that still exist
        # but would strand the surplus, so the old version is removed first.
        await self.delete_document(session_id, file_id)

        await self.client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(
                    id=point_id(file_id, chunk.index),
                    vector=vector,
                    payload={
                        "file_id": file_id,
                        "filename": filename,
                        "chunk_index": chunk.index,
                        "location": chunk.location,
                        "text": chunk.text,
                        "active": active,
                    },
                )
                for chunk, vector in zip(chunks, vectors)
            ],
        )
        return len(chunks)

    async def set_active(self, session_id: str, file_id: str, active: bool) -> None:
        """Include or exclude a document from future searches.

        Deactivating keeps the vectors, so re-activating costs nothing --
        which is the point of having activation separate from deletion.
        """
        name = collection_name(session_id)
        if not await self.client.collection_exists(name):
            return
        await self.client.set_payload(
            collection_name=name,
            payload={"active": active},
            points=models.Filter(must=[_file_condition(file_id)]),
        )

    async def delete_document(self, session_id: str, file_id: str) -> None:
        """Remove one document's chunks, leaving the rest of the session intact."""
        name = collection_name(session_id)
        if not await self.client.collection_exists(name):
            return
        await self.client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[_file_condition(file_id)])
            ),
        )

    # -- reading -------------------------------------------------------------

    async def search(
        self,
        session_id: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        active_only: bool = True,
        min_score: float = 0.0,
    ) -> list[StoredChunk]:
        """Return the nearest chunks within one session.

        A session with nothing indexed returns an empty list rather than
        raising: asking a question before uploading anything is ordinary use,
        not an error.

        `min_score` is a cosine floor. At 0.0 the nearest chunks are returned
        whatever their similarity, which is why an unrelated question still
        gets passages back. See RetrievalConfig.min_dense_score.
        """
        name = collection_name(session_id)
        if not await self.client.collection_exists(name):
            return []

        query_filter = (
            models.Filter(must=[models.FieldCondition(
                key="active", match=models.MatchValue(value=True))])
            if active_only
            else None
        )
        response = await self.client.query_points(
            collection_name=name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            score_threshold=min_score or None,
        )
        return [_to_stored_chunk(point.payload, point.score) for point in response.points]

    async def iter_chunks(
        self,
        session_id: str,
        *,
        active_only: bool = True,
    ) -> list[StoredChunk]:
        """Return every chunk in a session, for lexical search over payloads.

        Scored 0.0: these are not ranked, only enumerated.
        """
        name = collection_name(session_id)
        if not await self.client.collection_exists(name):
            return []

        scroll_filter = (
            models.Filter(must=[models.FieldCondition(
                key="active", match=models.MatchValue(value=True))])
            if active_only
            else None
        )

        collected: list[StoredChunk] = []
        offset = None
        while True:
            points, offset = await self.client.scroll(
                collection_name=name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            collected.extend(_to_stored_chunk(point.payload, 0.0) for point in points)
            if offset is None:
                break
        return collected

    async def list_documents(self, session_id: str) -> list[dict]:
        """Summarise the session's indexed documents for the UI.

        One entry per file with its chunk count and activation state.
        """
        summary: dict[str, dict] = {}
        for chunk in await self.iter_chunks(session_id, active_only=False):
            entry = summary.setdefault(
                chunk.file_id,
                {"file_id": chunk.file_id, "filename": chunk.filename, "chunks": 0},
            )
            entry["chunks"] += 1

        name = collection_name(session_id)
        if await self.client.collection_exists(name):
            active_ids = {
                chunk.file_id
                for chunk in await self.iter_chunks(session_id, active_only=True)
            }
            for file_id, entry in summary.items():
                entry["active"] = file_id in active_ids
        return sorted(summary.values(), key=lambda entry: entry["filename"])


def _file_condition(file_id: str) -> models.FieldCondition:
    return models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))


def _to_stored_chunk(payload: dict | None, score: float) -> StoredChunk:
    payload = payload or {}
    return StoredChunk(
        text=payload.get("text", ""),
        file_id=payload.get("file_id", ""),
        filename=payload.get("filename", ""),
        chunk_index=payload.get("chunk_index", -1),
        location=payload.get("location"),
        score=score,
    )
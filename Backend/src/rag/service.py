"""
The RAG pipeline, assembled.

Everything below this module is a component with one job. This is where they
become a pipeline: extract, chunk, embed, store on the way in; retrieve, fuse,
fit to budget, attribute on the way out.

It is also where the two decisions that are about product rather than
algorithm live:

- **Not every document should be indexed.** Indexing a three-page PDF that
  fits in the prompt costs money and adds latency for no benefit, and the
  answer is worse, because the model sees three retrieved fragments instead
  of the whole document. Small files are refused with a reason rather than
  silently accepted.
- **A retrieved passage must be attributable.** The task asks for retrieved
  information to be marked with its source, and that only works if the source
  survives extraction, chunking and storage. It does, and `format_context`
  is where it becomes visible to the model.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from ..text_extraction import UnsupportedFileType, extract_segments
from .chunking import Chunk, chunk_segments, estimate_tokens
from .embedding import Embedder, EmbeddingError
from .retrieval import RetrievalConfig, Retriever, fit_to_budget
from .store import DocumentStore, StoredChunk

logger = logging.getLogger(__name__)

# Documents shorter than this are better handled by putting them in the prompt
# whole. Roughly five pages of prose.
DEFAULT_MIN_INDEX_CHARS = 20_000

# How much of the prompt retrieved context may occupy. SAGE already spends
# around 13k tokens on a simple question before any document is involved.
DEFAULT_CONTEXT_BUDGET = 2_000


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        logger.warning("%s is not a number; using %s", name, default)
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        logger.warning("%s is not an integer; using %d", name, default)
        return default


@dataclass
class RagConfig:
    """Runtime configuration, one field per ablation axis.

    Read from the environment so a configuration can be changed without a code
    change -- which is what makes the evaluation a sweep rather than a series
    of edits.
    """

    min_index_chars: int = DEFAULT_MIN_INDEX_CHARS
    context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET
    chunk_tokens: int = 300
    chunk_overlap: int = 50
    hybrid: bool = True
    rerank: bool = False
    min_dense_score: float = 0.0

    @classmethod
    def from_env(cls) -> "RagConfig":
        return cls(
            min_index_chars=_env_int("RAG_MIN_INDEX_CHARS", DEFAULT_MIN_INDEX_CHARS),
            context_budget_tokens=_env_int("RAG_CONTEXT_BUDGET", DEFAULT_CONTEXT_BUDGET),
            chunk_tokens=_env_int("RAG_CHUNK_SIZE", 300),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 50),
            hybrid=_env_flag("ENABLE_HYBRID_SEARCH", True),
            rerank=_env_flag("ENABLE_RERANKING", False),
            min_dense_score=_env_float("RAG_MIN_DENSE_SCORE", 0.0),
        )


@dataclass
class IndexResult:
    """What happened to one document.

    `indexed` false is not necessarily an error: a document can be refused
    because it is small enough to read directly, which is the better outcome.
    `reason` is written for the user, since it is shown to them.
    """

    indexed: bool
    filename: str
    chunks: int = 0
    reason: str = ""


@dataclass
class SearchResult:
    """Retrieved context, ready to hand to a model."""

    context: str
    sources: list[dict] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.context)


def format_context(chunks: list[StoredChunk]) -> tuple[str, list[dict]]:
    """Render chunks as numbered, attributed passages.

    Numbering is what lets the model cite: without a handle, an instruction to
    mark sources produces either nothing or an invented filename. The same
    numbers are returned separately so the frontend can turn them into links
    rather than parsing them back out of the model's prose.
    """
    blocks = []
    sources = []
    for number, chunk in enumerate(chunks, start=1):
        where = f"{chunk.filename}, {chunk.location}" if chunk.location else chunk.filename
        blocks.append(f"[{number}] {where}\n{chunk.text}")
        sources.append({
            "number": number,
            "filename": chunk.filename,
            "file_id": chunk.file_id,
            "location": chunk.location,
            "chunk_index": chunk.chunk_index,
        })
    return "\n\n".join(blocks), sources


class RagService:
    """Indexing and retrieval for one deployment, scoped per session."""

    def __init__(
        self,
        store: DocumentStore,
        embedder: Embedder,
        *,
        config: RagConfig | None = None,
        reranker=None,
    ):
        self.config = config or RagConfig()
        self.store = store
        self.embedder = embedder
        self.retriever = Retriever(
            store,
            embedder,
            config=RetrievalConfig(
                hybrid=self.config.hybrid,
                rerank=self.config.rerank,
                min_dense_score=self.config.min_dense_score,
            ),
            reranker=reranker,
        )
        self._summaries: dict[str, list[dict]] = {}

    # -- indexing ------------------------------------------------------------

    async def index_document(
        self,
        session_id: str,
        file_id: str,
        filename: str,
        data: bytes,
    ) -> IndexResult:
        """Extract, chunk, embed and store one uploaded file."""
        try:
            segments = extract_segments(filename, data)
        except UnsupportedFileType as error:
            return IndexResult(False, filename, reason=str(error))

        total_characters = sum(len(segment.text) for segment in segments)
        if total_characters == 0:
            # Readable file, no text. Almost always a scanned PDF with no text
            # layer. Distinguished from an unsupported format because the
            # remedy is different: this one needs OCR, not another file type.
            return IndexResult(
                False, filename,
                reason=(
                    f"No text could be extracted from '{filename}'. "
                    "If it is a scanned document, it needs OCR before it can be indexed."
                ),
            )

        if total_characters < self.config.min_index_chars:
            # Refusing here is the better answer, not a limitation. Retrieval
            # would hand the model a few fragments of a document it could have
            # read whole.
            return IndexResult(
                False, filename,
                reason=(
                    f"'{filename}' is small enough to read directly "
                    f"({total_characters} characters); indexing it would give "
                    "the model fragments instead of the whole document."
                ),
            )

        chunks = chunk_segments(
            segments,
            target_tokens=self.config.chunk_tokens,
            overlap_tokens=self.config.chunk_overlap,
        )
        if not chunks:
            return IndexResult(False, filename, reason=f"'{filename}' produced no chunks.")

        try:
            vectors = await self.embedder.embed_documents([chunk.text for chunk in chunks])
        except (EmbeddingError, ValueError) as error:
            logger.warning("Embedding failed for %s: %s", filename, error)
            return IndexResult(False, filename, reason=f"Could not embed '{filename}': {error}")

        written = await self.store.upsert_document(
            session_id, file_id, filename, chunks, vectors
        )
        # The lexical index is cached per session; leaving it stale would make
        # this document findable by dense search and invisible to keyword
        # search, which reads as retrieval being unreliable.
        self.retriever.invalidate(session_id)
        self._summaries.pop(session_id, None)

        return IndexResult(True, filename, chunks=written)

    # -- document management -------------------------------------------------

    async def set_active(self, session_id: str, file_id: str, active: bool) -> None:
        await self.store.set_active(session_id, file_id, active)
        self.retriever.invalidate(session_id)
        self._summaries.pop(session_id, None)

    async def remove_document(self, session_id: str, file_id: str) -> None:
        await self.store.delete_document(session_id, file_id)
        self.retriever.invalidate(session_id)
        self._summaries.pop(session_id, None)

    async def clear_session(self, session_id: str) -> None:
        """Drop everything a session indexed. Called from session teardown."""
        await self.store.delete_session(session_id)
        self.retriever.invalidate(session_id)
        self._summaries.pop(session_id, None)

    async def list_documents(self, session_id: str) -> list[dict]:
        """Summarise the session's indexed documents, cached.

        The tool layer needs this on every turn, to decide whether to offer a
        search tool at all and to name the searchable documents in its
        description. Reading it from the vector store each time would add two
        scrolls to every message. It is loaded once per session and updated on
        write instead.

        Loading is lazy rather than eager because collections outlive the
        process: after a restart, a session's documents are still indexed, and
        a cache populated only by writes would report none of them.
        """
        if session_id not in self._summaries:
            self._summaries[session_id] = await self.store.list_documents(session_id)
        return self._summaries[session_id]

    # -- retrieval -----------------------------------------------------------

    async def search(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 10,
        budget_tokens: int | None = None,
        file_ids: set[str] | None = None,
    ) -> SearchResult:
        """Retrieve context for `query`, trimmed to the prompt budget.

        `file_ids` restricts the search to particular documents; None searches
        everything indexed for the session.
        """
        result = await self.retriever.retrieve(
            session_id, query, limit=limit, file_ids=file_ids
        )
        if not result.chunks:
            return SearchResult(context="", degraded=result.degraded)

        kept = fit_to_budget(
            result.chunks,
            budget_tokens if budget_tokens is not None else self.config.context_budget_tokens,
            count_tokens=estimate_tokens,
        )
        context, sources = format_context(kept)
        return SearchResult(context=context, sources=sources, degraded=result.degraded)


def chunks_of(segments, config: RagConfig) -> list[Chunk]:
    """Chunk segments with a service configuration. Exposed for the harness."""
    return chunk_segments(
        segments,
        target_tokens=config.chunk_tokens,
        overlap_tokens=config.chunk_overlap,
    )
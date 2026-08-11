"""
Tests for the vector-store side of session teardown.

Indexed documents are the embedded form of a session's uploads, so they have
to be removed wherever the uploads are. There are three such places in
session_manager.py, and each is covered here -- a leak on any one of them
leaves document text in Qdrant with no session left to own or reclaim it.
"""

import warnings

import pytest
from qdrant_client import AsyncQdrantClient

import src.session_manager as session_manager
from src.rag.service import RagConfig, RagService
from src.rag.store import DocumentStore, collection_name

VECTOR_SIZE = 8


class FakeEmbedder:
    def _vector(self, text):
        vector = [0.0] * VECTOR_SIZE
        for character in text.lower():
            if character.isalnum():
                vector[ord(character) % VECTOR_SIZE] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    async def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    async def embed_query(self, text):
        return self._vector(text)

    async def dimensions(self):
        return VECTOR_SIZE


def long_text(topic: str, characters: int = 1200) -> bytes:
    sentence = f"This section explains {topic} for building operations. "
    body = ""
    while len(body) < characters:
        body += sentence
    return body.encode()


@pytest.fixture
async def rag(monkeypatch):
    """A live RAG service, patched in as the one session_manager will find."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
        client = AsyncQdrantClient(":memory:")
        service = RagService(
            DocumentStore(client, vector_size=VECTOR_SIZE),
            FakeEmbedder(),
            config=RagConfig(min_index_chars=200),
        )

        async def fake_factory():
            return service

        import src.rag.factory as factory
        monkeypatch.setattr(factory, "get_rag_service", fake_factory)
        yield service
        await client.close()


async def index_something(service, session_id: str) -> None:
    await service.index_document(session_id, "f1", "manual.txt", long_text("thermostats"))
    assert await service.search(session_id, "thermostat")


async def is_gone(service, session_id: str) -> bool:
    return not await service.store.client.collection_exists(collection_name(session_id))


# -- the three teardown paths -----------------------------------------------

@pytest.mark.anyio
async def test_deleting_a_session_drops_its_vectors(rag, monkeypatch):
    """The ordinary path, next to delete_all_files_from_disk."""
    monkeypatch.setattr(session_manager, "sessions", {})
    await index_something(rag, "s1")

    await session_manager.delete_session("s1")

    assert await is_gone(rag, "s1")


@pytest.mark.anyio
async def test_shutdown_without_a_database_drops_vectors(rag, monkeypatch):
    """Without Mongo, sessions are lost on shutdown and their files deleted.
    The vectors derived from those files have to go too, or they persist with
    no session left to reclaim them."""
    await index_something(rag, "s1")
    monkeypatch.setattr(session_manager, "sessions", {"s1": object()})
    monkeypatch.setattr(session_manager.db_client, "is_db_configured", lambda: False)
    monkeypatch.setattr(session_manager, "delete_all_files_from_disk", lambda _: None)

    await session_manager.on_shutdown()

    assert await is_gone(rag, "s1")


@pytest.mark.anyio
async def test_deleting_one_session_leaves_the_others(rag, monkeypatch):
    monkeypatch.setattr(session_manager, "sessions", {})
    await index_something(rag, "alice")
    await index_something(rag, "bob")

    await session_manager.delete_session("alice")

    assert await is_gone(rag, "alice")
    assert not await is_gone(rag, "bob")


# -- failure must not block teardown ----------------------------------------

@pytest.mark.anyio
async def test_an_unreachable_vector_store_does_not_block_deletion(rag, monkeypatch):
    """A session must still be deleted when an optional service is down;
    otherwise Qdrant being unavailable blocks cleanup of files and database
    rows as well."""
    deleted = []
    monkeypatch.setattr(session_manager, "sessions", {})
    monkeypatch.setattr(session_manager, "delete_all_files_from_disk", deleted.append)

    async def explode(_):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(rag, "clear_session", explode)

    await session_manager.delete_session("s1")  # must not raise

    assert deleted == ["s1"]


@pytest.mark.anyio
async def test_teardown_is_harmless_when_rag_is_disabled(rag, monkeypatch):
    """RAG not configured is an ordinary state, not an error path."""
    async def no_service():
        return None

    import src.rag.factory as factory
    monkeypatch.setattr(factory, "get_rag_service", no_service)
    monkeypatch.setattr(session_manager, "sessions", {})

    await session_manager.delete_session("s1")


@pytest.mark.anyio
async def test_deleting_a_session_that_indexed_nothing_is_harmless(rag, monkeypatch):
    monkeypatch.setattr(session_manager, "sessions", {})
    await session_manager.delete_session("never-used")
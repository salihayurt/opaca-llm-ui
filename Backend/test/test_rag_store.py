"""
Tests for src/rag/store.py.

Run against an in-memory Qdrant instance, so no server, no network and no API
key are involved. Vectors are hand-written unit vectors rather than real
embeddings: the store's job is storage, filtering and identity, none of which
depend on what the numbers mean.

Async tests use @pytest.mark.anyio, matching test_admin.py.
"""

import warnings

import pytest
from qdrant_client import AsyncQdrantClient

from src.rag.chunking import Chunk
from src.rag.store import DocumentStore, collection_name, point_id

VECTOR_SIZE = 4


@pytest.fixture
def anyio_backend():
    """Run these tests on asyncio only.

    anyio's default fixture parametrises over backends, and what it picks
    depends on the version: 4.10 parametrises over asyncio and trio whether or
    not trio is installed, while 4.14 only offers backends actually present.
    Left to the default, the same suite errors on one machine and passes on
    another.

    Pinning is also the technically correct choice here rather than a
    workaround: qdrant-client's async client is built on asyncio, so a trio
    run would be testing a combination that cannot occur in production.
    """
    return "asyncio"


@pytest.fixture
async def store():
    """A store backed by a fresh in-memory Qdrant.

    Local Qdrant ignores payload indexes and says so on every collection
    creation. The indexes matter on a real server, so the production code
    keeps creating them and the warning is filtered here rather than removed
    there -- at the cost that these tests cannot verify the indexes are
    actually used, only that the code path runs.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
        client = AsyncQdrantClient(":memory:")
        yield DocumentStore(client, vector_size=VECTOR_SIZE)
        await client.close()


def chunks(count: int, prefix: str = "text") -> list[Chunk]:
    return [
        Chunk(text=f"{prefix} {i}", location=f"page {i + 1}", index=i)
        for i in range(count)
    ]


def vectors(count: int, axis: int = 0) -> list[list[float]]:
    """`count` distinct vectors clustered around one axis."""
    out = []
    for i in range(count):
        vector = [0.0] * VECTOR_SIZE
        vector[axis] = 1.0
        vector[(axis + 1) % VECTOR_SIZE] = i * 0.01
        out.append(vector)
    return out


# -- naming and identity ----------------------------------------------------

def test_collection_name_is_namespaced_per_session():
    assert collection_name("abc").startswith("sage-session")
    assert collection_name("abc") != collection_name("abd")


def test_collection_name_sanitises_session_ids():
    """Session ids come from the client and reach Qdrant's HTTP path."""
    name = collection_name("../../etc/passwd")
    assert "/" not in name and ".." not in name


def test_collection_name_rejects_empty_session():
    with pytest.raises(ValueError):
        collection_name("")


def test_point_ids_are_deterministic_and_distinct():
    """Random ids would make re-indexing a file double every chunk."""
    assert point_id("f1", 0) == point_id("f1", 0)
    assert point_id("f1", 0) != point_id("f1", 1)
    assert point_id("f1", 0) != point_id("f2", 0)


# -- writing and reading ----------------------------------------------------

@pytest.mark.anyio
async def test_upsert_then_search_returns_the_chunks(store):
    written = await store.upsert_document("s1", "f1", "manual.pdf", chunks(3), vectors(3))
    assert written == 3

    hits = await store.search("s1", [1.0, 0.0, 0.0, 0.0], limit=5)
    assert len(hits) == 3
    assert all(hit.filename == "manual.pdf" for hit in hits)
    assert {hit.chunk_index for hit in hits} == {0, 1, 2}


@pytest.mark.anyio
async def test_search_returns_citation_metadata(store):
    """A retrieved passage that cannot be attributed is not much use."""
    await store.upsert_document("s1", "f1", "guide.docx", chunks(1), vectors(1))
    hit = (await store.search("s1", [1.0, 0.0, 0.0, 0.0]))[0]
    assert hit.filename == "guide.docx"
    assert hit.location == "page 1"
    assert hit.file_id == "f1"
    assert hit.text


@pytest.mark.anyio
async def test_search_on_empty_session_returns_nothing(store):
    """Asking before uploading is ordinary use, not an error."""
    assert await store.search("never-used", [1.0, 0.0, 0.0, 0.0]) == []
    assert await store.iter_chunks("never-used") == []
    assert await store.list_documents("never-used") == []


@pytest.mark.anyio
async def test_upsert_rejects_mismatched_vector_count(store):
    with pytest.raises(ValueError):
        await store.upsert_document("s1", "f1", "a.pdf", chunks(3), vectors(2))


@pytest.mark.anyio
async def test_upsert_rejects_wrong_dimensionality(store):
    """Switching embedding model changes the vector size; mixing dimensions
    silently would corrupt the index rather than fail."""
    with pytest.raises(ValueError) as excinfo:
        await store.upsert_document("s1", "f1", "a.pdf", chunks(1), [[1.0, 0.0]])
    assert "embedding model" in str(excinfo.value)


@pytest.mark.anyio
async def test_upsert_of_empty_document_is_a_no_op(store):
    assert await store.upsert_document("s1", "f1", "blank.pdf", [], []) == 0


# -- the defect this store exists to fix ------------------------------------

@pytest.mark.anyio
async def test_sessions_cannot_see_each_other(store):
    """The FAISS implementation held one process-wide index, so one user's
    uploads were retrievable from another user's session."""
    await store.upsert_document("alice", "f1", "salary.pdf", chunks(2), vectors(2))
    await store.upsert_document("bob", "f2", "notes.pdf", chunks(2), vectors(2))

    alice_hits = await store.search("alice", [1.0, 0.0, 0.0, 0.0], limit=10)
    bob_hits = await store.search("bob", [1.0, 0.0, 0.0, 0.0], limit=10)

    assert {hit.filename for hit in alice_hits} == {"salary.pdf"}
    assert {hit.filename for hit in bob_hits} == {"notes.pdf"}


@pytest.mark.anyio
async def test_reindexing_replaces_rather_than_duplicates(store):
    """Duplicated chunks compete with each other for slots in the retrieved
    context, crowding out other documents."""
    await store.upsert_document("s1", "f1", "manual.pdf", chunks(3), vectors(3))
    await store.upsert_document("s1", "f1", "manual.pdf", chunks(3), vectors(3))

    assert len(await store.iter_chunks("s1")) == 3


@pytest.mark.anyio
async def test_reindexing_with_fewer_chunks_leaves_no_orphans(store):
    """Deterministic ids overwrite the chunks that still exist, but a shorter
    document would strand the surplus from the previous version."""
    await store.upsert_document("s1", "f1", "manual.pdf", chunks(5), vectors(5))
    await store.upsert_document("s1", "f1", "manual.pdf", chunks(2), vectors(2))

    stored = await store.iter_chunks("s1")
    assert len(stored) == 2
    assert {chunk.chunk_index for chunk in stored} == {0, 1}


# -- activation -------------------------------------------------------------

@pytest.mark.anyio
async def test_deactivated_documents_are_excluded_from_search(store):
    await store.upsert_document("s1", "f1", "a.pdf", chunks(2), vectors(2))
    await store.upsert_document("s1", "f2", "b.pdf", chunks(2), vectors(2))

    await store.set_active("s1", "f2", False)

    hits = await store.search("s1", [1.0, 0.0, 0.0, 0.0], limit=10)
    assert {hit.filename for hit in hits} == {"a.pdf"}


@pytest.mark.anyio
async def test_deactivation_is_reversible_without_reindexing(store):
    """Vectors are kept, which is the point of activation being separate from
    deletion."""
    await store.upsert_document("s1", "f1", "a.pdf", chunks(2), vectors(2))
    await store.set_active("s1", "f1", False)
    assert await store.search("s1", [1.0, 0.0, 0.0, 0.0]) == []

    await store.set_active("s1", "f1", True)
    assert len(await store.search("s1", [1.0, 0.0, 0.0, 0.0])) == 2


@pytest.mark.anyio
async def test_inactive_chunks_are_still_visible_when_asked_for(store):
    await store.upsert_document("s1", "f1", "a.pdf", chunks(2), vectors(2))
    await store.set_active("s1", "f1", False)

    assert await store.iter_chunks("s1", active_only=True) == []
    assert len(await store.iter_chunks("s1", active_only=False)) == 2


@pytest.mark.anyio
async def test_set_active_on_unknown_session_is_harmless(store):
    await store.set_active("nope", "f1", False)


# -- deletion ---------------------------------------------------------------

@pytest.mark.anyio
async def test_delete_document_leaves_the_others(store):
    await store.upsert_document("s1", "f1", "a.pdf", chunks(2), vectors(2))
    await store.upsert_document("s1", "f2", "b.pdf", chunks(2), vectors(2))

    await store.delete_document("s1", "f1")

    remaining = await store.iter_chunks("s1")
    assert {chunk.filename for chunk in remaining} == {"b.pdf"}


@pytest.mark.anyio
async def test_delete_session_removes_everything(store):
    """Called from session teardown; without it collections outlive their
    owner and accumulate with no expiry."""
    await store.upsert_document("s1", "f1", "a.pdf", chunks(2), vectors(2))
    await store.delete_session("s1")

    assert not await store.client.collection_exists(collection_name("s1"))
    assert await store.search("s1", [1.0, 0.0, 0.0, 0.0]) == []


@pytest.mark.anyio
async def test_delete_session_does_not_touch_other_sessions(store):
    await store.upsert_document("s1", "f1", "a.pdf", chunks(2), vectors(2))
    await store.upsert_document("s2", "f2", "b.pdf", chunks(2), vectors(2))

    await store.delete_session("s1")

    assert len(await store.search("s2", [1.0, 0.0, 0.0, 0.0], limit=10)) == 2


@pytest.mark.anyio
async def test_deleting_unknown_things_is_harmless(store):
    await store.delete_session("nope")
    await store.delete_document("nope", "f1")


# -- listing ----------------------------------------------------------------

@pytest.mark.anyio
async def test_list_documents_summarises_the_session(store):
    await store.upsert_document("s1", "f1", "manual.pdf", chunks(3), vectors(3))
    await store.upsert_document("s1", "f2", "guide.docx", chunks(2), vectors(2))
    await store.set_active("s1", "f2", False)

    listing = await store.list_documents("s1")

    assert [entry["filename"] for entry in listing] == ["guide.docx", "manual.pdf"]
    by_name = {entry["filename"]: entry for entry in listing}
    assert by_name["manual.pdf"]["chunks"] == 3
    assert by_name["manual.pdf"]["active"] is True
    assert by_name["guide.docx"]["active"] is False


# -- ranking ----------------------------------------------------------------

@pytest.mark.anyio
async def test_search_orders_by_similarity(store):
    """Cosine distance: the chunk whose vector points along the query axis
    must come first."""
    aligned = [1.0, 0.0, 0.0, 0.0]
    orthogonal = [0.0, 1.0, 0.0, 0.0]
    await store.upsert_document(
        "s1", "f1", "a.pdf",
        [Chunk(text="near", location=None, index=0),
         Chunk(text="far", location=None, index=1)],
        [aligned, orthogonal],
    )

    hits = await store.search("s1", aligned, limit=2)
    assert [hit.text for hit in hits] == ["near", "far"]
    assert hits[0].score > hits[1].score


@pytest.mark.anyio
async def test_search_respects_the_limit(store):
    await store.upsert_document("s1", "f1", "a.pdf", chunks(10), vectors(10))
    assert len(await store.search("s1", [1.0, 0.0, 0.0, 0.0], limit=4)) == 4
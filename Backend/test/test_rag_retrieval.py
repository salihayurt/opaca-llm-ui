"""
Tests for src/rag/retrieval.py.

Run against in-memory Qdrant with a deterministic fake embedder, so the whole
suite works with no server and no API key. The fake maps text to vectors by a
fixed rule, which makes dense results predictable enough to assert on.
"""

import pytest
from qdrant_client import AsyncQdrantClient

from src.rag.chunking import Chunk
from src.rag.retrieval import (
    LexicalCache,
    RetrievalConfig,
    Retriever,
    fit_to_budget,
    reciprocal_rank_fusion,
)
from src.rag.store import DocumentStore, StoredChunk

VECTOR_SIZE = 8


@pytest.fixture
def anyio_backend():
    """Pin to asyncio; see test_rag_store.py."""
    return "asyncio"


class FakeEmbedder:
    """Deterministic bag-of-characters embedding.

    Text sharing characters lands nearby, so semantic-ish behaviour can be
    simulated without a model, and identical text always yields an identical
    vector.
    """

    def __init__(self, size: int = VECTOR_SIZE, fail: bool = False):
        self.size = size
        self.fail = fail

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for character in text.lower():
            if character.isalnum():
                vector[ord(character) % self.size] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    async def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    async def embed_query(self, text):
        if self.fail:
            raise RuntimeError("embedding backend down")
        return self._vector(text)

    async def dimensions(self):
        return self.size


@pytest.fixture
async def store():
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
        client = AsyncQdrantClient(":memory:")
        yield DocumentStore(client, vector_size=VECTOR_SIZE)
        await client.close()


async def index(store, session, file_id, filename, texts):
    embedder = FakeEmbedder()
    chunks = [Chunk(text=text, location=f"page {i + 1}", index=i)
              for i, text in enumerate(texts)]
    vectors = await embedder.embed_documents(texts)
    await store.upsert_document(session, file_id, filename, chunks, vectors)


def stored(file_id: str, chunk_index: int, text: str = "x") -> StoredChunk:
    return StoredChunk(text=text, file_id=file_id, filename="f",
                       chunk_index=chunk_index, location=None, score=0.0)


# -- fusion -----------------------------------------------------------------

def test_fusion_promotes_chunks_found_by_both_retrievers():
    """Agreement between the two retrievers is what hybrid search is for."""
    a, b, c = stored("f", 1), stored("f", 2), stored("f", 3)
    fused = reciprocal_rank_fusion([[a, b], [c, a]])
    assert fused[0] is a


def test_fusion_deduplicates():
    a, b = stored("f", 1), stored("f", 2)
    fused = reciprocal_rank_fusion([[a, b], [b, a]])
    assert len(fused) == 2


def test_fusion_combines_by_rank_not_by_score():
    """Cosine similarity and BM25 scores are on incomparable scales;
    normalising them against each other needs distributional assumptions that
    do not hold across corpora."""
    high = stored("f", 1, "high")
    high = StoredChunk(**{**high.__dict__, "score": 999.0})
    low = StoredChunk(**{**stored("f", 2, "low").__dict__, "score": 0.001})
    fused = reciprocal_rank_fusion([[low, high]])
    assert fused[0] is low


def test_fusion_is_deterministic():
    chunks = [stored("f", i) for i in range(5)]
    first = reciprocal_rank_fusion([chunks[:3], chunks[2:]])
    second = reciprocal_rank_fusion([chunks[:3], chunks[2:]])
    assert [c.chunk_index for c in first] == [c.chunk_index for c in second]


def test_fusion_of_nothing_is_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# -- budget -----------------------------------------------------------------

def test_budget_keeps_a_prefix_not_a_subset():
    """Skipping an oversized chunk to fit a lower-ranked one silently
    overrides the retriever's ranking."""
    chunks = [stored("f", 0, "a " * 10), stored("f", 1, "b " * 400), stored("f", 2, "c")]
    kept = fit_to_budget(chunks, 100, count_tokens=lambda t: len(t.split()))
    assert [c.chunk_index for c in kept] == [0]


def test_budget_always_keeps_the_best_chunk():
    """Returning nothing because the single best passage is long is a worse
    answer than returning it and letting the generator truncate."""
    chunks = [stored("f", 0, "word " * 500)]
    kept = fit_to_budget(chunks, 10, count_tokens=lambda t: len(t.split()))
    assert len(kept) == 1


def test_budget_fills_up_to_the_limit():
    chunks = [stored("f", i, "word " * 10) for i in range(10)]
    kept = fit_to_budget(chunks, 35, count_tokens=lambda t: len(t.split()))
    assert len(kept) == 3


def test_budget_of_nothing_is_empty():
    assert fit_to_budget([], 100) == []


# -- retrieval end to end ---------------------------------------------------

@pytest.mark.anyio
async def test_retrieves_relevant_chunks(store, anyio_backend):
    await index(store, "s1", "f1", "manual.pdf", [
        "The thermostat is reset by holding the button for three seconds.",
        "Filters must be replaced every six months.",
        "The canteen serves lunch until two.",
    ])
    retriever = Retriever(store, FakeEmbedder())
    result = await retriever.retrieve("s1", "how do I reset the thermostat")

    assert result
    assert "thermostat" in result.chunks[0].text
    assert not result.degraded


@pytest.mark.anyio
async def test_empty_query_returns_nothing(store, anyio_backend):
    await index(store, "s1", "f1", "a.pdf", ["some text"])
    result = await Retriever(store, FakeEmbedder()).retrieve("s1", "   ")
    assert not result


@pytest.mark.anyio
async def test_unindexed_session_returns_nothing(store, anyio_backend):
    result = await Retriever(store, FakeEmbedder()).retrieve("empty", "anything")
    assert not result.chunks


@pytest.mark.anyio
async def test_respects_the_limit(store, anyio_backend):
    await index(store, "s1", "f1", "a.pdf", [f"filter passage {i}" for i in range(20)])
    result = await Retriever(store, FakeEmbedder()).retrieve("s1", "filter", limit=5)
    assert len(result.chunks) == 5


@pytest.mark.anyio
async def test_inactive_documents_are_not_retrieved(store, anyio_backend):
    await index(store, "s1", "f1", "a.pdf", ["thermostat reset procedure"])
    await index(store, "s1", "f2", "b.pdf", ["thermostat calibration guide"])
    await store.set_active("s1", "f2", False)

    retriever = Retriever(store, FakeEmbedder())
    result = await retriever.retrieve("s1", "thermostat")
    assert {chunk.filename for chunk in result.chunks} == {"a.pdf"}


# -- what hybrid retrieval is for -------------------------------------------

@pytest.mark.anyio
async def test_lexical_stage_finds_an_exact_code(store, anyio_backend):
    """The query dense retrieval handles worst: an identifier read off a
    display, which carries no distributional meaning."""
    await index(store, "s1", "f1", "manual.pdf", [
        "General guidance on climate control systems and their operation.",
        "Fehler 22 bedeutet, dass der Filter gewechselt werden muss.",
        "An overview of the available operating modes for this unit.",
    ])
    retriever = Retriever(store, FakeEmbedder(), config=RetrievalConfig(hybrid=True))
    result = await retriever.retrieve("s1", "Fehler 22")

    assert "lexical" in result.stages
    assert "Fehler 22" in result.chunks[0].text


@pytest.mark.anyio
async def test_hybrid_can_be_switched_off_for_ablation(store, anyio_backend):
    await index(store, "s1", "f1", "a.pdf", ["thermostat reset"])
    retriever = Retriever(store, FakeEmbedder(), config=RetrievalConfig(hybrid=False))
    result = await retriever.retrieve("s1", "thermostat")
    assert result.stages == ["dense"]


@pytest.mark.anyio
async def test_both_stages_run_when_hybrid_is_on(store, anyio_backend):
    await index(store, "s1", "f1", "a.pdf", ["thermostat reset procedure"])
    result = await Retriever(store, FakeEmbedder()).retrieve("s1", "thermostat")
    assert "dense" in result.stages and "lexical" in result.stages
    assert "fusion" in result.stages


# -- degradation is visible -------------------------------------------------

@pytest.mark.anyio
async def test_dense_failure_degrades_rather_than_fails(store, anyio_backend):
    """Lexical search can still answer, so the query proceeds -- but the
    evaluation must be able to tell that it did."""
    await index(store, "s1", "f1", "a.pdf", ["thermostat reset procedure"])
    retriever = Retriever(store, FakeEmbedder(fail=True))
    result = await retriever.retrieve("s1", "thermostat")

    assert result.chunks
    assert "dense" in result.degraded
    assert "dense" not in result.stages


@pytest.mark.anyio
async def test_rerank_failure_keeps_the_fused_order(store, anyio_backend):
    class BrokenReranker:
        async def rerank(self, query, chunks):
            raise RuntimeError("model not loaded")

    await index(store, "s1", "f1", "a.pdf", [f"filter passage {i}" for i in range(20)])
    retriever = Retriever(
        store, FakeEmbedder(),
        config=RetrievalConfig(rerank=True, min_candidates_to_rerank=1),
        reranker=BrokenReranker(),
    )
    result = await retriever.retrieve("s1", "filter")

    assert result.chunks
    assert "rerank:error" in result.degraded
    assert "rerank" not in result.stages


@pytest.mark.anyio
async def test_a_reranker_that_drops_candidates_is_refused(store, anyio_backend):
    """Dropping candidates changes what the generator can see, which is a
    different intervention from reordering them."""
    class TruncatingReranker:
        async def rerank(self, query, chunks):
            return chunks[:3]

    await index(store, "s1", "f1", "a.pdf", [f"filter passage {i}" for i in range(20)])
    retriever = Retriever(
        store, FakeEmbedder(),
        config=RetrievalConfig(rerank=True, min_candidates_to_rerank=1),
        reranker=TruncatingReranker(),
    )
    result = await retriever.retrieve("s1", "filter")
    assert "rerank:size-mismatch" in result.degraded


# -- adaptive reranking -----------------------------------------------------

@pytest.mark.anyio
async def test_reranking_is_skipped_on_a_small_candidate_pool(store, anyio_backend):
    """A cross-encoder exists to filter a large pool; on a handful it spends
    its latency budget rearranging results the generator would see anyway."""
    called = False

    class Reranker:
        async def rerank(self, query, chunks):
            nonlocal called
            called = True
            return chunks

    await index(store, "s1", "f1", "a.pdf", ["filter one", "filter two"])
    retriever = Retriever(
        store, FakeEmbedder(),
        config=RetrievalConfig(rerank=True, min_candidates_to_rerank=12),
        reranker=Reranker(),
    )
    result = await retriever.retrieve("s1", "filter")

    assert not called
    assert "rerank:too-few-candidates" in result.degraded


@pytest.mark.anyio
async def test_reranking_runs_on_a_large_enough_pool(store, anyio_backend):
    class ReverseReranker:
        async def rerank(self, query, chunks):
            return list(reversed(chunks))

    await index(store, "s1", "f1", "a.pdf", [f"filter passage {i}" for i in range(20)])
    retriever = Retriever(
        store, FakeEmbedder(),
        config=RetrievalConfig(rerank=True, min_candidates_to_rerank=5),
        reranker=ReverseReranker(),
    )
    result = await retriever.retrieve("s1", "filter")

    assert "rerank" in result.stages
    assert not result.degraded


@pytest.mark.anyio
async def test_reranker_is_ignored_when_the_flag_is_off(store, anyio_backend):
    called = False

    class Reranker:
        async def rerank(self, query, chunks):
            nonlocal called
            called = True
            return chunks

    await index(store, "s1", "f1", "a.pdf", [f"filter passage {i}" for i in range(20)])
    retriever = Retriever(
        store, FakeEmbedder(),
        config=RetrievalConfig(rerank=False),
        reranker=Reranker(),
    )
    await retriever.retrieve("s1", "filter")
    assert not called


# -- lexical cache ----------------------------------------------------------

@pytest.mark.anyio
async def test_new_documents_are_searchable_after_invalidation(store, anyio_backend):
    """Lexical search reads a cached index; forgetting to invalidate it makes
    a newly indexed document lexically invisible while dense search finds it,
    which reads as retrieval being flaky."""
    retriever = Retriever(store, FakeEmbedder())
    await index(store, "s1", "f1", "a.pdf", ["thermostat reset"])
    await retriever.retrieve("s1", "thermostat")

    await index(store, "s1", "f2", "b.pdf", ["Fehler 22 filter change"])
    retriever.invalidate("s1")

    result = await retriever.retrieve("s1", "Fehler 22")
    assert any("Fehler 22" in chunk.text for chunk in result.chunks)


@pytest.mark.anyio
async def test_cache_is_scoped_per_session(store, anyio_backend):
    cache = LexicalCache()
    await index(store, "alice", "f1", "a.pdf", ["thermostat reset"])
    await index(store, "bob", "f2", "b.pdf", ["thermostat calibration"])

    retriever = Retriever(store, FakeEmbedder(), cache=cache)
    alice = await retriever.retrieve("alice", "thermostat")
    bob = await retriever.retrieve("bob", "thermostat")

    assert {c.filename for c in alice.chunks} == {"a.pdf"}
    assert {c.filename for c in bob.chunks} == {"b.pdf"}


@pytest.mark.anyio
async def test_invalidating_one_session_leaves_others_cached(store, anyio_backend):
    cache = LexicalCache()
    await index(store, "s1", "f1", "a.pdf", ["thermostat"])
    await index(store, "s2", "f2", "b.pdf", ["thermostat"])
    retriever = Retriever(store, FakeEmbedder(), cache=cache)
    await retriever.retrieve("s1", "thermostat")
    await retriever.retrieve("s2", "thermostat")

    cache.invalidate("s1")
    assert "s1" not in cache._entries
    assert "s2" in cache._entries
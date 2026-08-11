"""
Tests for src/rag/embedding.py.

The LiteLLM call is stubbed throughout: embedding is the one part of the
pipeline that costs money and needs credentials, and a suite that cannot run
without an API key cannot run in CI. What is tested here is everything around
the call -- batching, ordering, retries, dimension handling -- which is where
the bugs actually live.
"""

import asyncio

import litellm
import pytest

from src.rag.embedding import (
    EmbeddingError,
    LiteLLMEmbedder,
)


@pytest.fixture
def anyio_backend():
    """Pin to asyncio; see test_rag_store.py for why."""
    return "asyncio"


class FakeProvider:
    """Records calls and returns deterministic vectors derived from the text.

    Deriving the vector from the input is what makes ordering testable: if a
    vector can be traced back to its text, a mix-up is visible.
    """

    def __init__(self, size: int = 4, fail_times: int = 0, error=None):
        self.size = size
        self.calls: list[list[str]] = []
        self.fail_times = fail_times
        self.error = error or litellm.APIConnectionError(
            message="boom", model="m", llm_provider="p"
        )

    async def __call__(self, **kwargs):
        texts = kwargs["input"]
        self.calls.append(list(texts))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        return {
            "data": [
                {"embedding": [float(len(text))] + [0.0] * (self.size - 1)}
                for text in texts
            ]
        }


def embedder(provider, **kwargs) -> LiteLLMEmbedder:
    kwargs.setdefault("backoff_seconds", 0.0)
    return LiteLLMEmbedder(model="test-model", request=provider, **kwargs)


# -- basics -----------------------------------------------------------------

@pytest.mark.anyio
async def test_embeds_documents(anyio_backend):
    provider = FakeProvider()
    vectors = await embedder(provider).embed_documents(["one", "three"])
    assert len(vectors) == 2
    assert all(len(vector) == 4 for vector in vectors)


@pytest.mark.anyio
async def test_empty_input_makes_no_call(anyio_backend):
    """Indexing a document that produced no chunks should cost nothing."""
    provider = FakeProvider()
    assert await embedder(provider).embed_documents([]) == []
    assert provider.calls == []


@pytest.mark.anyio
async def test_blank_text_is_rejected_not_embedded(anyio_backend):
    """Providers disagree on what to return for empty input, and a zero vector
    would match every query equally -- indistinguishable from bad retrieval
    rather than from a bug."""
    provider = FakeProvider()
    with pytest.raises(ValueError):
        await embedder(provider).embed_documents(["fine", "   "])
    assert provider.calls == []


@pytest.mark.anyio
async def test_blank_query_is_rejected(anyio_backend):
    with pytest.raises(ValueError):
        await embedder(FakeProvider()).embed_query("  ")


@pytest.mark.anyio
async def test_api_key_is_passed_through_when_set(anyio_backend):
    """Keys are held per session, so they arrive as an argument rather than
    from the environment."""
    seen = {}

    async def provider(**kwargs):
        seen.update(kwargs)
        return {"data": [{"embedding": [1.0, 0.0]}]}

    await LiteLLMEmbedder(model="m", api_key="sk-test", request=provider).embed_query("q")
    assert seen["api_key"] == "sk-test"


@pytest.mark.anyio
async def test_api_key_is_omitted_when_unset(anyio_backend):
    """Passing api_key=None would override a key LiteLLM found in the
    environment."""
    seen = {}

    async def provider(**kwargs):
        seen.update(kwargs)
        return {"data": [{"embedding": [1.0, 0.0]}]}

    await LiteLLMEmbedder(model="m", request=provider).embed_query("q")
    assert "api_key" not in seen


# -- batching and ordering --------------------------------------------------

@pytest.mark.anyio
async def test_large_input_is_split_into_batches(anyio_backend):
    """The FAISS implementation sent every chunk of a document in one call,
    which fails past the provider's per-request cap."""
    provider = FakeProvider()
    texts = [f"text number {i}" for i in range(250)]
    await embedder(provider, max_batch_items=100).embed_documents(texts)
    assert [len(call) for call in provider.calls] == [100, 100, 50]


@pytest.mark.anyio
async def test_batches_also_respect_a_token_limit(anyio_backend):
    """Item count alone is not enough: a few very long chunks can exceed the
    per-request token cap while staying under the item cap."""
    provider = FakeProvider()
    texts = ["word " * 500 for _ in range(6)]
    await embedder(provider, max_batch_items=100, max_batch_tokens=1000).embed_documents(texts)
    assert len(provider.calls) > 1


@pytest.mark.anyio
async def test_order_survives_batching(anyio_backend):
    """Callers zip vectors against chunks. A misalignment would attach every
    chunk to the wrong vector, and retrieval would still return results --
    just the wrong ones."""
    provider = FakeProvider()
    texts = ["a" * (i + 1) for i in range(50)]
    vectors = await embedder(provider, max_batch_items=7).embed_documents(texts)

    # The stub encodes text length in the first component.
    assert [vector[0] for vector in vectors] == [float(len(text)) for text in texts]


@pytest.mark.anyio
async def test_order_survives_batches_completing_out_of_order(anyio_backend):
    """Batches run concurrently, so a later one can finish first."""
    call_count = 0

    async def provider(**kwargs):
        nonlocal call_count
        call_count += 1
        # Make the first batch the slowest.
        await asyncio.sleep(0.02 if call_count == 1 else 0.0)
        return {"data": [{"embedding": [float(len(t)), 0.0]} for t in kwargs["input"]]}

    texts = ["a" * (i + 1) for i in range(20)]
    vectors = await LiteLLMEmbedder(
        model="m", request=provider, max_batch_items=5
    ).embed_documents(texts)

    assert [vector[0] for vector in vectors] == [float(len(text)) for text in texts]


@pytest.mark.anyio
async def test_short_input_uses_a_single_call(anyio_backend):
    provider = FakeProvider()
    await embedder(provider).embed_documents(["a", "b", "c"])
    assert len(provider.calls) == 1


# -- retries ----------------------------------------------------------------

@pytest.mark.anyio
async def test_transient_failures_are_retried(anyio_backend):
    provider = FakeProvider(fail_times=2)
    vectors = await embedder(provider, max_attempts=3).embed_documents(["text"])
    assert len(vectors) == 1
    assert len(provider.calls) == 3


@pytest.mark.anyio
async def test_retries_are_bounded(anyio_backend):
    provider = FakeProvider(fail_times=99)
    with pytest.raises(EmbeddingError):
        await embedder(provider, max_attempts=3).embed_documents(["text"])
    assert len(provider.calls) == 3


@pytest.mark.anyio
async def test_authentication_errors_are_not_retried(anyio_backend):
    """A missing or wrong key will be just as wrong on the next attempt;
    retrying only delays the error the user needs to see."""
    provider = FakeProvider(
        fail_times=99,
        error=litellm.AuthenticationError(message="bad key", model="m", llm_provider="p"),
    )
    with pytest.raises(EmbeddingError):
        await embedder(provider, max_attempts=3).embed_documents(["text"])
    assert len(provider.calls) == 1


@pytest.mark.anyio
async def test_bad_request_is_not_retried(anyio_backend):
    provider = FakeProvider(
        fail_times=99,
        error=litellm.BadRequestError(message="nope", model="m", llm_provider="p"),
    )
    with pytest.raises(EmbeddingError):
        await embedder(provider, max_attempts=2).embed_documents(["text"])
    assert len(provider.calls) == 1


# -- response shapes --------------------------------------------------------

@pytest.mark.anyio
async def test_object_style_responses_are_understood(anyio_backend):
    """LiteLLM returns dict-like responses for some providers and object-like
    for others."""
    class Item:
        embedding = [1.0, 2.0]

    class Response:
        data = [Item()]

    async def provider(**kwargs):
        return Response()

    vector = await LiteLLMEmbedder(model="m", request=provider).embed_query("q")
    assert vector == [1.0, 2.0]


@pytest.mark.anyio
async def test_short_response_is_an_error_not_silent_truncation(anyio_backend):
    """Fewer vectors than inputs would otherwise leave some chunks with an
    empty vector, which the store would then reject one at a time."""
    async def provider(**kwargs):
        return {"data": [{"embedding": [1.0, 0.0]}]}

    with pytest.raises(EmbeddingError):
        await LiteLLMEmbedder(model="m", request=provider).embed_documents(["a", "b"])


# -- dimensions -------------------------------------------------------------

@pytest.mark.anyio
async def test_known_model_dimensions_need_no_call(anyio_backend):
    """The collection has to be created before anything is indexed, so paying
    for a probe on every startup would be waste."""
    provider = FakeProvider()
    known = LiteLLMEmbedder(model="text-embedding-3-small", request=provider)
    assert await known.dimensions() == 1536
    assert provider.calls == []


@pytest.mark.anyio
async def test_unknown_model_dimensions_are_probed_once(anyio_backend):
    provider = FakeProvider(size=7)
    unknown = embedder(provider)
    assert await unknown.dimensions() == 7
    assert await unknown.dimensions() == 7
    assert len(provider.calls) == 1


@pytest.mark.anyio
async def test_changing_dimensions_mid_process_is_an_error(anyio_backend):
    """A collection is created with a fixed vector size. Silently accepting a
    different one would fail later, chunk by chunk, with a confusing message."""
    sizes = iter([4, 8])

    async def provider(**kwargs):
        size = next(sizes)
        return {"data": [{"embedding": [1.0] * size} for _ in kwargs["input"]]}

    instance = LiteLLMEmbedder(model="m", request=provider)
    await instance.embed_query("first")
    with pytest.raises(EmbeddingError) as excinfo:
        await instance.embed_query("second")
    assert "rebuilt" in str(excinfo.value)


# -- concurrency ------------------------------------------------------------

@pytest.mark.anyio
async def test_concurrent_batches_are_capped(anyio_backend):
    """Unbounded concurrency on a large document is a self-inflicted rate
    limit."""
    in_flight = 0
    peak = 0

    async def provider(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return {"data": [{"embedding": [1.0, 0.0]} for _ in kwargs["input"]]}

    texts = [f"text {i}" for i in range(60)]
    await LiteLLMEmbedder(
        model="m", request=provider, max_batch_items=5, max_concurrent=3
    ).embed_documents(texts)

    assert peak <= 3
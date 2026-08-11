"""
Turning text into vectors.

Routed through LiteLLM rather than the OpenAI SDK directly. That closes the
`# TODO change to LiteLLM` left in `vector_storage.py`, and it matters beyond
tidiness: SAGE is a publicly funded, EU-based project, so whether document
text may leave GT-ARC's infrastructure for a US embedding API is an
institutional decision, not one this module should make. Going through
LiteLLM means switching to a self-hosted BGE-M3 or Qwen3-Embedding later is
one configuration value rather than a rewrite. It also matches how the rest
of the backend already talks to models (`abstract_method.py`, `file_utils.py`).

Three things this module is careful about, all of which the FAISS
implementation got wrong:

1. **It never blocks the event loop.** `vector_storage.py` calls the
   synchronous `openai.embeddings.create` from inside an async backend, which
   stalls every other request for the duration of indexing -- tens of seconds
   for a large document.

2. **It batches, with a limit.** The old code passed every chunk of a
   document in one call. Beyond a few thousand chunks that exceeds the
   provider's per-request cap and fails outright, having already spent the
   time to get there.

3. **It preserves order.** Batches complete concurrently and out of order,
   but the returned vectors line up with the input texts, because callers zip
   them against chunks and a silent misalignment would attach every chunk to
   the wrong vector -- retrieval would still "work", just wrongly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Protocol

import litellm

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"

# Known output sizes, used to create a collection before anything is embedded.
# Unknown models are probed once instead (see LiteLLMEmbedder.dimensions).
KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "BAAI/bge-m3": 1024,
    "BAAI/bge-base-en-v1.5": 768,
}

# Providers cap how many inputs and how many tokens one request may carry.
# These sit well inside OpenAI's limits (2048 inputs) and leave room for
# self-hosted servers, which are usually configured far lower.
MAX_BATCH_ITEMS = 128
MAX_BATCH_TOKENS = 100_000

MAX_CONCURRENT_BATCHES = 5

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 1.0

# Retrying these only wastes time and money: the request is wrong, or the
# credentials are, and it will be equally wrong on the next attempt.
NON_RETRYABLE = (
    litellm.AuthenticationError,
    litellm.PermissionDeniedError,
    litellm.BadRequestError,
    litellm.NotFoundError,
    litellm.ContextWindowExceededError,
)


class Embedder(Protocol):
    """What the rest of the pipeline needs from an embedding backend."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def dimensions(self) -> int: ...


class EmbeddingError(Exception):
    """Raised when embedding fails after exhausting retries."""


def _estimate_tokens(text: str) -> int:
    """Rough token count for batch sizing only, not for budgeting."""
    return max(len(text.split()), len(text) // 4)


async def _default_request(**kwargs):
    return await litellm.aembedding(**kwargs)


class LiteLLMEmbedder:
    """Embeds text through LiteLLM, batched and with bounded concurrency.

    `request` exists so tests can supply a stub: embedding is the one place in
    the pipeline that costs money and needs credentials, and a suite that
    cannot run without an API key will not run in CI.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        request: Callable[..., Awaitable] = _default_request,
        max_batch_items: int = MAX_BATCH_ITEMS,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
        max_concurrent: int = MAX_CONCURRENT_BATCHES,
        max_attempts: int = MAX_ATTEMPTS,
        backoff_seconds: float = BACKOFF_SECONDS,
    ):
        self.model = model
        self.api_key = api_key
        self._request = request
        self.max_batch_items = max_batch_items
        self.max_batch_tokens = max_batch_tokens
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._dimensions: int | None = KNOWN_DIMENSIONS.get(model)

    # -- public API ----------------------------------------------------------

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, returning vectors in the same order.

        Empty or whitespace-only texts are rejected rather than embedded.
        Providers differ on what they return for them, and a caller that gets
        back a zero vector would store a chunk matching every query equally --
        which looks like retrieval working badly rather than like a bug.
        """
        if not texts:
            return []

        for position, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"text at position {position} is empty; nothing to embed")

        batches = list(self._batch(texts))
        results = await asyncio.gather(*(self._embed_batch(batch) for _, batch in batches))

        vectors: list[list[float]] = [[] for _ in texts]
        for (indices, _), batch_vectors in zip(batches, results):
            if len(batch_vectors) != len(indices):
                raise EmbeddingError(
                    f"provider returned {len(batch_vectors)} vectors for "
                    f"{len(indices)} inputs"
                )
            for index, vector in zip(indices, batch_vectors):
                vectors[index] = vector

        self._remember_dimensions(vectors[0])
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Separate from embed_documents because instruction-tuned embedding
        models -- BGE-M3 and the E5 family among them -- expect queries and
        passages to be prefixed differently, and using one path for both
        silently costs retrieval quality on those models.
        """
        if not text or not text.strip():
            raise ValueError("query is empty; nothing to embed")
        vectors = await self._embed_batch([text])
        self._remember_dimensions(vectors[0])
        return vectors[0]

    async def dimensions(self) -> int:
        """The model's vector size, needed before a collection can be created.

        Looked up for known models, and otherwise discovered by embedding one
        short probe. The result is cached, so an unknown model costs a single
        extra call for the lifetime of the process.
        """
        if self._dimensions is None:
            probe = await self._embed_batch(["dimension probe"])
            self._remember_dimensions(probe[0])
        return self._dimensions

    # -- internals -----------------------------------------------------------

    def _batch(self, texts: list[str]):
        """Group texts into requests, respecting both provider limits.

        Yields (original indices, texts) so ordering can be restored after the
        batches finish out of order.
        """
        indices: list[int] = []
        batch: list[str] = []
        tokens = 0

        for index, text in enumerate(texts):
            text_tokens = _estimate_tokens(text)
            too_many_items = len(batch) >= self.max_batch_items
            too_many_tokens = batch and tokens + text_tokens > self.max_batch_tokens
            if too_many_items or too_many_tokens:
                yield indices, batch
                indices, batch, tokens = [], [], 0

            indices.append(index)
            batch.append(text)
            tokens += text_tokens

        if batch:
            yield indices, batch

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with self._semaphore:
            return await self._request_with_retry(texts)

    async def _request_with_retry(self, texts: list[str]) -> list[list[float]]:
        kwargs = {"model": self.model, "input": texts}
        if self.api_key:
            kwargs["api_key"] = self.api_key

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._request(**kwargs)
                return _vectors_from(response)
            except NON_RETRYABLE as error:
                raise EmbeddingError(f"embedding rejected by {self.model}: {error}") from error
            except Exception as error:
                last_error = error
                if attempt == self.max_attempts:
                    break
                delay = self.backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Embedding attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt, self.max_attempts, error, delay,
                )
                await asyncio.sleep(delay)

        raise EmbeddingError(
            f"embedding failed after {self.max_attempts} attempts: {last_error}"
        ) from last_error

    def _remember_dimensions(self, vector: list[float]) -> None:
        """Record the vector size, and refuse to change it mid-process.

        A collection is created with a fixed size. If the configured model
        changed under a running process, later chunks would be rejected by the
        store one at a time with a confusing message; failing here names the
        cause instead.
        """
        size = len(vector)
        if self._dimensions is None:
            self._dimensions = size
        elif self._dimensions != size:
            raise EmbeddingError(
                f"{self.model} returned {size}-dimensional vectors but "
                f"{self._dimensions} were expected. The embedding model "
                "appears to have changed; the index must be rebuilt."
            )


def _vectors_from(response) -> list[list[float]]:
    """Pull embeddings out of a LiteLLM response.

    The response is dict-like for some providers and object-like for others,
    so both shapes are handled rather than assuming one.
    """
    data = response["data"] if isinstance(response, dict) else response.data
    vectors = []
    for item in data:
        vector = item["embedding"] if isinstance(item, dict) else item.embedding
        vectors.append(list(vector))
    return vectors
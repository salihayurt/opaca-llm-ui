"""
Putting retrieval together.

Dense search finds passages that mean the same thing as the query; lexical
search finds passages containing the same tokens. Neither subsumes the other,
which is why this module runs both and fuses the results rather than picking
one.

Every stage is switchable through `RetrievalConfig`, because the point of the
evaluation is to compare configurations against the existing baseline one axis
at a time. A single flag that turns everything on at once would produce one
number and no explanation.

One warning, recorded here because it has already caused a wasted measurement
once: retrieval is deliberately forgiving. If reranking fails, it logs and
continues with the fused order rather than failing the query. That is right in
production and dangerous in measurement, because a silently skipped stage
produces a row identical to the baseline and invites the conclusion "this
feature does nothing". `RetrievalResult.degraded` exists so the harness can
tell a genuine null result from a stage that never ran.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from .embedding import Embedder
from .lexical import BM25Index
from .store import DocumentStore, StoredChunk

logger = logging.getLogger(__name__)

# Reciprocal rank fusion constant. 60 is the value from the original paper and
# the de facto default; it damps the influence of the top ranks enough that one
# retriever's confident mistake cannot dominate the other's correct ordering.
RRF_K = 60

# How many candidates each retriever contributes before fusion.
CANDIDATE_POOL = 30

# Below this many candidates, reranking is skipped. A cross-encoder exists to
# filter a large pool; asked to reorder a handful it spends its latency budget
# to rearrange results the generator would have seen anyway.
MIN_CANDIDATES_TO_RERANK = 12


class Reranker(Protocol):
    """Reorders candidates by relevance to the query.

    Kept as a protocol with no implementation here on purpose. The choice of
    reranker is unresolved: the CPU-viable cross-encoder is English-only, the
    multilingual one needs a GPU we do not have, and the API-based option
    costs roughly 1.5 s per query. That is a decision for the evaluation, and
    pulling in sentence-transformers -- and with it torch -- before the
    decision is made would burden every deployment for a feature that may not
    ship enabled.
    """

    async def rerank(self, query: str, chunks: list[StoredChunk]) -> list[StoredChunk]: ...


@dataclass
class RetrievalConfig:
    """Which stages run. One flag per ablation axis."""

    hybrid: bool = True
    rerank: bool = False
    candidate_pool: int = CANDIDATE_POOL
    min_candidates_to_rerank: int = MIN_CANDIDATES_TO_RERANK
    rrf_k: int = RRF_K


@dataclass
class RetrievalResult:
    """Retrieved chunks plus what actually happened while retrieving them."""

    chunks: list[StoredChunk]
    stages: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.chunks)


def chunk_key(chunk: StoredChunk) -> tuple[str, int]:
    """Identity of a chunk across result lists."""
    return (chunk.file_id, chunk.chunk_index)


def reciprocal_rank_fusion(
    rankings: list[list[StoredChunk]],
    *,
    k: int = RRF_K,
) -> list[StoredChunk]:
    """Merge several ranked lists into one.

    Scores are combined by rank, not by value, which is the property that
    makes this work at all here: a cosine similarity and a BM25 score are on
    incomparable scales, and normalising them against each other requires
    assumptions about their distributions that do not hold across corpora.
    Ranks need no such assumption.

    A chunk retrieved by both retrievers accumulates both contributions, so
    agreement between the two is what promotes a passage -- which is the
    behaviour hybrid retrieval is supposed to have.
    """
    scores: dict[tuple[str, int], float] = {}
    seen: dict[tuple[str, int], StoredChunk] = {}
    first_rank: dict[tuple[str, int], int] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            key = chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in seen:
                seen[key] = chunk
                first_rank[key] = rank

    # Ties broken by best rank achieved, then by identity, so the order is
    # reproducible across runs -- an unstable tie-break would surface in the
    # evaluation as a difference with no cause.
    ordered = sorted(
        scores,
        key=lambda key: (-scores[key], first_rank[key], key),
    )
    return [seen[key] for key in ordered]


def fit_to_budget(
    chunks: list[StoredChunk],
    budget_tokens: int,
    *,
    count_tokens=None,
) -> list[StoredChunk]:
    """Take the longest prefix of `chunks` fitting inside `budget_tokens`.

    A prefix rather than a subset: the ranking is the retriever's judgement,
    and skipping past an oversized chunk to fit a lower-ranked one silently
    overrides it.

    The top-ranked chunk is always kept, even alone exceeding the budget.
    Returning nothing because the single best passage is long would be a worse
    answer than returning it and letting the generator truncate.
    """
    if not chunks:
        return []
    if count_tokens is None:
        from .chunking import estimate_tokens as count_tokens

    kept = [chunks[0]]
    used = count_tokens(chunks[0].text)
    for chunk in chunks[1:]:
        size = count_tokens(chunk.text)
        if used + size > budget_tokens:
            break
        kept.append(chunk)
        used += size
    return kept


class LexicalCache:
    """Per-session BM25 indexes, rebuilt only when the session's chunks change.

    Lexical search needs the whole corpus, so without a cache every query
    scrolls the entire collection and re-tokenises it. Indexing happens rarely
    and queries happen constantly, so the index is built once and invalidated
    on write.
    """

    def __init__(self):
        self._entries: dict[str, tuple[BM25Index, list[StoredChunk]]] = {}

    def invalidate(self, session_id: str) -> None:
        self._entries.pop(session_id, None)

    def clear(self) -> None:
        self._entries.clear()

    async def get(
        self,
        store: DocumentStore,
        session_id: str,
    ) -> tuple[BM25Index, list[StoredChunk]]:
        if session_id not in self._entries:
            chunks = await store.iter_chunks(session_id, active_only=True)
            self._entries[session_id] = (BM25Index.build([c.text for c in chunks]), chunks)
        return self._entries[session_id]


class Retriever:
    """Hybrid retrieval over one session's indexed documents."""

    def __init__(
        self,
        store: DocumentStore,
        embedder: Embedder,
        *,
        config: RetrievalConfig | None = None,
        reranker: Reranker | None = None,
        cache: LexicalCache | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.reranker = reranker
        self.cache = cache or LexicalCache()

    def invalidate(self, session_id: str) -> None:
        """Drop cached lexical state. Call after any change to the session."""
        self.cache.invalidate(session_id)

    async def retrieve(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 10,
    ) -> RetrievalResult:
        """Retrieve the chunks most relevant to `query` within one session."""
        if not query or not query.strip():
            return RetrievalResult(chunks=[])

        result = RetrievalResult(chunks=[])
        rankings: list[list[StoredChunk]] = []

        dense = await self._dense(session_id, query, result)
        if dense:
            rankings.append(dense)
            result.stages.append("dense")

        if self.config.hybrid:
            lexical = await self._lexical(session_id, query, result)
            if lexical:
                rankings.append(lexical)
                result.stages.append("lexical")

        if not rankings:
            return result

        candidates = (
            reciprocal_rank_fusion(rankings, k=self.config.rrf_k)
            if len(rankings) > 1
            else rankings[0]
        )
        if len(rankings) > 1:
            result.stages.append("fusion")

        candidates = await self._maybe_rerank(query, candidates, result)

        result.chunks = candidates[:limit]
        return result

    # -- stages --------------------------------------------------------------

    async def _dense(self, session_id, query, result) -> list[StoredChunk]:
        try:
            vector = await self.embedder.embed_query(query)
        except Exception as error:
            # Without a query vector there is no dense stage at all. Lexical
            # search can still answer, so the query proceeds degraded rather
            # than failing outright.
            logger.warning("Dense retrieval unavailable: %s", error)
            result.degraded.append("dense")
            return []
        return await self.store.search(
            session_id, vector, limit=self.config.candidate_pool, active_only=True
        )

    async def _lexical(self, session_id, query, result) -> list[StoredChunk]:
        try:
            index, chunks = await self.cache.get(self.store, session_id)
        except Exception as error:
            logger.warning("Lexical retrieval unavailable: %s", error)
            result.degraded.append("lexical")
            return []
        return [
            chunks[position]
            for position, _ in index.top_k(query, self.config.candidate_pool)
        ]

    async def _maybe_rerank(self, query, candidates, result) -> list[StoredChunk]:
        if not self.config.rerank or self.reranker is None:
            return candidates

        if len(candidates) < self.config.min_candidates_to_rerank:
            # Not a failure: there is nothing for a cross-encoder to filter.
            # Recorded anyway, so the evaluation can separate "reranking did
            # not help" from "reranking did not run".
            result.degraded.append("rerank:too-few-candidates")
            return candidates

        try:
            reordered = await self.reranker.rerank(query, candidates)
        except Exception as error:
            logger.warning("Reranking failed, keeping fused order: %s", error)
            result.degraded.append("rerank:error")
            return candidates

        if len(reordered) != len(candidates):
            # A reranker that drops candidates changes what the generator can
            # see, which is a different intervention from reordering. Refuse
            # it rather than silently accepting a narrowed pool.
            logger.warning(
                "Reranker returned %d of %d candidates; keeping fused order",
                len(reordered), len(candidates),
            )
            result.degraded.append("rerank:size-mismatch")
            return candidates

        result.stages.append("rerank")
        return reordered
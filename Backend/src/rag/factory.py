"""
Building the RAG service once per process.

Collections are per session, but the Qdrant client and the embedder are not:
opening a connection per session would be wasteful, and the embedder holds
only configuration.

Everything here is written so that RAG being unavailable is an ordinary
state, not a failure. If Qdrant is not running or no embedding model is
configured, `get_rag_service()` returns None, the Documents tool group offers
no tools, and SAGE behaves exactly as it did before RAG existed. A backend
that refuses to start because an optional feature is misconfigured would be a
worse outcome than one that quietly does without it.
"""

from __future__ import annotations

import logging
import os

from .embedding import DEFAULT_MODEL, KNOWN_DIMENSIONS, LiteLLMEmbedder
from .service import RagConfig, RagService
from .store import DocumentStore

logger = logging.getLogger(__name__)

_service: RagService | None = None
_initialised = False


def _enabled() -> bool:
    value = os.environ.get("ENABLE_RAG", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


async def get_rag_service() -> RagService | None:
    """Return the process-wide RAG service, or None if it is unavailable.

    Built on first use rather than at import time, because the vector size
    depends on the embedding model and an unknown model has to be probed --
    which is a network call, and import time is the wrong place for one.
    """
    global _service, _initialised

    if _initialised:
        return _service
    _initialised = True

    if not _enabled():
        logger.info("RAG is disabled by ENABLE_RAG; document tools will not be offered")
        return None

    model = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    try:
        from qdrant_client import AsyncQdrantClient

        embedder = LiteLLMEmbedder(model=model)

        # A model whose size is not known in advance would have to be probed,
        # which costs an API call before anything has been indexed. Known
        # models skip it; unknown ones pay it once.
        if model in KNOWN_DIMENSIONS:
            size = KNOWN_DIMENSIONS[model]
        else:
            size = await embedder.dimensions()

        client = AsyncQdrantClient(url=url)
        await client.get_collections()  # fail here rather than mid-conversation

        _service = RagService(
            DocumentStore(client, vector_size=size),
            embedder,
            config=RagConfig.from_env(),
        )
        logger.info("RAG ready: %s (%d dimensions) at %s", model, size, url)
    except Exception as error:
        # Deliberately broad. Any failure to reach Qdrant, resolve the model,
        # or authenticate means RAG is unavailable -- which SAGE must survive.
        logger.warning("RAG unavailable (%s); document tools will not be offered", error)
        _service = None

    return _service


def reset_rag_service() -> None:
    """Forget the cached service. For tests and for reconfiguration."""
    global _service, _initialised
    _service = None
    _initialised = False
"""
The baseline, run unmodified.

This reproduces `vector_storage.py` -- GT-ARC's existing RAG implementation --
exactly as written, so the comparison measures the pipeline rather than a
tidied-up version of it. Every defect is preserved deliberately:

- one process-wide index with no session isolation
- chunk ids from `i // size` while stepping by `size - overlap`, which
  collide
- a fixed distance cut-off of 0.3 on squared L2, i.e. cosine above 0.85
- 500-word windows split on whitespace, with no boundary awareness
- text-embedding-ada-002
- every chunk of a document sent in one embedding call

Fixing any of these would flatter the baseline and understate what the new
pipeline changes. The point of the number is that it is the number the
existing code produces.

Two things are adapted rather than reproduced, and neither affects retrieval:

- `extract_text` used PyPDF2, which is unmaintained; this uses the same pypdf
  the new pipeline uses. The text extracted is what is being compared, not the
  library that extracted it, and holding the extractor constant isolates the
  retrieval difference.
- `create_embedding` is called through the same async path as the new
  pipeline, so both pay the same network cost per call. The batching
  behaviour -- one call for the whole document -- is preserved.
"""

from __future__ import annotations

import numpy as np


class BaselineStore:
    """vector_storage.VectorStorage, reproduced.

    Kept as a class rather than the original module-level singleton so the
    harness can hold several at once, but the absence of session scoping is
    otherwise intact: one instance holds one flat index for everything added
    to it, exactly as the original does for the whole process.
    """

    MODEL = "text-embedding-ada-002"
    CHUNK_SIZE = 500
    OVERLAP = 50
    DISTANCE_CUTOFF = 0.3

    def __init__(self, embed):
        # `embed` takes a list of strings and returns a list of vectors. The
        # harness supplies it so both pipelines are measured against the same
        # provider and the same failure handling.
        self._embed = embed
        self.chunks: list[tuple[str, int, str]] = []
        self._vectors: np.ndarray | None = None

    def chunk_text(self, filename: str, text: str) -> list[tuple[str, int, str]]:
        """The original sliding-window splitter, unchanged.

        Note `i // size` for the chunk index while the loop steps by
        `size - overlap`: with the original 500/50 that yields
        0, 0, 1, 2, ..., 9, 9, 10 -- a duplicate roughly every tenth chunk.
        Preserved, because a citation built on that index is part of what the
        comparison is about.
        """
        words = text.split()
        chunks = []
        size, overlap = self.CHUNK_SIZE, self.OVERLAP
        for i in range(0, len(words), size - overlap):
            chunk = " ".join(words[i:i + size])
            if chunk:
                chunks.append((filename, i // size, chunk))
        return chunks

    async def add_to_index(self, filename: str, text: str) -> int:
        """Chunk, embed and store, in one embedding call as the original does."""
        chunks = self.chunk_text(filename, text)
        if not chunks:
            return 0
        self.chunks.extend(chunks)

        vectors = np.array(await self._embed([chunk for _, _, chunk in chunks]))
        self._vectors = (
            vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        )
        return len(chunks)

    async def retrieve(self, query: str, k: int = 3) -> list[tuple[float, tuple]]:
        """Dense search with the original fixed distance cut-off.

        FAISS `IndexFlatL2` returns squared L2. ada-002 vectors are unit-norm,
        so d = 2 - 2*cos and the cut-off of 0.3 keeps only cosine above 0.85.
        Reproduced here with numpy rather than FAISS: the arithmetic is
        identical for an exhaustive flat index, and it avoids a dependency
        whose only role would be to compute the same distances more slowly.
        """
        if not self.chunks or self._vectors is None:
            return []

        query_vector = np.array((await self._embed([query]))[0])
        distances = np.sum((self._vectors - query_vector) ** 2, axis=1)

        order = np.argsort(distances)[:k]
        return [
            (float(distances[i]), self.chunks[i])
            for i in order
            if distances[i] < self.DISTANCE_CUTOFF
        ]

    async def retrieve_unfiltered(self, query: str, k: int = 10) -> list[tuple[float, tuple]]:
        """The same search with the cut-off removed.

        Not part of the baseline's behaviour -- it exists so the evaluation can
        separate two questions that the baseline conflates. If it retrieves
        nothing, is that because the ranking failed, or because the ranking was
        fine and the threshold discarded it? Reporting both tells us which,
        and the answer decides whether the new pipeline's min_dense_score
        should be set at all.
        """
        if not self.chunks or self._vectors is None:
            return []

        query_vector = np.array((await self._embed([query]))[0])
        distances = np.sum((self._vectors - query_vector) ** 2, axis=1)
        order = np.argsort(distances)[:k]
        return [(float(distances[i]), self.chunks[i]) for i in order]
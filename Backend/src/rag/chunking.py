"""
Splitting extracted documents into retrieval units.

Chunking is the highest-leverage decision in this pipeline. In the TU Berlin
RAG ablation (see docs/rag_design.md), the chunker mattered more than the
retriever: the best chunker paired with the worst retriever beat the best
retriever paired with the worst chunker. Small chunks won everywhere, and
large parent windows collapsed outright -- `parent1024/256` scored an
all-hops rate of 0.0000 on MultiHop-RAG, because too few of the documents a
question needed ever fit the context budget.

Hence the shape here: small, boundary-aware chunks with a modest overlap.

Two invariants this module is built around, both of which had to be repaired
in earlier implementations:

1. **A chunk never spans two locations.** Segments carry where they came from
   ("page 4", "sheet Budget"). If a chunk mixed text from page 4 and page 5,
   its citation would be a lie. Segment boundaries are therefore hard
   boundaries, including for overlap. Small adjacent segments are merged into
   a group with a combined label instead, so a 40-slide deck does not turn
   into 40 chunks too short to match anything.

2. **Every chunk carries new content, and every chunk index is unique.** A
   trailing window that lies entirely inside the previous chunk's overlap
   adds nothing, yet costs an embedding, and -- because BM25 normalizes for
   length -- scores disproportionately high, letting an empty chunk displace
   real content in the candidate pool. Duplicate indices are equally bad:
   they make a citation ambiguous about which passage it refers to.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..text_extraction import TextSegment

# Defaults, overridable per call and ultimately from configuration.
TARGET_TOKENS = 300
OVERLAP_TOKENS = 50

# A segment shorter than this fraction of the target is merged with the ones
# after it rather than becoming a chunk of its own.
MERGE_BELOW = 0.5

# Cap on how many segments merge into one group, so a combined location label
# stays short enough to show in a citation.
MAX_MERGED_SEGMENTS = 4

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit.

    `index` is the chunk's position in the document, unique and stable for a
    given input, so it can identify a passage in a citation and serve as a
    deterministic id when writing to the vector store -- re-indexing the same
    file then overwrites rather than duplicating.
    """

    text: str
    location: str | None
    index: int


def estimate_tokens(text: str) -> int:
    """Approximate the token count of `text` without a model tokenizer.

    Deliberately not tiktoken. Its BPE table is downloaded on first use, so a
    real tokenizer would make chunk boundaries depend on network access at
    indexing time -- the same document could be split differently on two
    machines, and CI without egress could not run at all. Chunk sizes only
    need to be approximately right; exact accounting belongs at query time,
    where the prompt budget is actually enforced.

    The estimate is `max(words, characters / 4)`, which follows OpenAI's
    rule of thumb of roughly four characters per token while never returning
    fewer tokens than words. It over-estimates slightly for long German
    compounds, which is the safe direction: chunks come out a little smaller
    than the target rather than overflowing it.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return max(len(stripped.split()), math.ceil(len(stripped) / 4))


def _combine_locations(locations: list[str | None]) -> str | None:
    """Build one label for a group of merged segments.

    Distinct labels are joined rather than summarised, so the result stays
    truthful for any format: "page 3, page 4" is unambiguous, whereas
    "pages 3-4" would have to assume the labels are numeric and contiguous.
    """
    seen = [location for location in dict.fromkeys(locations) if location]
    if not seen:
        return None
    return ", ".join(seen)


def _split_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in _PARAGRAPH_SPLIT.split(text)]
    return [part for part in parts if part]


def _split_oversized(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    count_tokens,
) -> list[str]:
    """Hard-split a single paragraph that exceeds the target on its own.

    Reached by PDF pages extracted as one unbroken block and by long tables.
    Splitting is by words, since there is no smaller structural boundary left
    to respect.
    """
    words = text.split()
    if not words:
        return []

    # Convert token budgets to word counts using this text's own ratio, so a
    # dense German paragraph yields fewer words per window than a sparse
    # English one instead of both being cut at the same word count.
    total_tokens = count_tokens(text)
    tokens_per_word = max(total_tokens / len(words), 1e-6)
    window = max(int(target_tokens / tokens_per_word), 1)
    overlap = min(int(overlap_tokens / tokens_per_word), window - 1) if window > 1 else 0
    step = max(window - overlap, 1)

    pieces = []
    for start in range(0, len(words), step):
        # Drop a trailing window that lies entirely inside the previous
        # window's overlap: it repeats content already indexed, wastes an
        # embedding, and BM25's length normalization would score the
        # resulting stub high enough to displace real content.
        if start > 0 and start + window >= len(words) and len(words) - start <= overlap:
            break
        pieces.append(" ".join(words[start:start + window]))
    return pieces


def _group_segments(
    segments: list[TextSegment],
    target_tokens: int,
    count_tokens,
) -> list[tuple[str, str | None]]:
    """Merge segments that are too short to stand alone.

    Presentation slides and short spreadsheet sheets routinely produce
    segments of a few dozen tokens. Indexed individually they are too small
    to carry enough context to match a query, so consecutive short segments
    are joined and their labels combined.
    """
    threshold = target_tokens * MERGE_BELOW
    groups: list[tuple[str, str | None]] = []

    buffer_texts: list[str] = []
    buffer_locations: list[str | None] = []
    buffer_tokens = 0

    def flush():
        nonlocal buffer_tokens
        if buffer_texts:
            groups.append(("\n\n".join(buffer_texts), _combine_locations(buffer_locations)))
        buffer_texts.clear()
        buffer_locations.clear()
        buffer_tokens = 0

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        tokens = count_tokens(text)
        if tokens >= threshold:
            flush()
            groups.append((text, segment.location))
            continue

        buffer_texts.append(text)
        buffer_locations.append(segment.location)
        buffer_tokens += tokens
        if buffer_tokens >= threshold or len(buffer_texts) >= MAX_MERGED_SEGMENTS:
            flush()

    flush()
    return groups


def _chunk_group(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    count_tokens,
) -> list[str]:
    """Pack one group's paragraphs into chunks of about `target_tokens`."""
    pieces: list[str] = []
    for paragraph in _split_paragraphs(text):
        if count_tokens(paragraph) > target_tokens:
            pieces.extend(_split_oversized(paragraph, target_tokens, overlap_tokens, count_tokens))
        else:
            pieces.append(paragraph)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = count_tokens(piece)
        if current and current_tokens + piece_tokens > target_tokens:
            chunks.append("\n\n".join(current))
            # Carry the tail of the finished chunk into the next one so an
            # answer straddling the boundary is retrievable from either side.
            tail: list[str] = []
            tail_tokens = 0
            for previous in reversed(current):
                previous_tokens = count_tokens(previous)
                if tail_tokens + previous_tokens > overlap_tokens:
                    break
                tail.insert(0, previous)
                tail_tokens += previous_tokens
            current = tail
            current_tokens = tail_tokens

        current.append(piece)
        current_tokens += piece_tokens

    if current:
        candidate = "\n\n".join(current)
        # Same guard as in _split_oversized, one level up: if the final chunk
        # is nothing but the overlap carried from its predecessor, it holds no
        # new content and must not be emitted.
        if chunks and candidate and candidate in chunks[-1]:
            return chunks
        chunks.append(candidate)

    return [chunk for chunk in chunks if chunk.strip()]


def chunk_segments(
    segments: list[TextSegment],
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    count_tokens=estimate_tokens,
) -> list[Chunk]:
    """Split extracted segments into overlapping, located retrieval units.

    Chunks are numbered consecutively across the whole document, so `index`
    identifies a passage uniquely and reproducibly.
    """
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")

    chunks: list[Chunk] = []
    for text, location in _group_segments(segments, target_tokens, count_tokens):
        for piece in _chunk_group(text, target_tokens, overlap_tokens, count_tokens):
            chunks.append(Chunk(text=piece, location=location, index=len(chunks)))
    return chunks


def chunk_text(
    text: str,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    count_tokens=estimate_tokens,
) -> list[str]:
    """Chunk a plain string, for callers with no segment structure."""
    chunks = chunk_segments(
        [TextSegment(text=text)],
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        count_tokens=count_tokens,
    )
    return [chunk.text for chunk in chunks]
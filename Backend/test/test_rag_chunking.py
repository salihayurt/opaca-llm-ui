"""
Tests for src/rag/chunking.py.

Pure functions, no I/O: this suite runs in milliseconds and needs neither a
vector store nor an API key. That matters because chunking is where the
ablation says most of the retrieval quality is decided, so it should be the
fastest thing to re-check after a change.

Token counts use the module's own estimator unless a test needs exact
arithmetic, in which case a word-counting stub is injected so the expected
numbers can be reasoned about directly.
"""

import pytest

from src.rag.chunking import (
    Chunk,
    chunk_segments,
    chunk_text,
    estimate_tokens,
)
from src.text_extraction import TextSegment


def words(count: int, prefix: str = "w") -> str:
    """A paragraph of `count` distinct words, so content loss is detectable."""
    return " ".join(f"{prefix}{i}" for i in range(count))


def count_words(text: str) -> int:
    """Token counter stub: one token per word, for exact expectations."""
    return len(text.split())


# -- estimator --------------------------------------------------------------

def test_estimate_tokens_is_zero_for_empty_input():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   \n\n ") == 0


def test_estimate_tokens_never_undercounts_words():
    """Under-counting would let a chunk exceed the target and overflow the
    prompt budget it was sized for."""
    for text in ["a b c d e", "one", "x " * 50]:
        assert estimate_tokens(text) >= len(text.split())


def test_estimate_tokens_grows_with_german_compounds():
    """Long compounds cost more tokens than their word count suggests; the
    character-based term is what catches that."""
    assert estimate_tokens("Raumtemperaturregler Betriebsanleitung") > 2


# -- basic behaviour --------------------------------------------------------

def test_empty_input_yields_no_chunks():
    assert chunk_segments([]) == []
    assert chunk_segments([TextSegment(text="   ")]) == []
    assert chunk_text("") == []


def test_short_document_is_one_chunk():
    chunks = chunk_segments([TextSegment(text="A short note about valves.")])
    assert len(chunks) == 1
    assert chunks[0].text == "A short note about valves."
    assert chunks[0].index == 0


def test_overlap_must_be_smaller_than_target():
    with pytest.raises(ValueError):
        chunk_segments([TextSegment(text="x")], target_tokens=100, overlap_tokens=100)


# -- invariant: unique indices ----------------------------------------------

def test_chunk_indices_are_unique_and_consecutive():
    """The previous implementation derived the index by dividing the window
    start by the chunk size while stepping by size minus overlap, producing
    ids like 0, 0, 1, ..., 9, 9. A duplicate index makes a citation ambiguous
    about which passage it points to."""
    segments = [
        TextSegment(text=words(400), location=f"page {page}")
        for page in range(1, 6)
    ]
    chunks = chunk_segments(segments, target_tokens=100, overlap_tokens=20,
                            count_tokens=count_words)
    indices = [chunk.index for chunk in chunks]
    assert len(indices) == len(set(indices)), "duplicate chunk index"
    assert indices == list(range(len(chunks)))


# -- invariant: no chunk without new content --------------------------------

@pytest.mark.parametrize("word_count", [201, 205, 250, 301, 400, 1000, 1001])
def test_no_chunk_repeats_its_predecessor(word_count):
    """A trailing window falling entirely inside the previous chunk's overlap
    carries nothing new. It costs an embedding, and BM25 normalizes for
    length, so the resulting stub outranks real content in the candidate
    pool."""
    chunks = chunk_text(words(word_count), target_tokens=100, overlap_tokens=20,
                        count_tokens=count_words)
    for previous, current in zip(chunks, chunks[1:]):
        new = set(current.split()) - set(previous.split())
        assert new, f"chunk adds nothing to its predecessor: {current[:60]!r}"


@pytest.mark.parametrize("word_count", [50, 201, 205, 250, 700, 1001])
def test_chunking_loses_no_content(word_count):
    source = words(word_count)
    chunks = chunk_text(source, target_tokens=100, overlap_tokens=20,
                        count_tokens=count_words)
    covered = set(" ".join(chunks).split())
    assert set(source.split()) <= covered


def test_chunks_stay_within_the_target_size():
    chunks = chunk_text(words(1000), target_tokens=100, overlap_tokens=20,
                        count_tokens=count_words)
    assert all(count_words(chunk) <= 100 for chunk in chunks)


def test_overlap_is_actually_present():
    """Without overlap, an answer straddling a boundary is retrievable from
    neither side."""
    chunks = chunk_text(words(600), target_tokens=100, overlap_tokens=30,
                        count_tokens=count_words)
    assert len(chunks) > 2
    for previous, current in zip(chunks, chunks[1:]):
        shared = set(previous.split()) & set(current.split())
        assert shared, "consecutive chunks share nothing"


# -- invariant: a chunk never spans two locations ---------------------------

def test_a_chunk_never_mixes_two_pages():
    """If a chunk contained text from page 1 and page 2, its citation would
    name a page holding only half the passage.

    Sizes are chosen so packing *would* mix them if segment boundaries were
    ignored: each segment is four 40-word paragraphs, and the 300-token
    target leaves room to pull a paragraph across the boundary.
    """
    def paragraphs(prefix: str) -> str:
        return "\n\n".join(words(40, f"{prefix}{i}_") for i in range(4))

    segments = [
        TextSegment(text=paragraphs("alpha"), location="page 1"),
        TextSegment(text=paragraphs("beta"), location="page 2"),
    ]
    chunks = chunk_segments(segments, target_tokens=300, overlap_tokens=40,
                            count_tokens=count_words)

    for chunk in chunks:
        assert not ("alpha" in chunk.text and "beta" in chunk.text), \
            f"chunk labelled {chunk.location!r} spans two pages"

    assert {chunk.location for chunk in chunks} == {"page 1", "page 2"}


def test_locations_are_preserved_on_every_chunk():
    segments = [TextSegment(text=words(300), location="sheet Budget")]
    chunks = chunk_segments(segments, target_tokens=100, overlap_tokens=20,
                            count_tokens=count_words)
    assert len(chunks) > 1
    assert all(chunk.location == "sheet Budget" for chunk in chunks)


def test_missing_location_stays_none():
    chunks = chunk_segments([TextSegment(text=words(300))],
                            target_tokens=100, overlap_tokens=20,
                            count_tokens=count_words)
    assert all(chunk.location is None for chunk in chunks)


# -- merging short segments -------------------------------------------------

def test_short_segments_are_merged_with_a_combined_location():
    """A 40-slide deck of one-line slides would otherwise produce 40 chunks
    too small to match anything."""
    segments = [
        TextSegment(text=f"Slide {n} about air handling.", location=f"slide {n}")
        for n in range(1, 5)
    ]
    chunks = chunk_segments(segments, target_tokens=100, overlap_tokens=20,
                            count_tokens=count_words)
    assert len(chunks) < len(segments)
    assert "slide 1" in chunks[0].location
    assert "slide 2" in chunks[0].location


def test_merging_is_capped_so_labels_stay_readable():
    segments = [TextSegment(text="Tiny.", location=f"slide {n}") for n in range(1, 13)]
    chunks = chunk_segments(segments, target_tokens=1000, overlap_tokens=20,
                            count_tokens=count_words)
    assert all(chunk.location.count(",") <= 3 for chunk in chunks)


def test_large_segments_are_never_merged():
    segments = [
        TextSegment(text=words(200, prefix="a"), location="page 1"),
        TextSegment(text=words(200, prefix="b"), location="page 2"),
    ]
    chunks = chunk_segments(segments, target_tokens=100, overlap_tokens=20,
                            count_tokens=count_words)
    assert all("," not in (chunk.location or "") for chunk in chunks)


# -- structural boundaries --------------------------------------------------

def test_paragraphs_are_not_split_when_they_fit():
    text = "\n\n".join([words(20, "para1_"), words(20, "para2_"), words(20, "para3_")])
    chunks = chunk_text(text, target_tokens=100, overlap_tokens=10,
                        count_tokens=count_words)
    assert len(chunks) == 1


def test_table_rows_survive_intact():
    """Extraction emits one row per line specifically so chunking cannot cut
    a record in half and leave both halves unmatchable."""
    table = "\n".join([
        "Part | Interval | Cost",
        "Filter | 6 months | 40 EUR",
        "Belt | 24 months | 120 EUR",
    ])
    chunks = chunk_text(table, target_tokens=100, overlap_tokens=10,
                        count_tokens=count_words)
    joined = "\n".join(chunks)
    assert "Filter | 6 months | 40 EUR" in joined
    assert "Belt | 24 months | 120 EUR" in joined


def test_oversized_paragraph_is_hard_split():
    """PDF pages frequently extract as one unbroken block with no blank
    lines, so there is no structural boundary left to respect."""
    chunks = chunk_text(words(500), target_tokens=100, overlap_tokens=20,
                        count_tokens=count_words)
    assert len(chunks) > 1
    assert all(count_words(chunk) <= 100 for chunk in chunks)


# -- determinism ------------------------------------------------------------

def test_chunking_is_deterministic():
    """Chunk indices double as vector-store ids, so re-indexing the same file
    must overwrite rather than append a second copy of every chunk."""
    segments = [TextSegment(text=words(500), location="page 1")]
    first = chunk_segments(segments, count_tokens=count_words)
    second = chunk_segments(segments, count_tokens=count_words)
    assert first == second
    assert all(isinstance(chunk, Chunk) for chunk in first)
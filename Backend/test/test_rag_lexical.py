"""
Tests for src/rag/lexical.py.

Pure functions over strings: no store, no network, no API key. The two
failure modes exercised most heavily here -- punctuation swallowing tokens
and IDF collapsing on tiny collections -- are both silent, so they need tests
rather than inspection.
"""

import pytest

from src.rag.lexical import BM25Index, tokenize


def top_texts(index: BM25Index, texts: list[str], query: str, k: int = 3) -> list[str]:
    return [texts[position] for position, _ in index.top_k(query, k)]


# -- tokenisation -----------------------------------------------------------

def test_trailing_punctuation_does_not_swallow_the_token():
    """`text.lower().split()` yields 'march.', which can never equal the query
    token 'march'. BM25 matches by equality, so the term is simply invisible
    -- and terms before a full stop are disproportionately the dates and
    identifiers worth searching for."""
    assert "march" in tokenize("The deadline is March.")
    assert "march." not in tokenize("The deadline is March.")


@pytest.mark.parametrize("text", [
    "Reset the unit, then wait.",
    "Error: E14!",
    "(see page 12)",
    "temperature; humidity",
])
def test_punctuation_is_stripped_everywhere(text):
    assert all(not any(c in token for c in ".,;:!?()") for token in tokenize(text))


def test_identifiers_survive_as_units():
    """A part number split on its hyphens loses the exact match a user typing
    the full code expects."""
    assert "hvac-2200-b" in tokenize("Replace unit HVAC-2200-B.")


def test_identifiers_are_also_findable_by_their_parts():
    """Users search for the family as often as the exact code."""
    tokens = tokenize("HVAC-2200-B")
    assert "hvac" in tokens
    assert "2200" in tokens


def test_exact_code_outranks_a_partial_match():
    texts = ["Unit HVAC-2200-B requires a filter change.",
             "Unit HVAC-3100-C requires a belt change."]
    index = BM25Index.build(texts)
    assert top_texts(index, texts, "HVAC-2200-B", k=1) == [texts[0]]


def test_apostrophes_stay_inside_words():
    assert "operator's" in tokenize("the operator's manual")


def test_digits_are_searchable():
    tokens = tokenize("Error code 22 in room 7")
    assert "22" in tokens and "7" in tokens


def test_case_is_normalised():
    assert tokenize("Fehler") == tokenize("FEHLER") == tokenize("fehler")


def test_umlauts_are_preserved():
    """Stripping them would break German search on exactly the words that
    distinguish one term from another."""
    assert "für" in tokenize("Anleitung für Räume")
    assert "räume" in tokenize("Anleitung für Räume")


def test_empty_text_yields_no_tokens():
    assert tokenize("") == []
    assert tokenize("   ...   ") == []


# -- idf on small collections -----------------------------------------------

@pytest.mark.parametrize("size", [1, 2, 3, 10])
def test_idf_is_positive_for_any_collection_size(size):
    """The classic Robertson-Sparck-Jones IDF is non-positive for every term
    when N <= 2 -- exactly a session holding one short document. Lexical
    search would then contribute nothing while still costing a full scan."""
    index = BM25Index.build([f"document number {i}" for i in range(size)])
    assert index.idf("document") > 0
    assert index.idf("absent") > 0


def test_a_single_document_is_still_searchable():
    texts = ["The thermostat is reset by holding the button."]
    index = BM25Index.build(texts)
    assert index.top_k("thermostat", 5), "single-document session returned nothing"


def test_common_terms_are_not_penalised():
    """A negative IDF would make a document score *worse* for containing the
    query term."""
    texts = ["the filter is here"] * 5 + ["something else entirely"]
    index = BM25Index.build(texts)
    assert index.idf("the") > 0


def test_rare_terms_outweigh_common_ones():
    texts = ["maintenance schedule", "maintenance log", "maintenance thermostat"]
    index = BM25Index.build(texts)
    assert index.idf("thermostat") > index.idf("maintenance")


# -- ranking ----------------------------------------------------------------

def test_matching_documents_rank_above_others():
    texts = [
        "The air handling unit runs at sixty percent.",
        "Replace the filter every six months.",
        "Room booking is handled by the calendar agent.",
    ]
    index = BM25Index.build(texts)
    assert top_texts(index, texts, "filter replacement", k=1) == [texts[1]]


def test_documents_sharing_no_term_are_omitted():
    """Passing them on as low-ranked candidates lets fusion promote text that
    matched nothing at all."""
    texts = ["thermostat reset procedure", "catering menu for the canteen"]
    index = BM25Index.build(texts)
    results = index.top_k("thermostat", 10)
    assert len(results) == 1


def test_no_match_returns_nothing():
    index = BM25Index.build(["thermostat reset procedure"])
    assert index.top_k("completely unrelated words", 5) == []


def test_repeated_terms_increase_the_score():
    texts = ["filter", "filter filter filter"]
    index = BM25Index.build(texts)
    scores = index.scores("filter")
    assert scores[1] > scores[0]


def test_short_chunks_are_favoured_at_equal_term_frequency():
    """BM25 length normalisation rewards brevity, and substantially so: with
    the query term appearing once in each, a one-word chunk scores about 2.3x
    a full sentence on the same subject.

    This is not a defect to fix here -- it is correct BM25 -- but it is the
    reason chunking must never emit a degenerate chunk. A trailing window
    holding nothing but overlap would be short, would contain the query term,
    and would therefore outrank the real passage it was copied from. The
    guard in chunking.py and this behaviour are two halves of one invariant.
    """
    passage = ("The filter must be replaced regularly according to the "
               "maintenance schedule provided by the manufacturer of this unit")
    stub = "filter"
    index = BM25Index.build([passage, stub])
    scores = index.scores("filter")
    assert scores[1] > scores[0]


def test_term_frequency_can_outweigh_brevity():
    """Saturation still lets a passage genuinely about the term win over a
    stub that merely mentions it once."""
    passage = "The filter must be replaced every six months. " * 6 + "filter"
    index = BM25Index.build([passage, "filter"])
    scores = index.scores("filter")
    assert scores[0] > scores[1]


def test_top_k_respects_its_limit():
    texts = [f"filter number {i}" for i in range(10)]
    index = BM25Index.build(texts)
    assert len(index.top_k("filter", 3)) == 3


def test_results_are_ordered_best_first():
    texts = ["filter", "filter filter", "filter filter filter"]
    index = BM25Index.build(texts)
    scores = [score for _, score in index.top_k("filter", 3)]
    assert scores == sorted(scores, reverse=True)


def test_ranking_is_deterministic():
    """The evaluation compares configurations, so an unstable tie-break would
    show as a retrieval difference with no cause."""
    texts = ["filter a", "filter b", "filter c"]
    index = BM25Index.build(texts)
    assert index.top_k("filter", 3) == index.top_k("filter", 3)


# -- edges ------------------------------------------------------------------

def test_empty_index_is_harmless():
    index = BM25Index.build([])
    assert len(index) == 0
    assert index.top_k("anything", 5) == []
    assert index.scores("anything") == []


def test_empty_query_matches_nothing():
    index = BM25Index.build(["some text"])
    assert index.top_k("", 5) == []
    assert index.top_k("...", 5) == []


def test_scores_align_with_document_order():
    texts = ["alpha", "beta", "gamma"]
    index = BM25Index.build(texts)
    scores = index.scores("beta")
    assert len(scores) == 3
    assert scores[1] > 0 and scores[0] == 0 and scores[2] == 0


# -- the case hybrid retrieval exists for -----------------------------------

def test_an_error_code_is_found_verbatim():
    """The query pattern dense retrieval handles worst: a user reading a code
    off a display. It carries no distributional meaning, so an embedding
    places it near everything and near nothing."""
    texts = [
        "General troubleshooting advice for climate control systems.",
        "Fehler 22 bedeutet, dass der Filter gewechselt werden muss.",
        "The unit supports several operating modes.",
    ]
    index = BM25Index.build(texts)
    assert top_texts(index, texts, "Fehler 22", k=1) == [texts[1]]
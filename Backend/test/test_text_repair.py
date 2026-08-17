"""
Tests for src/text_repair.py.

The corruption these describe is real and was found by running the pipeline
against the GDPR as published by EUR-Lex: the extractor produced 'REGUL A
TIONS ... Ar ticle 33 ... Notif ication of a personal dat a breach', and 23
of 24 evaluation questions failed as a result.

Two things matter and they pull against each other. The repair has to fix
enough of the damage to make lexical search work, and it must not join words
that were correctly separate -- a wrong join produces a token matching
nothing, which is worse than a split leaving two tokens that at least match
themselves. Both directions are tested.
"""

import pytest

from src.text_repair import (
    build_vocabulary,
    looks_shattered,
    repair_spacing,
)

# Vocabulary evidence: the words the extractor got right elsewhere. A legal
# text repeats itself constantly, which is what makes self-supplied evidence
# work here.
INTACT = (
    "regulations regulations articles article article notification notification "
    "movement movement data data personal personal european european parliament "
    "parliament breach breach supervisory supervisory authority authority "
    "processing processing controller controller"
)


def repair(broken: str, evidence: str = INTACT) -> str:
    document = evidence + "\n" + broken
    return repair_spacing(document, build_vocabulary(document)).split("\n")[1]


def test_the_document_is_not_the_only_evidence():
    """The corruption is deterministic: a word broken once is broken every
    time, so 'aware' and 'supervisory' never appear intact anywhere in the
    GDPR. Building the vocabulary from the document alone therefore cannot
    repair them, which is why a frequency list is consulted first."""
    assert repair("awar e", evidence="") == "aware"
    assert repair("super visor y", evidence="") == "supervisory"


# -- the damage observed in the real document -------------------------------

def test_repairs_the_corruption_found_in_eur_lex_pdfs():
    broken = "REGUL A TIONS Ar ticle 33 Notif ication of a personal breac h mo v ement"
    assert repair(broken) == (
        "REGULATIONS Article 33 Notification of a personal breach movement"
    )


@pytest.mark.parametrize("broken,expected", [
    ("Ar ticle", "Article"),
    ("Notif ication", "Notification"),
    ("mo v ement", "movement"),
    ("pro cessing", "processing"),
    ("contro ller", "controller"),
    # The split falls anywhere in the word, not only near the start, and the
    # pieces are often words in their own right -- 'author' and 'ity', 'occur'
    # and 'red', 'super' and 'visor'. All taken verbatim from the GDPR as
    # extracted.
    ("awar e", "aware"),
    ("breac h", "breach"),
    ("author ity", "authority"),
    ("occur red", "occurred"),
    ("dela y", "delay"),
    ("af ter", "after"),
    ("super visor y", "supervisory"),
    ("pr inciple", "principle"),
])
def test_repairs_each_shape_of_split(broken, expected):
    assert repair(broken) == expected


def test_repairs_a_whole_passage_from_the_real_document():
    """Verbatim from the GDPR as pypdf extracts it."""
    broken = (
        "the controller becomes awar e that a personal data breac h has occur red, "
        "notify the super visor y author ity without undue dela y and not later "
        "than 72 hours af ter having become awar e of it"
    )
    expected = (
        "the controller becomes aware that a personal data breach has occurred, "
        "notify the supervisory authority without undue delay and not later "
        "than 72 hours after having become aware of it"
    )
    assert repair(broken, evidence="") == expected


def test_the_repair_makes_the_text_searchable_again():
    """The point of the exercise. Lexical search matches tokens by equality,
    so a shattered document contains no token 'article' at all and a query for
    it finds nothing -- BM25 contributes nothing to hybrid retrieval."""
    from src.rag.lexical import tokenize

    broken = "Ar ticle 33 Notif ication of a personal breac h"
    assert "article" not in tokenize(broken)
    assert "notification" not in tokenize(broken)
    assert "breach" not in tokenize(broken)

    fixed = tokenize(repair(broken))
    assert "article" in fixed
    assert "notification" in fixed
    assert "breach" in fixed


# -- what must not happen ---------------------------------------------------

def test_correctly_separated_words_are_left_alone():
    """A wrong join is worse than a missed one: it creates a token that
    matches nothing, while a split leaves two that match themselves."""
    text = "the supervisory authority and the personal data of a natural person"
    assert repair(text) == text


@pytest.mark.parametrize("text", [
    "processing of personal data",
    "the controller shall notify the supervisory authority",
    "data protection by design and by default",
])
def test_ordinary_prose_is_unchanged(text):
    assert repair(text) == text


def test_a_common_word_can_be_absorbed_and_that_is_accepted():
    """'direct or indirect' becomes 'director indirect', and it is left that
    way on purpose.

    A ceiling on how common an absorbed piece may be prevents it. Swept
    against both documents, that ceiling costs far more than it saves: without
    it the repair recovers 17 of 24 English questions and 7 of 16 German, with
    it at 6.5 the counts fall to 12 and 3. Five rounds of tuning against
    handfuls of examples like this one took the measured result from 17 to 11
    before the sweep was written.

    The reason the trade runs this way is that the two errors are not equal in
    consequence. A wrong join damages one word and leaves the rest of the
    passage intact and findable; a refused repair can leave a whole passage
    unreachable, because the terms it would be found by no longer exist as
    tokens.
    """
    assert repair("whether direct or indirect") == "whether director indirect"

    # What the same setting buys: German compounds break into single letters,
    # and refusing those was most of the cost.
    assert repair("REGUL A TIONS") == "REGULATIONS"
    assert repair("personal dat a breach") == "personal data breach"


def test_numbers_are_never_absorbed():
    """'Article' followed by '33' is a heading, not a broken word. Joining
    them yields 'article33', which matches neither the word nor the number."""
    assert repair("Ar ticle 33") == "Article 33"
    assert repair("Ar ticle 5 and Ar ticle 6") == "Article 5 and Article 6"


def test_a_string_that_is_not_a_word_is_left_split():
    """'ref ore' is the tail of 'therefore' with its head on another line. No
    join of the visible pieces is a word, so it stays as it is -- the
    conservative outcome, and the right one."""
    assert repair("ref ore") == "ref ore"
    assert repair("xyz zyx") == "xyz zyx"


def test_pieces_that_are_words_can_still_be_joined():
    """The rule that matters. 'author' and 'ity' are both real strings, so a
    test of 'are the pieces unknown' refuses the repair. What separates a
    genuine join from a false one is that the join is more plausible than its
    pieces: 'authority' beats 'ity', while 'personaldata' -- a string no
    frequency list has seen -- loses to both 'personal' and 'data'."""
    assert repair("author ity") == "authority"
    assert repair("personal data") == "personal data"


def test_line_structure_survives():
    """Locations are derived from segment boundaries; a repair that reflowed
    the text would move every citation."""
    document = INTACT + "\nAr ticle 33\nNotif ication of a breach"
    repaired = repair_spacing(document, build_vocabulary(document))
    assert repaired.count("\n") == document.count("\n")
    assert repaired.split("\n")[1] == "Article 33"


# -- detection --------------------------------------------------------------

def test_shattered_text_is_recognised():
    shattered = "REGUL A TIONS Ar ticle 33 Notif ication of a personal dat a breach " * 30
    assert looks_shattered(shattered)


def test_ordinary_text_is_not_flagged():
    """Detection gates the repair, so a false positive would put every
    ordinary document through a transformation it does not need."""
    normal = (
        "The controller shall implement appropriate technical and organisational "
        "measures to ensure a level of security appropriate to the risk. "
    ) * 30
    assert not looks_shattered(normal)


def test_short_text_is_not_judged():
    """Too little text to tell, and guessing would risk mangling it."""
    assert not looks_shattered("Ar ticle 33")


# -- vocabulary -------------------------------------------------------------

def test_vocabulary_ignores_words_seen_only_once():
    """One appearance could itself be a fragment that happens to be a word.
    Requiring repetition is what keeps the evidence trustworthy."""
    vocabulary = build_vocabulary("unique appears once. common common common")
    assert vocabulary["common"] >= 2
    assert vocabulary["unique"] < 2


def test_vocabulary_ignores_very_short_words():
    vocabulary = build_vocabulary("the and for data processing")
    assert "the" not in vocabulary
    assert "data" in vocabulary
"""
Repairing words that PDF extraction split apart.

Some PDFs position individual characters with separate drawing operations to
control kerning. Extractors read the gaps as word boundaries, and the text
comes out shattered: EUR-Lex publishes the Official Journal this way, so the
GDPR extracts as

    REGUL A TIONS ... Ar ticle 33 ... Notif ication of a personal dat a breach

This is not cosmetic. Lexical search matches tokens by equality, so a document
in this state contains no token 'article' and no token 'notification' at all
-- a query for either finds nothing, and BM25 contributes nothing to hybrid
retrieval. Chunk boundaries drift too, since they are measured in tokens.

The repair joins a fragment to its neighbour when the join is a word and the
fragment alone is not. Deciding that needs a vocabulary, and rather than ship
a word list this builds one from the document itself: any long word the
extractor got right somewhere is evidence for the same word elsewhere. A
legal text repeats its vocabulary constantly, which is what makes this work
here; it would work less well on a document that says everything once.

Deliberately conservative. Joining two words that were correctly separate is
worse than leaving one split, because a wrong join creates a token that
matches nothing while a split leaves two tokens that at least match
themselves. So a join happens only when the evidence is direct.
"""

from __future__ import annotations

import re
from collections import Counter

from wordfreq import zipf_frequency

# How often a word must appear intact in the document before it is trusted as
# supporting evidence. Only used as a fallback where the frequency list has
# nothing to say -- domain terms, product names, identifiers.
_MIN_EVIDENCE = 2

# How many pieces a single word may have been broken into. Three covers what
# was observed ('super visor y', 'REGUL A TIONS'); more would mostly add
# chances to glue a phrase into a word that happens to exist.
_MAX_PIECES = 3

# Zipf frequency above which a string is treated as a real word.
_MIN_ZIPF = 2.0

# Score given to a word the frequency list does not know but the document uses
# repeatedly. Raised to 4.0 at one point, on the reasoning that German terms
# of art -- 'Kohaerenzverfahren', 'Datenschutzvorschriften' -- score zero in
# any frequency list and deserve stronger evidence. Measured, that made things
# worse in both languages, and much worse in German: 14/2 against 17/7. A
# high score lets a term the document happens to contain outrank the pieces of
# a join that should not happen. Left at the minimum.
_DOCUMENT_TERM_ZIPF = 2.0

# Whether the ceiling is waived when the other pieces are not words. Added on
# the strength of two examples and measured afterwards: it changes nothing at
# any ceiling (12/3 with it, 12/3 without), because with no ceiling there is
# nothing to waive. Kept as a flag rather than deleted, since it becomes
# relevant again if a ceiling is ever reinstated.
_NEIGHBOUR_EXEMPTION = False

# A join is accepted when it is at least this much more frequent than its
# least frequent piece. The absolute test alone does not work, because the
# pieces are usually real words too: 'author' and 'ity' both exist, as do
# 'occur' and 'red', so requiring the pieces to be unknown refuses almost
# every genuine repair.
#
# Comparing instead is decisive. Measured on the corruption in the GDPR,
# genuine repairs run from +0.10 ('occur red' to 'occurred') to +4.34
# ('pr inciple' to 'principle'), while pairs that must stay separate run from
# -3.32 ('supervisory authority') to -5.20 ('personal data') -- a false join
# produces a string no frequency list has ever seen, so it scores zero and the
# difference is sharply negative. Zero sits in the gap.
_MIN_GAIN = 0.0

# A ceiling on how common an absorbed piece may be was tried, to stop 'direct
# or indirect' becoming 'director indirect'. Swept against both documents, it
# costs far more than it saves: no ceiling recovers 17 of 24 English questions
# and 7 of 16 German, while a ceiling at 6.5 recovers 12 and 3.
#
# The wrong joins it prevents are real but rare, and they leave the passage
# still findable -- 'director indirect' keeps every other word of its context.
# The repairs it refuses remove whole passages from reach. Left off, with the
# constant kept so the sweep can revisit it on a different document.
_MAX_PIECE_ZIPF = 99.0

# Languages the frequency list is consulted in. A join is accepted if any of
# them recognises it, so a bilingual document does not need to be detected.
_LANGUAGES = ("en", "de")

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


# Umlauts and the sharp s, folded to ASCII. The document carries the real
# characters, but a join is judged against both spellings, so a term the
# frequency list only knows in one form is still recognised.
_FOLD = str.maketrans({
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
})


def _fold(word: str) -> str:
    return word.translate(_FOLD)


def build_vocabulary(text: str, min_length: int = 4) -> Counter:
    """Words the extractor got right, counted.

    Only words of `min_length` or more. Short words are where false joins come
    from, since almost any pair of fragments glues into some short string that
    exists somewhere. Four is the lowest that still repairs 'dat a' -- one of
    the commonest shapes of this defect, and one that matters, since 'data' is
    the single most frequent content word in this document.
    """
    counts: Counter = Counter()
    for match in _WORD.finditer(text.lower()):
        word = match.group()
        if len(word) < min_length:
            continue
        counts[word] += 1
        folded = _fold(word)
        if folded != word:
            counts[folded] += 1
    return counts


def repair_spacing(text: str, vocabulary: Counter | None = None) -> str:
    """Join fragments that the extractor split, leaving everything else alone.

    Works on a token stream rather than with a regular expression, because the
    decision depends on the neighbours: 'a' between two words is a fragment in
    'dat a breach' and a word in 'a personal breach', and only the join tells
    them apart.
    """
    if vocabulary is None:
        vocabulary = build_vocabulary(text)

    def frequency(candidate: str) -> float:
        """How plausible a string is as a word.

        The frequency list is asked first because it knows words the document
        never spells correctly -- 'aware' and 'supervisory' never appear
        intact anywhere in the GDPR as extracted. The document is the fallback
        for what the list does not cover: domain vocabulary, product names,
        identifiers. A word the document uses repeatedly is given a modest
        score, enough to beat a fragment but not to outrank common words.
        """
        lowered = candidate.lower()
        forms = {lowered, _fold(lowered)}
        best = max(
            zipf_frequency(form, language)
            for form in forms
            for language in _LANGUAGES
        )
        if best >= _MIN_ZIPF:
            return best

        # Not in the frequency list. That is ordinary for the vocabulary a
        # document is actually about -- 'Kohaerenzverfahren' and
        # 'Datenschutzvorschriften' score zero, being terms of art rather than
        # common words -- so the document's own usage is consulted. A term the
        # document spells correctly several times is treated as a solid word,
        # above the fragments it would be competing with.
        seen = max(vocabulary.get(form, 0) for form in forms)
        if seen >= _MIN_EVIDENCE:
            return _DOCUMENT_TERM_ZIPF
        return best

    def is_word(candidate: str) -> bool:
        return frequency(candidate) >= _MIN_ZIPF

    # Split on spaces only; line structure is preserved so locations survive.
    lines = text.split("\n")
    repaired_lines = []

    for line in lines:
        tokens = line.split(" ")
        out: list[str] = []
        index = 0

        while index < len(tokens):
            token = tokens[index]

            # Prefer the longest join that the evidence supports, so
            # 'mo v ement' is repaired as one word rather than as 'mo' plus a
            # separate attempt at 'v ement'.
            joined = None
            consumed = 0
            for span in range(_MAX_PIECES, 1, -1):
                if index + span > len(tokens):
                    continue
                pieces = tokens[index:index + span]

                # Never absorb a number. 'Article' followed by '33' is a
                # heading, not a broken word, and joining them produces
                # 'article33' -- a token that matches neither the word nor
                # the number.
                if any(any(character.isdigit() for character in piece) for piece in pieces):
                    continue

                candidate = _strip("".join(pieces))
                if not candidate or not is_word(candidate):
                    continue

                # The join must be more plausible than the pieces it replaces.
                # Testing only that the pieces are unknown does not work: they
                # are usually words themselves.
                parts = [_strip(piece) for piece in pieces if _strip(piece)]

                # A piece as common as 'or' or a bare 'a' is a word in its own
                # right, and swallowing it turns 'direct or indirect' into
                # 'director indirect'. But the same shape is the commonest
                # form of the defect in German, where compounds break as
                # 'a ufsichtsbehoerden' and 'k ohaerenzverfahren'.
                #
                # What separates them is the neighbour. In 'direct or' both
                # pieces are words, so they were two words. In
                # 'a ufsichtsbehoerden' the remainder is not a word at all, so
                # it is one word broken. Refuse the join only when every piece
                # stands on its own.
                if any(frequency(part) > _MAX_PIECE_ZIPF for part in parts):
                    if not _NEIGHBOUR_EXEMPTION or all(is_word(part) for part in parts):
                        continue

                weakest = min(frequency(part) for part in parts)
                if frequency(candidate) - weakest < _MIN_GAIN:
                    continue

                joined, consumed = "".join(pieces), span
                break

            if joined is not None:
                out.append(joined)
                index += consumed
            else:
                out.append(token)
                index += 1

        repaired_lines.append(" ".join(out))

    return "\n".join(repaired_lines)


def _strip(token: str) -> str:
    """The letters in a token, ignoring punctuation attached to it."""
    return "".join(character for character in token if character.isalpha())


def looks_shattered(text: str, sample: int = 20000) -> bool:
    """Whether a document appears to have this problem at all.

    Checked before repairing, so an ordinary document is not put through a
    transformation it does not need. The signal is the share of one- and
    two-letter fragments: normal prose has some ('a', 'of', 'to'), shattered
    text has far more.
    """
    tokens = [_strip(token) for token in text[:sample].split()]
    tokens = [token for token in tokens if token]
    if len(tokens) < 200:
        return False
    stubs = sum(1 for token in tokens if len(token) <= 2)
    return stubs / len(tokens) > 0.22
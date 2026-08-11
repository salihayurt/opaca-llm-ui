"""
Lexical search over chunks.

Dense embeddings compress away exactly the tokens SAGE users search for.
An error code read off a display (`E14`, `Fehler 22`), a room name
(`Experience Hub`), a part number (`HVAC-2200-B`) carries almost no
distributional meaning, so a semantic vector places it near everything and
near nothing. BM25 matches it exactly. That is why retrieval here is hybrid
and not dense-only, and why the TU Berlin ablation found dense retrieval the
weakest option on both of its benchmarks.

Implemented rather than taken from `rank_bm25`, for two reasons worth stating
because "not invented here" is usually the wrong call:

- Tokenisation has to be ours anyway. The identifiers above only survive if
  the tokeniser is written around them, and German compounding needs a hook
  that a general-purpose library does not offer.
- The evaluation compares configurations against a baseline, so scoring must
  be deterministic and inspectable. A silent change in a dependency would
  show up as a retrieval regression with no visible cause.

It is roughly seventy lines. The cost of owning them is smaller than the cost
of not being able to explain a score.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# Free parameters of BM25. These are the standard values and are exposed so
# the evaluation can sweep them rather than treating them as folklore.
K1 = 1.5
B = 0.75

# A token is a run of letters, digits and marks that belong inside a word.
# Hyphens and apostrophes are kept so `HVAC-2200-B` and `operator's` survive
# as units; everything else is a separator.
_TOKEN = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)

# Splits a compound token into its parts, so `HVAC-2200-B` is findable by
# `HVAC` alone.
_PART = re.compile(r"[-'’]")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase search tokens.

    The obvious implementation, `text.lower().split()`, leaves punctuation
    attached: "the deadline is March." yields `march.`, which can never equal
    the query token `march`. Because BM25 matches by equality, that silently
    hides every term sitting before a full stop or comma -- which is
    disproportionately the dates and identifiers worth searching for.

    Hyphenated tokens are emitted whole *and* in parts, so `HVAC-2200-B`
    matches a query for `HVAC` while an exact search for the full code still
    scores higher, having matched more terms.
    """
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        token = match.group()
        tokens.append(token)
        parts = [part for part in _PART.split(token) if part]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


@dataclass
class BM25Index:
    """An in-memory BM25 index over a fixed set of documents."""

    documents: list[list[str]] = field(default_factory=list)
    document_frequency: Counter = field(default_factory=Counter)
    average_length: float = 0.0

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Index":
        documents = [tokenize(text) for text in texts]
        frequency: Counter = Counter()
        for tokens in documents:
            frequency.update(set(tokens))
        lengths = [len(tokens) for tokens in documents]
        average = sum(lengths) / len(lengths) if lengths else 0.0
        return cls(documents=documents, document_frequency=frequency, average_length=average)

    def __len__(self) -> int:
        return len(self.documents)

    def idf(self, term: str) -> float:
        """Inverse document frequency, in the variant that stays positive.

        The classic Robertson-Sparck-Jones form,
        `log((N - n + 0.5) / (n + 0.5))`, goes *negative* for any term
        appearing in more than about half the documents, so a common word
        actively penalises the documents containing it. Worse for us, it is
        non-positive for every term when N <= 2, which is precisely the state
        of a session holding one short document: lexical search silently
        contributes nothing while still costing a full scan.

        Lucene's variant, `log(1 + (N - n + 0.5) / (n + 0.5))`, is positive
        everywhere and degrades gracefully on tiny collections -- with a
        single document it still returns about 0.29 rather than 0.
        """
        total = len(self.documents)
        if total == 0:
            return 0.0
        containing = self.document_frequency.get(term, 0)
        return math.log(1 + (total - containing + 0.5) / (containing + 0.5))

    def scores(self, query: str) -> list[float]:
        """Score every document against `query`, in index order."""
        terms = tokenize(query)
        if not terms or not self.documents:
            return [0.0] * len(self.documents)

        weights = {term: self.idf(term) for term in set(terms)}
        results = []
        for tokens in self.documents:
            counts = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for term in set(terms):
                occurrences = counts.get(term, 0)
                if not occurrences:
                    continue
                # Length normalisation: without it a very short chunk
                # containing the term once outranks a substantial passage
                # about the same subject.
                denominator = occurrences + K1 * (
                    1 - B + B * (length / self.average_length if self.average_length else 1)
                )
                score += weights[term] * (occurrences * (K1 + 1)) / denominator
            results.append(score)
        return results

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return the `k` best (document index, score) pairs, best first.

        Documents scoring zero are omitted: they share no term with the query,
        and passing them on as low-ranked candidates would let fusion promote
        text that matched nothing at all.
        """
        scored = [
            (index, score)
            for index, score in enumerate(self.scores(query))
            if score > 0
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]
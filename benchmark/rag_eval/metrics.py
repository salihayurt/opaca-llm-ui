"""
Scoring retrieval against the question set.

Every metric here answers one question: did a chunk from the article that
contains the answer come back, and how far up? No generated answer, no LLM
call, so the whole configuration grid costs nothing beyond embedding the
document once.

That is the same discipline as the TU Berlin funnel: free retrieval metrics on
wide grids first, paid generation only for the handful of finalists that
survive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# "Article 33", "Artikel 33", "Art. 33", and for manuals "Section 7" or a
# bare numbered heading such as "7. Error Codes" at the start of a line.
_ARTICLE = re.compile(
    r"\b(?:article|artikel|art\.?|section|kapitel)\s*(\d{1,3})\b",
    re.IGNORECASE,
)
_HEADING = re.compile(r"(?:^|\n)\s*(\d{1,3})\.\s+[A-Z]")


def articles_in(text: str) -> set[int]:
    """Every article number named in a passage.

    A passage is credited if it mentions the gold article anywhere, not only
    as its own heading. This is deliberately generous, and the reason is that
    chunk boundaries do not line up with article boundaries: a chunk can carry
    the whole of Article 33 while its heading sits in the previous chunk.
    Requiring the heading would score a chunk containing the answer as a miss.

    A numbered heading counts as well, so a manual whose sections are written
    as "7. Error Codes" rather than "Section 7" is scored the same way.

    The cost is that a cross-reference counts too -- Article 35 saying "as
    referred to in Article 33" earns credit for 33. That inflates every
    configuration equally, so comparisons between them stay sound, but it
    means the absolute numbers here are an upper bound and should not be
    quoted as recall in any other sense.
    """
    found = {int(match.group(1)) for match in _ARTICLE.finditer(text)}
    found |= {int(match.group(1)) for match in _HEADING.finditer(text)}
    return found


@dataclass
class QuestionResult:
    """What one configuration did on one question."""

    question: str
    article: int
    category: str
    language: str
    rank: int | None          # 1-based rank of the first hit, None if missed
    retrieved: int            # how many passages came back at all
    returned_nothing: bool    # retrieval returned an empty list

    @property
    def hit(self) -> bool:
        return self.rank is not None


def score_question(question, passages: list[str]) -> QuestionResult:
    """Find the rank of the first passage covering the question's article."""
    rank = None
    for position, passage in enumerate(passages, start=1):
        if question.article in articles_in(passage):
            rank = position
            break

    return QuestionResult(
        question=question.text,
        article=question.article,
        category=question.category,
        language=question.language,
        rank=rank,
        retrieved=len(passages),
        returned_nothing=not passages,
    )


@dataclass
class Scores:
    """Aggregate metrics for one configuration over the question set."""

    n: int = 0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    answerable_rate: float = 0.0
    by_category: dict = field(default_factory=dict)
    by_language: dict = field(default_factory=dict)


def _recall_at(results: list[QuestionResult], k: int) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.rank is not None and r.rank <= k) / len(results)


def _mrr(results: list[QuestionResult]) -> float:
    if not results:
        return 0.0
    return sum(1 / r.rank for r in results if r.rank is not None) / len(results)


def aggregate(results: list[QuestionResult]) -> Scores:
    """Summarise results, overall and split by category and language.

    `answerable_rate` -- the share of questions that got any passage back at
    all -- is reported alongside recall because the baseline can return an
    empty list, and the two failures are different. A configuration that
    retrieves the wrong passage and one that retrieves nothing both score zero
    recall, but the first is a ranking problem and the second is a threshold
    problem, and they call for opposite fixes. Reporting recall alone would
    hide which one is happening.
    """
    scores = Scores(n=len(results))
    if not results:
        return scores

    scores.recall_at_1 = _recall_at(results, 1)
    scores.recall_at_3 = _recall_at(results, 3)
    scores.recall_at_5 = _recall_at(results, 5)
    scores.recall_at_10 = _recall_at(results, 10)
    scores.mrr = _mrr(results)
    scores.answerable_rate = sum(1 for r in results if not r.returned_nothing) / len(results)

    for category in sorted({r.category for r in results}):
        subset = [r for r in results if r.category == category]
        scores.by_category[category] = {
            "n": len(subset),
            "recall@3": _recall_at(subset, 3),
            "recall@10": _recall_at(subset, 10),
            "mrr": _mrr(subset),
        }

    for language in sorted({r.language for r in results}):
        subset = [r for r in results if r.language == language]
        scores.by_language[language] = {
            "n": len(subset),
            "recall@3": _recall_at(subset, 3),
            "recall@10": _recall_at(subset, 10),
            "mrr": _mrr(subset),
        }

    return scores


def format_scores(label: str, scores: Scores) -> str:
    lines = [
        f"{label}",
        f"  n={scores.n}  R@1={scores.recall_at_1:.3f}  R@3={scores.recall_at_3:.3f}  "
        f"R@5={scores.recall_at_5:.3f}  R@10={scores.recall_at_10:.3f}  "
        f"MRR={scores.mrr:.3f}  answerable={scores.answerable_rate:.3f}",
    ]
    for category, values in scores.by_category.items():
        lines.append(
            f"    {category:<9} n={values['n']:<3} R@3={values['recall@3']:.3f}  "
            f"R@10={values['recall@10']:.3f}  MRR={values['mrr']:.3f}"
        )
    for language, values in scores.by_language.items():
        lines.append(
            f"    [{language}]      n={values['n']:<3} R@3={values['recall@3']:.3f}  "
            f"R@10={values['recall@10']:.3f}  MRR={values['mrr']:.3f}"
        )
    return "\n".join(lines)
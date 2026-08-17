"""
The evaluation harness.

Runs the baseline and a grid of new-pipeline configurations over the same
question set and the same document, and writes one CSV row per configuration.

Ordered cheap to expensive, following the TU Berlin funnel. Every
configuration below is scored on retrieval alone -- no generated answers, no
LLM judge -- so the entire grid costs one embedding pass per chunking
configuration and nothing else. Generation is a separate, later step for
whichever configurations survive.

Run from the repository root:

    export OPENAI_API_KEY=sk-...
    python benchmark/rag_eval/run.py --document path/to/GDPR_EN.pdf

Add --dry-run to check the question set against the document without spending
anything.

Invoked as a script rather than with -m, because benchmark/ is a plain
directory of evaluation material rather than a package, and making it one to
suit this file would change how the existing benchmark suite is laid out --
including shadowing the benchmark/run.py that is already there.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Backend/ on the path for `src`, and this directory for the sibling modules,
# so the file works when run directly from anywhere.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "Backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qdrant_client import AsyncQdrantClient

from src.rag.chunking import chunk_segments
from src.rag.embedding import KNOWN_DIMENSIONS, LiteLLMEmbedder
from src.rag.retrieval import RetrievalConfig, Retriever
from src.rag.store import DocumentStore
from src.text_extraction import extract_segments

from baseline import BaselineStore
from metrics import QuestionResult, aggregate, format_scores, score_question
from questions import ALL


@dataclass(frozen=True)
class Config:
    """One point in the configuration grid."""

    label: str
    chunk_tokens: int = 300
    chunk_overlap: int = 50
    hybrid: bool = True
    rerank: bool = False
    min_dense_score: float = 0.0


# The grid. Each block varies one axis around a fixed centre, so a difference
# in the numbers is attributable to that axis rather than to a combination.
# This is the ablation shape the design document commits to: enabling
# everything at once produces one number and no explanation.
GRID = [
    # centre
    Config("hybrid-300"),

    # chunk size
    Config("hybrid-150", chunk_tokens=150, chunk_overlap=25),
    Config("hybrid-500", chunk_tokens=500, chunk_overlap=50),
    Config("hybrid-800", chunk_tokens=800, chunk_overlap=80),

    # retrieval mode
    Config("dense-only", hybrid=False),

    # relevance floor -- the open question from docs/rag_design.md. The
    # baseline sets the equivalent of 0.85 and answers almost nothing; we set
    # none and answer everything. Neither is measured.
    Config("floor-0.20", min_dense_score=0.20),
    Config("floor-0.30", min_dense_score=0.30),
    Config("floor-0.40", min_dense_score=0.40),
]


async def load_document(path: Path) -> tuple[str, list]:
    data = path.read_bytes()
    segments = extract_segments(path.name, data)
    text = "\n\n".join(segment.text for segment in segments)
    return text, segments


# Umlauts and the sharp s are written in ASCII in questions.py -- the file
# has to survive being pasted through editors and terminals that mangle them --
# while the document has the real characters. Folding both sides makes the
# comparison independent of that.
_FOLD = str.maketrans({
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00a0": " ",
})


def _normalise(text: str) -> str:
    """Fold accents, quotes and whitespace so a quote can be found reliably.

    PDF extraction introduces line breaks and doubled spaces in places the
    source has neither, so whitespace is collapsed rather than matched.
    """
    return " ".join(text.translate(_FOLD).lower().split())


def verify_questions(text: str) -> tuple[list, list]:
    """Drop questions whose quote cannot be found in the document.

    This is the step that makes the measurement trustworthy. The gold labels
    were written by reading the regulation, and a label that is wrong does not
    show up as a bad label -- it shows up as every configuration failing that
    question, which reads as a retrieval problem and would send us tuning the
    pipeline to fix a typo.

    Verifying the quote rather than just the article heading also catches a
    subtler error: a question whose answer is really in a recital, not in the
    numbered article. GDPR's recitals restate much of the articles, so an
    article number attached to a recital passage looks plausible and is wrong.
    """
    haystack = _normalise(text)
    kept, dropped = [], []
    for question in ALL:
        needle = _normalise(question.quote)
        if needle and needle in haystack:
            kept.append(question)
        else:
            dropped.append(question)
    return kept, dropped


async def run_baseline(text: str, questions: list, embed) -> list[QuestionResult]:
    store = BaselineStore(embed)
    chunks = await store.add_to_index("document", text)
    print(f"  baseline indexed {chunks} chunks")

    results = []
    for question in questions:
        hits = await store.retrieve(question.text, k=10)
        results.append(score_question(question, [chunk[2] for _, chunk in hits]))
    return results


async def run_baseline_unfiltered(text: str, questions: list, embed) -> list[QuestionResult]:
    """The baseline's ranking without its distance cut-off.

    Reported separately so a miss can be attributed: bad ranking, or a good
    ranking discarded by the threshold.
    """
    store = BaselineStore(embed)
    await store.add_to_index("document", text)
    results = []
    for question in questions:
        hits = await store.retrieve_unfiltered(question.text, k=10)
        results.append(score_question(question, [chunk[2] for _, chunk in hits]))
    return results


async def run_config(
    config: Config, segments: list, questions: list, embedder, client
) -> list[QuestionResult]:
    session = f"eval-{config.label}"
    store = DocumentStore(client, vector_size=await embedder.dimensions())
    await store.delete_session(session)

    chunks = chunk_segments(
        segments,
        target_tokens=config.chunk_tokens,
        overlap_tokens=config.chunk_overlap,
    )
    vectors = await embedder.embed_documents([chunk.text for chunk in chunks])
    await store.upsert_document(session, "doc", "document", chunks, vectors)
    print(f"  {config.label}: {len(chunks)} chunks")

    retriever = Retriever(
        store,
        embedder,
        config=RetrievalConfig(
            hybrid=config.hybrid,
            rerank=config.rerank,
            min_dense_score=config.min_dense_score,
        ),
    )

    results = []
    for question in questions:
        found = await retriever.retrieve(session, question.text, limit=10)
        # A stage that silently failed produces a row identical to the
        # baseline and invites the conclusion that the feature does nothing.
        if found.degraded:
            print(f"    ! degraded on {question.text[:40]!r}: {found.degraded}")
        results.append(score_question(question, [chunk.text for chunk in found.chunks]))

    await store.delete_session(session)
    return results


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("rag_eval_results.csv"))
    parser.add_argument("--dry-run", action="store_true",
                        help="verify the question set against the document and stop")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    text, segments = await load_document(args.document)
    print(f"document: {args.document.name}")
    print(f"  {len(segments)} segments, {len(text)} characters\n")

    questions, dropped = verify_questions(text)
    print(f"questions: {len(questions)} usable, {len(dropped)} dropped")
    for question in dropped:
        print(f"  dropped (article {question.article} not found): {question.text}")
    print()

    if args.dry_run:
        return 0
    if not questions:
        print("no usable questions; stopping")
        return 1

    embedder = LiteLLMEmbedder(model="text-embedding-3-small")
    calls = {"n": 0, "texts": 0}

    async def counted_embed(texts):
        calls["n"] += 1
        calls["texts"] += len(texts)
        return await embedder.embed_documents(texts)

    rows = []
    started = time.time()

    if not args.skip_baseline:
        print("baseline (ada-002, 500-word windows, cosine floor 0.85):")
        baseline_embedder = LiteLLMEmbedder(model=BaselineStore.MODEL)

        async def baseline_embed(texts):
            calls["n"] += 1
            calls["texts"] += len(texts)
            return await baseline_embedder.embed_documents(texts)

        results = await run_baseline(text, questions, baseline_embed)
        scores = aggregate(results)
        print(format_scores("  baseline", scores))
        rows.append({"config": "baseline", **_flatten(scores)})

        results = await run_baseline_unfiltered(text, questions, baseline_embed)
        scores = aggregate(results)
        print(format_scores("  baseline (no threshold)", scores))
        rows.append({"config": "baseline-no-threshold", **_flatten(scores)})
        print()

    client = AsyncQdrantClient(":memory:")
    for config in GRID:
        results = await run_config(config, segments, questions, embedder, client)
        scores = aggregate(results)
        print(format_scores(f"  {config.label}", scores))
        rows.append({"config": config.label, **asdict(config), **_flatten(scores)})
        print()
    await client.close()

    fields = sorted({key for row in rows for key in row})
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out}")
    print(f"{calls['n']} embedding calls, {calls['texts']} texts, "
          f"{time.time() - started:.0f}s")
    return 0


def _flatten(scores) -> dict:
    row = {
        "n": scores.n,
        "recall@1": round(scores.recall_at_1, 4),
        "recall@3": round(scores.recall_at_3, 4),
        "recall@5": round(scores.recall_at_5, 4),
        "recall@10": round(scores.recall_at_10, 4),
        "mrr": round(scores.mrr, 4),
        "answerable": round(scores.answerable_rate, 4),
    }
    for category, values in scores.by_category.items():
        row[f"{category}_recall@3"] = round(values["recall@3"], 4)
        row[f"{category}_mrr"] = round(values["mrr"], 4)
    for language, values in scores.by_language.items():
        row[f"{language}_recall@3"] = round(values["recall@3"], 4)
        row[f"{language}_mrr"] = round(values["mrr"], 4)
    return row


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
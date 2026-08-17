"""
Why did the quotes not match?

Run this before spending anything on the evaluation. It shows what the
extractor actually produced around each gold article, and how far the quote
is from it, so the cause is visible rather than guessed at.

    python benchmark/rag_eval/diagnose.py --document ~/Downloads/GDPR_EN.pdf

Three causes are worth telling apart, and they need different fixes:

  - The quote is right but the extracted text differs mechanically -- a
    hyphen at a line break, a non-breaking space, a ligature. Fix the
    normaliser.
  - The quote is right but sits in a part of the document the extractor
    dropped or reordered. Fix the extractor, or accept a smaller set.
  - The quote is wrong: it came from a recital, or from the model's memory
    rather than the page. Drop the question.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "Backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.text_extraction import extract_segments

from questions import ALL
from run import _normalise


def best_window(haystack: str, needle: str) -> tuple[float, str]:
    """The closest stretch of the document to the quote, and how close.

    Anchored rather than exhaustive. Sliding a window across 350,000
    characters and running a sequence comparison at every offset takes minutes
    per question; instead the rarest words of the quote are located first, and
    only the neighbourhoods around them are compared. If none of the quote's
    distinctive words appear anywhere, there is nothing to compare and the
    answer is already known.
    """
    words = [word for word in needle.split() if len(word) > 4]
    if not words:
        words = needle.split()[:3]

    # Rarest first: a word that occurs twice locates the passage, a word that
    # occurs six hundred times locates nothing.
    anchors = sorted(words, key=lambda word: haystack.count(word))[:4]

    offsets = []
    for anchor in anchors:
        start = 0
        found = 0
        while found < 25:
            position = haystack.find(anchor, start)
            if position < 0:
                break
            offsets.append(position)
            start = position + 1
            found += 1

    if not offsets:
        return 0.0, ""

    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(needle)

    best_ratio, best_text = 0.0, ""
    span = len(needle)
    for offset in offsets:
        for start in (offset - span, offset - span // 2, offset):
            window = haystack[max(start, 0):max(start, 0) + span]
            if not window:
                continue
            matcher.set_seq1(window)
            if matcher.quick_ratio() < best_ratio:
                continue
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best_ratio, best_text = ratio, window
    return best_ratio, best_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", required=True, type=Path)
    parser.add_argument("--language", default=None,
                        help="only check questions in this language")
    parser.add_argument("--show", type=int, default=8,
                        help="how many failing questions to detail")
    args = parser.parse_args()

    data = args.document.read_bytes()
    segments = extract_segments(args.document.name, data)
    raw = "\n\n".join(segment.text for segment in segments)
    text = _normalise(raw)

    print(f"document: {args.document.name}")
    print(f"  {len(segments)} segments, {len(raw)} characters\n")

    print("first 400 characters as extracted:")
    print("  " + repr(raw[:400]))
    print()

    # Does the document contain article headings at all? If not, the
    # extraction lost the structure and no quote will match.
    for marker in ("Article 33", "Artikel 33", "Article 1", "Artikel 1",
                   "Aufsichtsbehörde", "Verarbeitung", "personenbezogene"):
        print(f"  {marker!r:16} in raw text: {marker in raw}")
    print()

    questions = ALL
    if args.language:
        questions = [q for q in ALL if q.language == args.language]

    exact, near, missing = [], [], []
    for question in questions:
        needle = _normalise(question.quote)
        if needle in text:
            exact.append(question)
        else:
            ratio, window = best_window(text, needle)
            (near if ratio >= 0.75 else missing).append((question, ratio, window))

    print(f"exact matches : {len(exact)}/{len(questions)}")
    print(f"near misses   : {len(near)}   (>=0.75 similar text found)")
    print(f"not found     : {len(missing)}\n")

    if near:
        print("NEAR MISSES -- the text is there but differs mechanically.")
        print("These point at the normaliser, not at the question.\n")
        for question, ratio, window in near[:args.show]:
            print(f"  article {question.article} ({question.language}) ratio={ratio:.2f}")
            print(f"    wanted : {_normalise(question.quote)[:110]}")
            print(f"    found  : {window[:110]}")
            print()

    if missing:
        print("NOT FOUND -- no similar text anywhere.")
        print("Either the extractor lost this part, or the quote is wrong.\n")
        for question, ratio, window in missing[:args.show]:
            print(f"  article {question.article} ({question.language}) best ratio={ratio:.2f}")
            print(f"    wanted : {_normalise(question.quote)[:110]}")
            print(f"    closest: {window[:110]}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
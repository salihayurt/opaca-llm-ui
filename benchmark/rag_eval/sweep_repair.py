"""
Which repair settings actually recover the most questions?

The repair has been tuned five times against handfuls of examples, and the
measured result went 17, 12, 11 -- worse each time. Small examples are the
wrong instrument: a rule that fixes 'direct or' and breaks 'a
ufsichtsbehoerden' looks like progress until it meets the document.

This runs the settings against the real file and reports what matters, which
is how many evaluation questions survive verification. No API calls, so it
costs nothing to run as often as needed.

    python benchmark/rag_eval/sweep_repair.py --document ~/Downloads/GDPR_EN.pdf
    python benchmark/rag_eval/sweep_repair.py --document ~/Downloads/GDPR_DE.pdf

Run both. The two languages break differently -- English splits words near
their edges, German splits compounds into many short pieces -- and a setting
that suits one has repeatedly damaged the other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "Backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.text_repair as repair_module
from src.text_extraction import EXTRACTORS

from questions import ALL
from run import _normalise


# The axes worth sweeping, and why each one is in doubt.
#
#   ceiling      how common a piece may be and still be absorbed. Low values
#                protect 'direct or'; high values allow 'a ufsichtsbehoerden'.
#   neighbour    whether the ceiling is waived when the remaining pieces are
#                not words. Intended to get both, has not been measured.
#   doc_zipf     what a term the frequency list does not know but the document
#                repeats is worth. German terms of art all score zero.
SETTINGS = [
    ("ceiling=none  neighbour=off", 99.0, False, 2.0),
    ("ceiling=6.0   neighbour=off", 6.0, False, 2.0),
    ("ceiling=6.5   neighbour=off", 6.5, False, 2.0),
    ("ceiling=7.0   neighbour=off", 7.0, False, 2.0),
    ("ceiling=6.5   neighbour=on ", 6.5, True, 2.0),
    ("ceiling=6.5   neighbour=on   doc=4.0", 6.5, True, 4.0),
    ("ceiling=none  neighbour=off  doc=4.0", 99.0, False, 4.0),
]


def extract_raw(path: Path) -> list:
    """Extract without the repair, so each setting starts from the same text."""
    extension = path.suffix.lower()
    return EXTRACTORS[extension](path.read_bytes())


def usable(text: str, language: str | None) -> int:
    haystack = _normalise(text)
    questions = [q for q in ALL if language is None or q.language == language]
    return sum(1 for q in questions if _normalise(q.quote) in haystack)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", required=True, type=Path)
    args = parser.parse_args()

    segments = extract_raw(args.document)
    raw = "\n\n".join(segment.text for segment in segments)

    language = "de" if "_DE" in args.document.name.upper() else "en"
    total = sum(1 for q in ALL if q.language == language)

    print(f"document: {args.document.name}  ({language}, {total} questions in that language)")
    print(f"  {len(raw)} characters before repair")
    print(f"  no repair at all: {usable(raw, language)}/{total}\n")

    for label, ceiling, neighbour, doc_zipf in SETTINGS:
        repair_module._MAX_PIECE_ZIPF = ceiling
        repair_module._DOCUMENT_TERM_ZIPF = doc_zipf
        repair_module._NEIGHBOUR_EXEMPTION = neighbour

        vocabulary = repair_module.build_vocabulary(raw)
        repaired = "\n\n".join(
            repair_module.repair_spacing(segment.text, vocabulary)
            for segment in segments
        )
        print(f"  {label:38} {usable(repaired, language):>2}/{total}")

    print("\nPick the setting with the most usable questions in both languages.")
    print("Where they disagree, prefer the one that is not worst in either --")
    print("a setting that wins in English and collapses in German is not a win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
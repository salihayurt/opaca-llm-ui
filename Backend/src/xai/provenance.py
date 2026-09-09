"""Where each tool argument came from.

Classical local attribution (LIME, SHAP) approximates which input caused an
output, because the causal path runs through weights nobody can see. In a
tool-calling agent it runs through data: an argument value either occurs in an
earlier result, or in the user's request, or nowhere. That is decidable, so it
is decided here rather than estimated, and no model is involved.

The `unmatched` verdict is the point of the module. A value traceable to no
source is one the model produced on its own -- from a default, from the system
prompt, or from nothing. That is where an invented argument hides.

Two failure modes shape the rules below. Matching too eagerly is the worse
one: `1`, `true` and `de` occur in every result, so edges drawn from them are
noise that buries the real ones. Matching too strictly loses the reformatted
cases -- `2026-08-27` reappearing as `27.08.2026` is the same value.
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field

from ..models import ToolType
from .trace import ExecutionTrace, TraceCall

Source = Literal["user_query", "tool_result", "prior_argument", "unmatched"]
Kind = Literal["exact", "normalised", "substring", "reworded", "weak"]

# Below this length a string matches by coincidence more often than by
# derivation. Country codes and single letters are the casualties; they carry
# no explanatory weight anyway.
_MIN_MATCH_LENGTH = 4

# Values that appear in almost any result and must never draw an edge.
_UNINFORMATIVE = {"", "true", "false", "none", "null", "0", "1", "-1", "yes", "no"}


class DataFlowEdge(BaseModel):
    """One argument value, and where it came from.

    Computed, never model-generated. `source == "unmatched"` is a finding, not
    a gap: it means the value entered the call from outside the conversation.
    """
    to_call_id: str
    to_param_path: str
    value: str
    source: Source
    kind: Kind | None = None
    from_call_id: str | None = None
    from_call_name: str | None = None
    from_path: str | None = None

    @property
    def is_grounded(self) -> bool:
        """Whether the value traces to something outside the model.

        `prior_argument` is excluded: it says where the value was seen before,
        not where it originally came from. Chasing that chain to its root ends
        either at a real source or at an unmatched value, and until it does the
        honest answer is "not established".
        """
        return self.source in ("user_query", "tool_result")


def flatten(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Walk a structure, yielding (json path, leaf value) pairs."""
    if isinstance(value, dict):
        for key, val in value.items():
            yield from flatten(val, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, val in enumerate(value):
            yield from flatten(val, f"{path}[{index}]")
    else:
        yield path, value


_WHITESPACE = re.compile(r"\s+")
_DATE_PATTERNS = (
    re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$"),          # 2026-08-27
    re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$"),          # 27.08.2026
)

# Letters NFKD does not decompose, because they are letters in their own right
# rather than a base plus an accent.
_LETTER_MAP = str.maketrans({"ı": "i", "İ": "i", "ß": "ss", "ø": "o", "Ø": "o",
                             "æ": "ae", "Æ": "ae", "đ": "d", "ł": "l"})


def fold(text: str) -> str:
    """Drop diacritics so that differently-typed forms of a word compare equal.

    Users type ASCII, models emit the properly accented form: a query for
    "nasil calisan" comes back as "nasıl çalışan", and "koln" as "Köln".
    Without folding the value looks invented, which is the loudest possible way
    to be wrong about it. German and Turkish are both first-class here.
    """
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text.translate(_LETTER_MAP))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalise(value: Any) -> str:
    """Reduce a value to the form two equal values share.

    Dates are handled explicitly because SAGE's agents and its users disagree
    about them constantly -- an OPACA action returns ISO, a user writes German
    format, and the same day would otherwise look like two values.
    """
    text = fold(_WHITESPACE.sub(" ", str(value)).strip().lower())

    for pattern in _DATE_PATTERNS:
        match = pattern.match(text)
        if match:
            groups = match.groups()
            year, month, day = (groups if len(groups[0]) == 4 else (groups[2], groups[1], groups[0]))
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    return text


def is_informative(value: Any) -> bool:
    """Whether a value is distinctive enough for a match to mean anything."""
    if isinstance(value, bool) or value is None:
        return False
    text = str(value).strip()
    if text.lower() in _UNINFORMATIVE:
        return False
    if isinstance(value, (int, float)):
        # Small integers are indices and counts; they collide with everything.
        return abs(float(value)) >= 1000
    return len(text) >= _MIN_MATCH_LENGTH


def _search(haystack: Any, needle: Any, path: str = "",
            exact_only: bool = False) -> tuple[str, Kind] | None:
    """Find `needle` inside a structure. Returns the path and how it matched.

    Exact and normalised are checked across the whole structure before
    substring is considered anywhere, so a weaker match never wins over a
    stronger one that exists further along.

    `exact_only` is for low-entropy values, where anything looser would match
    by accident.
    """
    target = normalise(needle)
    if not target:
        return None

    leaves = list(flatten(haystack))

    for path_, leaf in leaves:
        if leaf == needle:
            return path_, "exact"

    if exact_only:
        return None

    for path_, leaf in leaves:
        if normalise(leaf) == target:
            return path_, "normalised"

    for path_, leaf in leaves:
        leaf_text = normalise(leaf)
        if len(target) >= _MIN_MATCH_LENGTH and target in leaf_text:
            return path_, "substring"

    return None


def _search_query(query: str, needle: Any) -> Kind | None:
    target = normalise(needle)
    if not target:
        return None
    query_text = normalise(query)
    if target == query_text:
        return "exact"
    if target in query_text:
        return "substring"

    # Search arguments are usually reworded rather than copied: "What is the
    # part number for the air filter?" becomes "air filter part number". Every
    # word came from the user, so calling it untraceable puts a reformulation
    # in the same bucket as a value the model made up, and the flag stops
    # meaning anything. Requires two or more words, since a single one that
    # occurs in the query would already have matched as a substring.
    words = [w for w in re.findall(r"\w+", target) if len(w) >= 3]
    if len(words) >= 2 and all(w in query_text for w in words):
        return "reworded"

    return None


def edges_for_call(call_id: str, args: dict[str, Any], trace: ExecutionTrace) -> list[DataFlowEdge]:
    """Resolve one call's arguments, including one that has not run yet.

    The approval dialog needs this before the call happens, which is the point
    at which the answer is worth something: afterwards it explains, beforehand
    it informs a decision.

    Whether the pending call is already in the trace depends on the method --
    tool-llm appends the step before invoking, self-orchestrated after -- so
    both are handled. Sibling calls in the same batch have no results yet, so
    searching them finds nothing rather than something wrong.
    """
    known = trace.call_by_id(call_id)
    earlier = ([c for c in trace.calls if c.order < known.order] if known
               else list(trace.calls))

    pending = TraceCall(id=call_id, order=len(trace.calls), step_id="", name="", type=ToolType.OPACA)
    return [
        _resolve(pending, path, value, earlier, trace.query)
        for path, value in flatten(args)
    ]


def build_edges(trace: ExecutionTrace) -> list[DataFlowEdge]:
    """Resolve every argument of every call to a source.

    Only calls *earlier in the sequence* are searched. A value cannot have come
    from a result that did not exist yet, and allowing it to would produce
    edges that read as derivations but are coincidences.
    """
    edges: list[DataFlowEdge] = []

    for call in trace.calls:
        earlier = [c for c in trace.calls if c.order < call.order]
        for path, value in flatten(call.args):
            edges.append(_resolve(call, path, value, earlier, trace.query))

    return edges


def _leaf_count(value: Any) -> int:
    return sum(1 for _ in flatten(value))


# A result with one leaf is a single value, not a haystack: the whole answer
# is the thing being matched. Two leaves is already a small object like
# {"count": 1, "ok": true}, where a 1 can line up with an unrelated 1.
_SINGLE_VALUE_LEAVES = 1


def _resolve(
        call: TraceCall,
        path: str,
        value: Any,
        earlier: list[TraceCall],
        query: str,
) -> DataFlowEdge:
    base = {"to_call_id": call.id, "to_param_path": path, "value": str(value)}

    if not is_informative(value):
        # Lookup actions return exactly this kind of value: GetRoomId answers
        # with 1, and that 1 becomes room_id in the next call. Refusing to
        # judge it loses the most common real chain there is.
        #
        # What makes the match safe is not the value but the source. When the
        # earlier result is a single value rather than a structure to hunt
        # through, an exact match is that result, not a coincidence inside it.
        # Anything larger goes back to being unjudgeable: in an object of two
        # or more fields a 1 can line up with an unrelated 1.
        for source_call in reversed(earlier):
            if _leaf_count(source_call.result) > _SINGLE_VALUE_LEAVES:
                continue
            found = _search(source_call.result, value, exact_only=True)
            if found:
                from_path, kind = found
                return DataFlowEdge(
                    **base,
                    source="tool_result",
                    kind=kind,
                    from_call_id=source_call.id,
                    from_call_name=source_call.name,
                    from_path=from_path or "result",
                )
        return DataFlowEdge(**base, source="unmatched", kind="weak")

    # Most recent result first: when a value occurs in several, the one the
    # model just saw is the likelier origin.
    for source_call in reversed(earlier):
        found = _search(source_call.result, value)
        if found:
            from_path, kind = found
            return DataFlowEdge(
                **base,
                source="tool_result",
                kind=kind,
                from_call_id=source_call.id,
                from_call_name=source_call.name,
                from_path=from_path or "result",
            )

    kind = _search_query(query, value)
    if kind:
        return DataFlowEdge(**base, source="user_query", kind=kind)

    # Last: the same value in an earlier call's arguments. This defers rather
    # than answers -- if that call's value was itself unmatched, so is this one
    # -- but it is not "invented", and reporting it as such would cry wolf on
    # every value the model carries from one step to the next.
    for source_call in reversed(earlier):
        found = _search(source_call.args, value)
        if found:
            from_path, kind = found
            return DataFlowEdge(
                **base,
                source="prior_argument",
                kind=kind,
                from_call_id=source_call.id,
                from_call_name=source_call.name,
                from_path=from_path,
            )

    return DataFlowEdge(**base, source="unmatched")


class ProvenanceSummary(BaseModel):
    """Counts over the edges, for the confidence signals and the UI badge."""
    total: int = 0
    from_query: int = 0
    from_results: int = 0
    reused: int = 0
    unmatched: int = 0
    weak: int = 0

    @property
    def coverage(self) -> float:
        """Share of judged parameters that resolved to a source.

        Weak parameters are excluded from both sides: they were never judged,
        so counting them either way would distort the number.
        """
        judged = self.total - self.weak
        if judged <= 0:
            return 1.0
        return (self.from_query + self.from_results) / judged


def summarise(edges: list[DataFlowEdge]) -> ProvenanceSummary:
    summary = ProvenanceSummary(total=len(edges))
    for edge in edges:
        if edge.kind == "weak":
            summary.weak += 1
        elif edge.source == "user_query":
            summary.from_query += 1
        elif edge.source == "tool_result":
            summary.from_results += 1
        elif edge.source == "prior_argument":
            summary.reused += 1
        else:
            summary.unmatched += 1
    return summary
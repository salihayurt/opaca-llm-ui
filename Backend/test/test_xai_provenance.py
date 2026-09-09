"""Resolving tool arguments to their source.

The module answers one question per argument: did this value come from the
user, from an earlier result, or from nowhere. The third answer is the useful
one, so the tests are weighted toward not producing it wrongly in either
direction -- a false `unmatched` cries wolf, a false match hides an invented
argument.
"""

import pytest

from src.models import AgentMessage, QueryResponse, ToolCall, ToolType
from src.xai.provenance import (
    build_edges, flatten, is_informative, normalise, summarise,
)
from src.xai.trace import build_trace


def _trace(calls, query="Book a room for tomorrow"):
    """One step holding the given (name, args, result) triples, in order."""
    tools = [
        ToolCall(id=f"m1/{i}", type=ToolType.OPACA, name=name, args=args, result=result)
        for i, (name, args, result) in enumerate(calls)
    ]
    return build_trace(QueryResponse(
        query=query, agent_messages=[AgentMessage(agent="a", tools=tools)]
    ))


def _edge(edges, path, call=-1):
    """The edge for one parameter. `call` picks which, when a name repeats."""
    matches = [e for e in edges if e.to_param_path == path]
    return matches[call] if call >= 0 else matches[-1]


class TestHelpers:
    def test_flatten_yields_json_paths(self):
        paths = dict(flatten({"a": {"b": 1}, "c": [10, 20]}))

        assert paths == {"a.b": 1, "c[0]": 10, "c[1]": 20}

    @pytest.mark.parametrize("left,right", [
        ("2026-08-27", "27.08.2026"),
        ("2026-08-27", "2026/08/27"),
        ("  Room  C.12 ", "room c.12"),
        ("2026-8-7", "07.08.2026"),
    ])
    def test_normalise_collapses_equivalent_forms(self, left, right):
        """Agents answer in ISO, users write German format, same day."""
        assert normalise(left) == normalise(right)

    def test_normalise_keeps_different_values_apart(self):
        assert normalise("2026-08-27") != normalise("2026-08-28")

    @pytest.mark.parametrize("value", ["C.12", "conference", "2026-08-27", 145000])
    def test_distinctive_values_are_judged(self, value):
        assert is_informative(value)

    @pytest.mark.parametrize("value", [True, False, None, "", "de", 1, 0, -1, "true", 42])
    def test_uninformative_values_are_not(self, value):
        """These occur in every result; edges drawn from them are noise."""
        assert not is_informative(value)


class TestSourceResolution:
    def test_a_value_from_an_earlier_result_is_linked(self):
        edges = build_edges(_trace([
            ("RoomAgent--GetFreeRooms", {}, {"rooms": [{"id": "C.12"}, {"id": "D.03"}]}),
            ("RoomAgent--BookRoom", {"room": "C.12"}, "ok"),
        ]))

        edge = _edge(edges, "room")
        assert edge.source == "tool_result"
        assert edge.kind == "exact"
        assert edge.from_call_name == "RoomAgent--GetFreeRooms"
        assert edge.from_path == "rooms[0].id"

    def test_a_value_from_the_users_message_is_linked(self):
        edges = build_edges(_trace(
            [("RoomAgent--BookRoom", {"room": "conference room"}, "ok")],
            query="Book the conference room please",
        ))

        assert _edge(edges, "room").source == "user_query"

    def test_a_reformatted_value_is_still_linked(self):
        """The case that makes normalisation worth having."""
        edges = build_edges(_trace([
            ("Cal--GetSlots", {}, {"date": "2026-08-27"}),
            ("Cal--Book", {"day": "27.08.2026"}, "ok"),
        ]))

        edge = _edge(edges, "day")
        assert edge.source == "tool_result"
        assert edge.kind == "normalised"

    def test_a_value_from_nowhere_is_flagged(self):
        """The finding the module exists for: an argument from outside."""
        edges = build_edges(_trace(
            [("RoomAgent--BookRoom", {"attendees": 8000}, "ok")],
            query="Book a room",
        ))

        edge = _edge(edges, "attendees")
        assert edge.source == "unmatched"
        assert not edge.is_grounded

    def test_nested_arguments_are_resolved_individually(self):
        edges = build_edges(_trace([
            ("A--search", {}, {"hits": ["Bauhaus Museum"]}),
            ("A--book", {"target": {"name": "Bauhaus Museum"}, "note": "Alpha77"}, "ok"),
        ]))

        assert _edge(edges, "target.name").source == "tool_result"
        assert _edge(edges, "note").source == "unmatched"


class TestOrdering:
    def test_only_earlier_calls_are_searched(self):
        """A value cannot come from a result that did not exist yet."""
        edges = build_edges(_trace([
            ("A--book", {"room": "C.12"}, "ok"),
            ("A--list", {}, {"rooms": ["C.12"]}),
        ]))

        assert _edge(edges, "room").source == "unmatched"

    def test_the_most_recent_source_wins(self):
        """When a value occurs twice, the one just seen is the likelier origin."""
        edges = build_edges(_trace([
            ("A--old", {}, {"room": "C.12"}),
            ("A--new", {}, {"room": "C.12"}),
            ("A--book", {"room": "C.12"}, "ok"),
        ]))

        assert _edge(edges, "room").from_call_name == "A--new"

    def test_a_result_beats_the_query(self):
        """Both contain it, but the model was looking at the result."""
        edges = build_edges(_trace(
            [("A--list", {}, {"room": "C.12"}), ("A--book", {"room": "C.12"}, "ok")],
            query="book room C.12",
        ))

        assert _edge(edges, "room").source == "tool_result"


class TestNoiseControl:
    def test_short_and_common_values_do_not_draw_edges(self):
        """`1` and `true` occur everywhere; matching them buries the real edges."""
        edges = build_edges(_trace([
            ("A--list", {}, {"count": 1, "ok": True}),
            ("A--book", {"limit": 1, "confirm": True}, "ok"),
        ]))

        assert _edge(edges, "limit").kind == "weak"
        assert _edge(edges, "confirm").kind == "weak"

    def test_a_weak_value_is_not_reported_as_invented(self):
        """We declined to judge it; saying "from nowhere" would be a claim."""
        edges = build_edges(_trace([("A--b", {"n": 1}, "ok")]))

        assert _edge(edges, "n").kind == "weak"
        assert not _edge(edges, "n").is_grounded

    def test_an_exact_match_wins_over_a_substring_elsewhere(self):
        """A weaker match further along must not beat a stronger one."""
        edges = build_edges(_trace([
            ("A--list", {}, {"note": "prefer C.12 today", "id": "C.12"}),
            ("A--book", {"room": "C.12"}, "ok"),
        ]))

        edge = _edge(edges, "room")
        assert edge.kind == "exact"
        assert edge.from_path == "id"


class TestSummary:
    def test_counts_add_up(self):
        edges = build_edges(_trace(
            [
                ("A--list", {}, {"room": "C.12"}),
                ("A--book", {"room": "C.12", "note": "Zeta9981", "n": 1}, "ok"),
            ],
            query="book something",
        ))

        summary = summarise(edges)
        assert summary.total == 3
        assert summary.from_results == 1
        assert summary.unmatched == 1
        assert summary.weak == 1

    def test_coverage_ignores_weak_parameters(self):
        """They were never judged, so counting them either way distorts it."""
        edges = build_edges(_trace([
            ("A--list", {}, {"room": "C.12"}),
            ("A--book", {"room": "C.12", "n": 1}, "ok"),
        ]))

        assert summarise(edges).coverage == 1.0

    def test_a_call_without_arguments_yields_nothing_to_report(self):
        edges = build_edges(_trace([("ScheduledTasks--GetScheduledTasks", {}, {})]))

        assert edges == []
        assert summarise(edges).coverage == 1.0


class TestCarriedForwardValues:
    """A value reused from an earlier call's arguments is not an invented one.

    Without this the matcher cries wolf on every value the model carries from
    one step to the next, which is most of them in a multi-step chain, and a
    warning that fires constantly is one nobody reads.
    """

    def test_a_value_reused_from_an_earlier_call_is_linked(self):
        edges = build_edges(_trace([
            ("Cal--GetSlots", {"date": "2026-08-28"}, {"slots": ["09:00"]}),
            ("Cal--Book", {"day": "28.08.2026"}, "ok"),
        ], query="book me a slot"))

        edge = _edge(edges, "day")
        assert edge.source == "prior_argument"
        assert edge.from_call_name == "Cal--GetSlots"
        assert edge.from_path == "date"

    def test_reuse_does_not_count_as_grounded(self):
        """It says where the value was seen before, not where it came from."""
        edges = build_edges(_trace([
            ("A--first", {"code": "Zeta9981"}, "ok"),
            ("A--second", {"code": "Zeta9981"}, "ok"),
        ], query="do the thing"))

        assert not _edge(edges, "code", call=1).is_grounded

    def test_a_real_source_still_wins_over_reuse(self):
        edges = build_edges(_trace([
            ("A--first", {"room": "C.12"}, {"room": "C.12"}),
            ("A--second", {"room": "C.12"}, "ok"),
        ], query="do the thing"))

        assert _edge(edges, "room", call=1).source == "tool_result"

    def test_the_query_wins_over_reuse(self):
        """The query is the root; an earlier argument only got it from there."""
        edges = build_edges(_trace([
            ("A--first", {"room": "Bauhaus"}, "ok"),
            ("A--second", {"room": "Bauhaus"}, "ok"),
        ], query="book the Bauhaus room"))

        assert _edge(edges, "room", call=1).source == "user_query"

    def test_genuinely_invented_values_are_still_flagged(self):
        """The alarm must still fire, or the change has broken the point."""
        edges = build_edges(_trace([
            ("A--list", {}, {"rooms": ["C.12"]}),
            ("A--book", {"room": "C.12", "organizer": "Dr. Weber"}, "ok"),
        ], query="book a room"))

        assert _edge(edges, "organizer").source == "unmatched"


class TestRealTraceRegressions:
    """Cases found by running the matcher over stored conversations.

    Both are the same failure: a value the user supplied, reported as invented.
    That is the loudest possible way to be wrong, and once the flag fires on
    ordinary values it stops carrying information.
    """

    def test_accented_forms_match_ascii_typed_queries(self):
        """Users type ASCII; models emit the properly accented form."""
        edges = build_edges(_trace(
            [("Chats--SearchChats", {"q": "nasıl çalışan bir sistem"}, "ok")],
            query="sage/opaca nasil calisan bir sistem",
        ))

        assert _edge(edges, "q").source == "user_query"

    def test_german_umlauts_fold(self):
        edges = build_edges(_trace(
            [("A--search", {"city": "Köln Hauptbahnhof"}, "ok")],
            query="trains from koln hauptbahnhof",
        ))

        assert _edge(edges, "city").source == "user_query"

    def test_eszett_folds(self):
        edges = build_edges(_trace(
            [("A--search", {"street": "Hauptstraße"}, "ok")],
            query="who lives on Hauptstrasse",
        ))

        assert _edge(edges, "street").source == "user_query"

    def test_a_reworded_search_query_is_traced_to_the_user(self):
        """Search arguments get reordered, not copied."""
        edges = build_edges(_trace(
            [("Docs--SearchDocuments", {"query": "air filter part number"}, "ok")],
            query="What is the part number for the air filter?",
        ))

        edge = _edge(edges, "query")
        assert edge.source == "user_query"
        assert edge.kind == "reworded"

    def test_rewording_needs_every_word_to_come_from_the_query(self):
        """Otherwise the model can smuggle a new term into a real one."""
        edges = build_edges(_trace(
            [("Docs--SearchDocuments", {"query": "air filter warranty period"}, "ok")],
            query="What is the part number for the air filter?",
        ))

        assert _edge(edges, "query").source == "unmatched"

    def test_an_invented_url_is_still_flagged(self):
        """A real hallucination from a stored chat: a placeholder document URL."""
        edges = build_edges(_trace(
            [("Files--ReadFileFromUrl", {"url": "https://example.com/document.pdf"}, "ok")],
            query="What does Error 22 mean according to this document",
        ))

        assert _edge(edges, "url").source == "unmatched"

    def test_a_converted_unit_is_reported_as_untraced(self):
        """"every 30 minutes" becoming 1800 is inference, not copying.

        Flagging it is correct -- the value did not come from the conversation
        -- but it is the honest limit of the method: we detect what was not
        copied, not what is wrong. The inference here happens to be right.
        """
        edges = build_edges(_trace(
            [("Tasks--ScheduleIntervalTask", {"delay_seconds": 1800}, "ok")],
            query="remind me about the filter check every 30 minutes",
        ))

        assert _edge(edges, "delay_seconds").source == "unmatched"


class TestLookupChains:
    """GetRoomId returns 1; that 1 becomes room_id in the next call.

    The most common chain in the smart-office agents, and the first real one
    tried against this code, where it drew no arrow: the value is too
    unremarkable to judge on its own. What makes it safe is the source. When
    the earlier result is a single value rather than a structure to hunt
    through, an exact match is that result, not a coincidence inside it.
    """

    def test_a_scalar_result_links_even_for_a_small_value(self):
        edges = build_edges(_trace([
            ("RoomAgent--GetRoomId", {"room_name": "Experience Hub"}, 1),
            ("SensorAgent--GetCo2Level", {"room_id": 1}, 600),
        ], query="What's the CO2 level in the Experience Hub?"))

        edge = _edge(edges, "room_id")
        assert edge.source == "tool_result"
        assert edge.from_call_name == "RoomAgent--GetRoomId"

    def test_a_single_key_result_counts_as_one_value(self):
        edges = build_edges(_trace([
            ("A--lookup", {}, {"id": 3}),
            ("B--use", {"thing_id": 3}, "ok"),
        ], query="do the thing"))

        assert _edge(edges, "thing_id").source == "tool_result"

    def test_a_small_value_inside_a_larger_result_still_does_not_link(self):
        """The case the rule guards: 1 occurs all over a structure."""
        edges = build_edges(_trace([
            ("A--list", {}, {"rooms": [{"id": 7, "floor": 1, "seats": 8},
                                       {"id": 9, "floor": 1, "seats": 4}],
                             "count": 2, "page": 1, "total": 1}),
            ("B--use", {"floor": 1}, "ok"),
        ], query="do the thing"))

        assert _edge(edges, "floor").kind == "weak"

    def test_a_single_value_source_still_requires_an_exact_match(self):
        """Anything looser than exact would match by accident at this size."""
        edges = build_edges(_trace([
            ("A--lookup", {}, 12),
            ("B--use", {"n": 1}, "ok"),
        ], query="do the thing"))

        assert _edge(edges, "n").kind == "weak"

    def test_the_lookup_argument_still_traces_to_the_user(self):
        """The whole chain: name from the user, id from the lookup."""
        edges = build_edges(_trace([
            ("RoomAgent--GetRoomId", {"room_name": "Experience Hub"}, 1),
            ("SensorAgent--GetCo2Level", {"room_id": 1}, 600),
        ], query="What's the CO2 level in the Experience Hub?"))

        assert _edge(edges, "room_name").source == "user_query"
        assert _edge(edges, "room_id").source == "tool_result"
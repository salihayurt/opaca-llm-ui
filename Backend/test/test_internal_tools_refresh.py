"""
Tests that every path which lists internal tools refreshes them first.

This file exists because of a bug the other tests could not see. Tools are
gathered from four places -- AbstractMethod.get_tools, SimpleBackend's
get_actions, the orchestrated worker setup, and the /internal-tools route --
and the refresh hook was originally added to only one of them. The document
tools were therefore invisible under the `simple` method: the model answered
from chat history and, when that failed, invented an answer, with no error
anywhere to show that a tool group had silently stayed empty.

The unit tests all called refresh() explicitly, so they proved the group
works when refreshed and said nothing about whether anything refreshes it.
These tests exercise the accessors instead.
"""

import pytest

from src.internal_tools.registry import InternalTools


class RecordingGroup:
    """A tool group that records whether it was refreshed before being listed."""

    GROUP_NAME = "Recording"

    def __init__(self, ctx):
        self.refreshed = False
        self.listed_without_refresh = False

    async def refresh(self):
        self.refreshed = True

    def tools(self):
        if not self.refreshed:
            self.listed_without_refresh = True
        return []


@pytest.fixture
def tools(monkeypatch):
    """InternalTools carrying only the recording group."""
    import src.internal_tools.registry as registry

    monkeypatch.setattr(registry, "TOOL_GROUPS", (RecordingGroup,))
    instance = InternalTools(session=None, agent_method=None)
    return instance


@pytest.mark.anyio
async def test_openai_format_refreshes_first(tools):
    """Used by AbstractMethod.get_tools and by the orchestrated worker setup."""
    await tools.get_internal_tools_openai()
    assert tools.groups[0].refreshed
    assert not tools.groups[0].listed_without_refresh


@pytest.mark.anyio
async def test_simple_format_refreshes_first(tools):
    """Used by the `simple` method -- the path where the bug appeared."""
    await tools.get_internal_tools_simple()
    assert tools.groups[0].refreshed
    assert not tools.groups[0].listed_without_refresh


@pytest.mark.anyio
async def test_container_format_refreshes_first(tools):
    """Used by the /internal-tools route that populates the UI."""
    await tools.get_internal_tools_containers()
    assert tools.groups[0].refreshed
    assert not tools.groups[0].listed_without_refresh


@pytest.mark.anyio
async def test_a_failing_refresh_does_not_lose_the_other_tools(monkeypatch):
    """One group failing to refresh must not cost the user every other tool in
    the turn."""
    class BrokenGroup:
        GROUP_NAME = "Broken"

        def __init__(self, ctx):
            pass

        async def refresh(self):
            raise RuntimeError("vector store unreachable")

        def tools(self):
            return []

    import src.internal_tools.registry as registry
    monkeypatch.setattr(registry, "TOOL_GROUPS", (BrokenGroup, RecordingGroup))
    instance = InternalTools(session=None, agent_method=None)

    await instance.get_internal_tools_openai()  # must not raise

    assert instance.groups[1].refreshed


@pytest.mark.anyio
async def test_groups_without_a_refresh_hook_are_skipped(monkeypatch):
    """Most groups have no async state and define no refresh."""
    class PlainGroup:
        GROUP_NAME = "Plain"

        def __init__(self, ctx):
            pass

        def tools(self):
            return []

    import src.internal_tools.registry as registry
    monkeypatch.setattr(registry, "TOOL_GROUPS", (PlainGroup,))
    instance = InternalTools(session=None, agent_method=None)

    assert await instance.get_internal_tools_openai() == []


def test_every_accessor_is_a_coroutine():
    """A synchronous accessor would be one a caller could use without
    refreshing, which is how the original bug was possible."""
    import inspect

    for name in (
        "get_internal_tools_simple",
        "get_internal_tools_openai",
        "get_internal_tools_containers",
    ):
        assert inspect.iscoroutinefunction(getattr(InternalTools, name)), name
"""Every step says what it was for.

Step 1 added the field; this is the part that fills it. Without it every step
reads as TOOL_CALL, which is what the trace looked like on the first real run:
evaluators, retries and the final answer all indistinguishable from an action.

The tests are structural rather than behavioural -- they read the source for
call sites that forgot to say -- because exercising the real thing needs a
platform and four LLM calls, and a missed site is silent: the default is valid,
so nothing fails, the trace is just wrong.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

METHOD_FILES = [
    "toolllm/tool_routes.py",
    "simple/simple_routes.py",
    "simple_tools/simple_tools_routes.py",
    "orchestrated/orchestrated_routes.py",
]


def _call_llm_sites(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "call_llm"):
            yield node


def _kwargs(call) -> set[str]:
    return {kw.arg for kw in call.keywords if kw.arg}


@pytest.mark.parametrize("filename", METHOD_FILES)
def test_every_call_says_which_round_it_belongs_to(filename):
    """Without it every step reads as round 0 and the chain view collapses."""
    for call in _call_llm_sites(SRC / filename):
        assert "iteration" in _kwargs(call), (
            f"{filename}:{call.lineno} calls call_llm without iteration=")


@pytest.mark.parametrize("filename", [
    "toolllm/tool_routes.py", "orchestrated/orchestrated_routes.py",
])
def test_multi_agent_methods_type_every_step(filename):
    """simple and simple-tools are excluded: one call does both jobs there, so
    the type is only knowable from what came back and is set afterwards."""
    for call in _call_llm_sites(SRC / filename):
        assert "step_type" in _kwargs(call), (
            f"{filename}:{call.lineno} calls call_llm without step_type=")


def test_the_correction_loop_is_not_typed_as_an_ordinary_call():
    """A retry is not a second freely chosen action, and a reader that cannot
    tell them apart reports one as the other."""
    source = (SRC / "toolllm/tool_routes.py").read_text()

    assert "StepType.CORRECTION" in source


def test_workers_are_linked_to_the_step_that_spawned_them():
    """Workers are appended from inside asyncio.gather, so list order says
    nothing. The parent link is the only way to rebuild the structure."""
    source = (SRC / "orchestrated/orchestrated_routes.py").read_text()

    assert "parent_id=orchestrator_message.id" in source
    assert source.count("parent_id=parent_id") >= 3


def test_the_single_agent_methods_mark_their_final_answer():
    for filename in ["simple/simple_routes.py", "simple_tools/simple_tools_routes.py"]:
        source = (SRC / filename).read_text()
        assert "step_type = StepType.OUTPUT" in source, filename
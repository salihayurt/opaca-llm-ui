"""
Tests for src/internal_tools/documents.py.

What matters here is not retrieval quality but what the model sees: when the
tool appears, what its description says, and what its result tells the model
to do. All three drive tool-selection accuracy, which is the metric SAGE's
own benchmark reports.
"""

import warnings

import pytest
from qdrant_client import AsyncQdrantClient

from src.internal_tools.documents import DocumentTools
from src.models import OpacaFile, SessionData
from src.rag.service import RagConfig, RagService
from src.rag.store import DocumentStore

VECTOR_SIZE = 8


class FakeEmbedder:
    def _vector(self, text):
        vector = [0.0] * VECTOR_SIZE
        for character in text.lower():
            if character.isalnum():
                vector[ord(character) % VECTOR_SIZE] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    async def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    async def embed_query(self, text):
        return self._vector(text)

    async def dimensions(self):
        return VECTOR_SIZE


class FakeContext:
    """Only the parts of InternalToolContext the group touches."""

    def __init__(self, session, chat_id="chat-1"):
        self.session = session
        self.chat_id = chat_id


@pytest.fixture
async def setup(tmp_path, monkeypatch):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
        client = AsyncQdrantClient(":memory:")
        service = RagService(
            DocumentStore(client, vector_size=VECTOR_SIZE),
            FakeEmbedder(),
            config=RagConfig(min_index_chars=200),
        )
        session = SessionData(session_id="s1")

        import src.file_utils as file_utils
        monkeypatch.setattr(file_utils, "FILES_PATH", str(tmp_path), raising=False)

        tools = DocumentTools(FakeContext(session), service=service)
        yield tools, session, service, tmp_path
        await client.close()


def add_file(session, name: str, content: bytes) -> OpacaFile:
    """Register an uploaded file and write its bytes where the tool looks."""
    from src.file_utils import create_path

    file = OpacaFile(content_type="text/plain", file_name=name)
    session.uploaded_files[file.file_id] = file
    path = create_path(session.session_id, file.file_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return file


def long_text(topic: str, characters: int = 1200) -> bytes:
    sentence = f"This section explains {topic} for building operations. "
    body = ""
    while len(body) < characters:
        body += sentence
    return body.encode()


def tool_names(tools) -> list[str]:
    return [tool.name for tool in tools]


# -- when the tool appears --------------------------------------------------

@pytest.mark.anyio
async def test_no_tool_when_nothing_is_uploaded(setup):
    """SAGE spends around 13k prompt tokens on a simple question already, and
    is judged on tool-selection accuracy. A session that uploads nothing
    should pay nothing for RAG."""
    tools, *_ = setup
    await tools.refresh()
    assert tools.tools() == []


@pytest.mark.anyio
async def test_search_is_offered_before_anything_is_indexed(setup):
    """The failure this design exists to prevent: with the search tool hidden
    until something was indexed, the model reached for SearchChats instead --
    which puts the whole transcript into an unbounded LLM call."""
    tools, session, _, _ = setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()
    assert tool_names(tools.tools()) == ["SearchDocuments"]


@pytest.mark.anyio
async def test_exactly_one_tool_is_ever_exposed(setup):
    tools, session, _, _ = setup
    for name in ("a.txt", "b.txt", "c.txt"):
        add_file(session, name, long_text("thermostats"))
    await tools.refresh()
    assert len(tools.tools()) == 1


@pytest.mark.anyio
async def test_unsupported_files_do_not_summon_the_tool(setup):
    """Offering a search over a file we cannot read spends prompt budget to
    produce an apology."""
    tools, session, _, _ = setup
    add_file(session, "photo.png", b"\x89PNG" + b"x" * 2000)
    await tools.refresh()
    assert tools.tools() == []


@pytest.mark.anyio
async def test_no_tool_when_rag_is_unavailable(setup):
    """Qdrant not running is an ordinary state. SAGE must behave exactly as it
    did before RAG existed."""
    tools, session, _, _ = setup
    add_file(session, "manual.txt", long_text("thermostats"))
    tools.service = None
    assert tools.tools() == []


@pytest.mark.anyio
async def test_a_failing_vector_store_does_not_break_the_turn(setup):
    tools, session, service, _ = setup
    add_file(session, "manual.txt", long_text("thermostats"))

    async def explode(*args, **kwargs):
        raise RuntimeError("connection refused")

    service.list_documents = explode
    await tools.refresh()  # must not raise


# -- description ------------------------------------------------------------

@pytest.mark.anyio
async def test_description_names_the_documents(setup):
    """Without the names the model guesses both ways: searching when nothing
    relevant is uploaded, and answering from memory when the answer was in a
    manual sitting right there."""
    tools, session, _, _ = setup
    add_file(session, "hvac-manual.txt", long_text("thermostats"))
    await tools.refresh()

    description = tools.tools()[0].description
    assert "hvac-manual.txt" in description


@pytest.mark.anyio
async def test_description_steers_away_from_chat_history(setup):
    """SearchChats advertises itself as searching 'this and past interactions
    about information on the given topic', which reads as a match for almost
    any question."""
    tools, session, _, _ = setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()

    description = tools.tools()[0].description.lower()
    assert "past conversations" in description


@pytest.mark.anyio
async def test_description_lists_documents_not_yet_indexed(setup):
    """Indexing is an implementation detail; from the model's side an
    uploaded document is searchable."""
    tools, session, _, _ = setup
    add_file(session, "fresh.txt", long_text("valves"))
    await tools.refresh()

    assert "fresh.txt" in tools.tools()[0].description


# -- indexing on demand -----------------------------------------------------

@pytest.mark.anyio
async def test_searching_indexes_an_unprepared_document(setup):
    """The user should not have to say 'index this' before asking about it."""
    tools, session, service, _ = setup
    add_file(session, "manual.txt", long_text("thermostat resets"))
    await tools.refresh()

    answer = await tools.tool_search_documents("thermostat")

    assert "thermostat" in answer.lower()
    assert await service.list_documents(session.session_id)


@pytest.mark.anyio
async def test_preparation_is_reported_to_the_model(setup):
    """A slow first search should be explained rather than silent."""
    tools, session, _, _ = setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()

    answer = await tools.tool_search_documents("thermostat")
    assert "Prepared for search" in answer
    assert "manual.txt" in answer


@pytest.mark.anyio
async def test_a_document_is_not_indexed_twice(setup):
    tools, session, service, _ = setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()

    await tools.tool_search_documents("thermostat")
    second = await tools.tool_search_documents("thermostat")

    assert "Prepared for search" not in second


@pytest.mark.anyio
async def test_a_refused_document_is_explained(setup):
    """Small files are refused deliberately -- retrieval would hand the model
    fragments of something it could read whole. The model should learn why a
    document is missing from the results."""
    tools, session, _, _ = setup
    add_file(session, "big.txt", long_text("thermostats"))
    add_file(session, "note.txt", b"A short note.")
    await tools.refresh()

    answer = await tools.tool_search_documents("thermostat")
    assert "note.txt" in answer
    assert "small enough to read directly" in answer


@pytest.mark.anyio
async def test_a_missing_file_on_disk_does_not_break_the_search(setup):
    """The record and the bytes can diverge; the other documents should still
    be searchable."""
    from src.file_utils import create_path

    tools, session, _, _ = setup
    good = add_file(session, "manual.txt", long_text("thermostats"))
    gone = add_file(session, "ghost.txt", long_text("valves"))
    create_path(session.session_id, gone.file_id).unlink()
    await tools.refresh()

    answer = await tools.tool_search_documents("thermostat")
    assert "ghost.txt" in answer
    assert "thermostat" in answer.lower()


# -- the result the model reads ---------------------------------------------

@pytest.mark.anyio
async def test_result_asks_for_citations(setup):
    """The instruction has to be in the result, not the description. The
    description is read when choosing a tool; by the time the model writes
    the answer, this text is what is in context. Observed with the
    instruction in the description only: correct answers, no citations."""
    tools, session, _, _ = setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()

    answer = await tools.tool_search_documents("thermostat")
    assert "[1]" in answer
    assert "Answer only from these passages" in answer


@pytest.mark.anyio
async def test_result_attributes_each_passage(setup):
    tools, session, _, _ = setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()

    answer = await tools.tool_search_documents("thermostat")
    assert "[1] manual.txt" in answer


@pytest.mark.anyio
async def test_no_match_tells_the_model_not_to_improvise(setup):
    """An empty result would leave the model free to invent an answer and
    present it as coming from the document."""
    tools, session, service, _ = setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()
    await tools.tool_search_documents("thermostat")

    service.retriever.config.min_dense_score = 0.99
    service.retriever.config.hybrid = False

    answer = await tools.tool_search_documents("zzzzz unrelated qqqqq")
    assert "manual.txt" in answer
    assert "do not cover it" in answer


@pytest.mark.anyio
async def test_empty_query_is_handled(setup):
    tools, session, _, _ = setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()

    assert "nothing was searched" in await tools.tool_search_documents("  ")


@pytest.mark.anyio
async def test_search_failure_is_reported_not_raised(setup):
    tools, session, service, _ = setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()

    async def explode(*args, **kwargs):
        raise RuntimeError("store unreachable")

    service.search = explode
    assert "failed" in await tools.tool_search_documents("thermostat")


# -- sources published to the frontend --------------------------------------

class RecordingSession(SessionData):
    """A session that records what was pushed over the websocket."""

    def model_post_init(self, __context):
        object.__setattr__(self, "_sent", []) if False else None
        self.__dict__["_sent"] = []

    async def websocket_send(self, message):
        if self.__dict__.get("_fail"):
            raise RuntimeError("socket closed")
        self.__dict__["_sent"].append(message)
        return True

    @property
    def sent(self):
        return self.__dict__.get("_sent", [])


@pytest.fixture
async def recording_setup(tmp_path, monkeypatch):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
        client = AsyncQdrantClient(":memory:")
        service = RagService(
            DocumentStore(client, vector_size=VECTOR_SIZE),
            FakeEmbedder(),
            config=RagConfig(min_index_chars=200),
        )
        session = RecordingSession(session_id="s2")

        import src.file_utils as file_utils
        monkeypatch.setattr(file_utils, "FILES_PATH", str(tmp_path), raising=False)

        yield DocumentTools(FakeContext(session), service=service), session, service
        await client.close()


@pytest.mark.anyio
async def test_sources_are_published_to_the_frontend(recording_setup):
    """The answer's citations depend on the model complying with an
    instruction, and on the output generator's prompt, which says nothing
    about citing. This message does not depend on either: whatever the model
    writes, the user still sees where the answer came from."""
    tools, session, _ = recording_setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()

    await tools.tool_search_documents("thermostat")

    published = [m for m in session.sent if type(m).__name__ == "DocumentSourcesMessage"]
    assert published, "no sources reached the frontend"
    assert published[0].chat_id == "chat-1"
    assert published[0].sources[0]["filename"] == "manual.txt"


@pytest.mark.anyio
async def test_sources_carry_what_a_citation_needs(recording_setup):
    tools, session, _ = recording_setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()

    await tools.tool_search_documents("thermostat")

    source = [m for m in session.sent
              if type(m).__name__ == "DocumentSourcesMessage"][0].sources[0]
    assert {"number", "filename", "file_id", "location", "chunk_index"} <= set(source)


@pytest.mark.anyio
async def test_nothing_is_published_when_there_is_no_match(recording_setup):
    """An empty source panel below an answer that came from general knowledge
    would be worse than no panel."""
    tools, session, service = recording_setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()
    await tools.tool_search_documents("thermostat")
    session.sent.clear()

    service.retriever.config.min_dense_score = 0.99
    service.retriever.config.hybrid = False
    await tools.tool_search_documents("zzzzz unrelated qqqqq")

    assert not [m for m in session.sent if type(m).__name__ == "DocumentSourcesMessage"]


@pytest.mark.anyio
async def test_a_websocket_failure_does_not_fail_the_search(recording_setup):
    """The answer is still correct without the source panel."""
    tools, session, _ = recording_setup
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()

    session.__dict__["_fail"] = True

    answer = await tools.tool_search_documents("thermostat")
    assert "thermostat" in answer.lower()


@pytest.mark.anyio
async def test_nothing_is_published_outside_a_chat(setup):
    """InternalTools is also built outside a turn -- by the /internal-tools
    route and when resuming scheduled tasks -- where there is no chat to
    address a message to."""
    tools, session, _, _ = setup
    tools.ctx.chat_id = None
    add_file(session, "manual.txt", long_text("thermostats"))
    await tools.refresh()

    await tools.tool_search_documents("thermostat")  # must not raise


# -- the group contract -----------------------------------------------------

def test_group_name_is_stable():
    """registry.py routes tool calls by group name, and the frontend shows it."""
    assert DocumentTools.GROUP_NAME == "Documents"
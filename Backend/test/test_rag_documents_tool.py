"""
Tests for src/internal_tools/documents.py.

The tool group is what the model actually sees, so what matters here is not
retrieval quality but *when* tools appear and what their descriptions say.
Both drive tool-selection accuracy, which is the metric SAGE's own benchmark
reports.
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

    def __init__(self, session):
        self.session = session


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

        # Files live on disk under a configurable root; point it at tmp_path
        # so the tests do not touch a real upload directory.
        import src.file_utils as file_utils
        monkeypatch.setattr(file_utils, "FILES_DIR", tmp_path, raising=False)

        tools = DocumentTools(FakeContext(session), service=service)
        yield tools, session, service, tmp_path
        await client.close()


def add_file(session, tmp_path, name: str, content: bytes) -> OpacaFile:
    """Register an uploaded file and write its bytes where the group looks."""
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


# -- conditional exposure ---------------------------------------------------

@pytest.mark.anyio
async def test_no_tools_when_nothing_is_uploaded(setup, anyio_backend):
    """SAGE spends around 13k prompt tokens on a simple question already, and
    is judged on tool-selection accuracy. A session that never uploads
    anything should pay nothing for RAG."""
    tools, *_ = setup
    await tools.refresh()
    assert tools.tools() == []


@pytest.mark.anyio
async def test_only_indexing_is_offered_before_anything_is_indexed(setup, anyio_backend):
    tools, session, _, tmp_path = setup
    add_file(session, tmp_path, "manual.txt", long_text("thermostats"))
    await tools.refresh()
    assert tool_names(tools.tools()) == ["IndexDocument"]


@pytest.mark.anyio
async def test_search_appears_once_something_is_indexed(setup, anyio_backend):
    tools, session, _, tmp_path = setup
    file = add_file(session, tmp_path, "manual.txt", long_text("thermostats"))
    await tools.tool_index_document(file.file_id)

    assert "SearchDocuments" in tool_names(tools.tools())


@pytest.mark.anyio
async def test_an_indexed_file_is_not_offered_for_indexing_again(setup, anyio_backend):
    """Offering it invites a wasted call that re-embeds a document already
    stored."""
    tools, session, _, tmp_path = setup
    file = add_file(session, tmp_path, "manual.txt", long_text("thermostats"))
    await tools.tool_index_document(file.file_id)

    assert tool_names(tools.tools()) == ["SearchDocuments"]


@pytest.mark.anyio
async def test_unsupported_files_are_not_offered_for_indexing(setup, anyio_backend):
    """Offering a call that can only fail spends prompt budget to produce an
    error."""
    tools, session, _, tmp_path = setup
    add_file(session, tmp_path, "photo.png", b"\x89PNG" + b"x" * 2000)
    await tools.refresh()
    assert tools.tools() == []


@pytest.mark.anyio
async def test_no_tools_when_rag_is_unavailable(setup, anyio_backend):
    """Qdrant not running is an ordinary state. SAGE must behave exactly as it
    did before RAG existed."""
    tools, session, _, tmp_path = setup
    add_file(session, tmp_path, "manual.txt", long_text("thermostats"))
    tools.service = None
    assert tools.tools() == []


@pytest.mark.anyio
async def test_a_failing_vector_store_silences_the_group(setup, anyio_backend):
    """An unreachable store must not break the turn."""
    tools, session, service, tmp_path = setup
    add_file(session, tmp_path, "manual.txt", long_text("thermostats"))

    async def explode(*args, **kwargs):
        raise RuntimeError("connection refused")

    service.list_documents = explode
    await tools.refresh()
    assert "SearchDocuments" not in tool_names(tools.tools())


# -- descriptions -----------------------------------------------------------

@pytest.mark.anyio
async def test_search_description_names_the_documents(setup, anyio_backend):
    """Without them the model guesses both ways: searching when nothing
    relevant is indexed, and answering from memory when the answer was in a
    manual."""
    tools, session, _, tmp_path = setup
    file = add_file(session, tmp_path, "hvac-manual.txt", long_text("thermostats"))
    await tools.tool_index_document(file.file_id)

    search = next(t for t in tools.tools() if t.name == "SearchDocuments")
    assert "hvac-manual.txt" in search.description


@pytest.mark.anyio
async def test_search_description_asks_for_citations(setup, anyio_backend):
    tools, session, _, tmp_path = setup
    file = add_file(session, tmp_path, "manual.txt", long_text("thermostats"))
    await tools.tool_index_document(file.file_id)

    search = next(t for t in tools.tools() if t.name == "SearchDocuments")
    assert "[n]" in search.description


@pytest.mark.anyio
async def test_index_description_lists_file_ids(setup, anyio_backend):
    """The model has to pass one, so it must be able to read them off the
    description."""
    tools, session, _, tmp_path = setup
    file = add_file(session, tmp_path, "manual.txt", long_text("thermostats"))
    await tools.refresh()

    index_tool = next(t for t in tools.tools() if t.name == "IndexDocument")
    assert file.file_id in index_tool.description
    assert "manual.txt" in index_tool.description


# -- indexing ---------------------------------------------------------------

@pytest.mark.anyio
async def test_indexing_reports_success_and_next_step(setup, anyio_backend):
    tools, session, _, tmp_path = setup
    file = add_file(session, tmp_path, "manual.txt", long_text("thermostats"))

    message = await tools.tool_index_document(file.file_id)
    assert "searchable" in message
    assert "SearchDocuments" in message


@pytest.mark.anyio
async def test_indexing_a_small_file_explains_why_not(setup, anyio_backend):
    tools, session, _, tmp_path = setup
    file = add_file(session, tmp_path, "note.txt", b"Just a short note.")

    message = await tools.tool_index_document(file.file_id)
    assert "small enough to read directly" in message


@pytest.mark.anyio
async def test_indexing_an_unknown_id_lists_the_real_ones(setup, anyio_backend):
    tools, session, _, tmp_path = setup
    file = add_file(session, tmp_path, "manual.txt", long_text("thermostats"))

    message = await tools.tool_index_document("not-a-real-id")
    assert "No uploaded file" in message
    assert file.file_id in message


@pytest.mark.anyio
async def test_a_missing_file_on_disk_is_reported(setup, anyio_backend):
    """The record and the bytes can diverge; the model should learn why rather
    than see a traceback."""
    from src.file_utils import create_path

    tools, session, _, tmp_path = setup
    file = add_file(session, tmp_path, "manual.txt", long_text("thermostats"))
    create_path(session.session_id, file.file_id).unlink()

    message = await tools.tool_index_document(file.file_id)
    assert "Could not read" in message


# -- searching --------------------------------------------------------------

@pytest.mark.anyio
async def test_search_returns_attributed_passages(setup, anyio_backend):
    tools, session, _, tmp_path = setup
    file = add_file(session, tmp_path, "manual.txt", long_text("thermostat resets"))
    await tools.tool_index_document(file.file_id)

    answer = await tools.tool_search_documents("thermostat")
    assert "[1] manual.txt" in answer


@pytest.mark.anyio
async def test_search_with_no_match_says_so_and_suggests_what_to_do(setup, anyio_backend):
    """Returning an empty string would leave the model to invent an answer and
    present it as coming from the document.

    Reaching this state at all requires a relevance floor: without one, dense
    search returns the nearest chunks whatever their similarity, so an
    unrelated question still gets passages back. The floor is off by default
    and set here to exercise the branch -- which value it should have is a
    measurement, not a guess.
    """
    tools, session, service, tmp_path = setup
    file = add_file(session, tmp_path, "manual.txt", long_text("thermostats"))
    await tools.tool_index_document(file.file_id)

    service.retriever.config.min_dense_score = 0.99
    service.retriever.config.hybrid = False

    answer = await tools.tool_search_documents("zzzzz unrelated qqqqq")
    assert "manual.txt" in answer
    assert "do not cover it" in answer


@pytest.mark.anyio
async def test_empty_query_is_handled(setup, anyio_backend):
    tools, session, _, tmp_path = setup
    file = add_file(session, tmp_path, "manual.txt", long_text("thermostats"))
    await tools.tool_index_document(file.file_id)

    assert "nothing was searched" in await tools.tool_search_documents("  ")


@pytest.mark.anyio
async def test_search_failure_is_reported_not_raised(setup, anyio_backend):
    tools, session, service, tmp_path = setup
    file = add_file(session, tmp_path, "manual.txt", long_text("thermostats"))
    await tools.tool_index_document(file.file_id)

    async def explode(*args, **kwargs):
        raise RuntimeError("store unreachable")

    service.search = explode
    assert "failed" in await tools.tool_search_documents("thermostat")


# -- the group contract -----------------------------------------------------

def test_group_name_is_stable():
    """registry.py routes tool calls by group name, and the frontend shows it."""
    assert DocumentTools.GROUP_NAME == "Documents"


@pytest.mark.anyio
async def test_at_most_two_tools_are_ever_exposed(setup, anyio_backend):
    """Activate, deactivate, list and remove are user management actions with
    a sidebar, not things a model reasons about mid-task."""
    tools, session, _, tmp_path = setup
    for name in ("a.txt", "b.txt", "c.txt"):
        add_file(session, tmp_path, name, long_text("thermostats"))
    await tools.refresh()
    assert len(tools.tools()) <= 2
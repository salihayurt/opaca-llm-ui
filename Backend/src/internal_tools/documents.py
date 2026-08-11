from __future__ import annotations

import logging
from textwrap import dedent

from ..file_utils import create_path
from ..models import InternalTool
from ..rag.factory import get_rag_service
from ..rag.service import RagService
from ..text_extraction import is_extractable
from .context import InternalToolContext

logger = logging.getLogger(__name__)


class DocumentTools:
    """Search over documents the user indexed in this session.

    Two tools reach the model, not six. SAGE already spends around 13,396
    prompt tokens on a single simple question, and the metric it is judged on
    is tool-selection accuracy, measured at 149/180 in its own benchmark.
    Every added tool is both prompt budget and one more option to choose
    wrongly among.

    Activating, deactivating and listing documents are user management
    actions, not things a model needs to reason about mid-task, and SAGE
    already has a Files sidebar for them. They belong on REST endpoints
    backing that UI, and are deliberately absent here.

    Both tools are conditional, following the pattern PlayBookTools
    established for LoadPlayBook: SearchDocuments only appears once something
    is indexed, and IndexDocument only when there is an uploaded file worth
    indexing. A session that never uploads anything pays nothing -- the group
    returns no tools and `registry.py` drops it from the prompt and the UI
    entirely.
    """

    GROUP_NAME = "Documents"

    def __init__(self, ctx: InternalToolContext, service: RagService | None = None):
        self.ctx = ctx
        self.service = service
        self._indexed: list[dict] = []

    def tools(self) -> list[InternalTool]:
        if self.service is None:
            # RAG is not configured, or its vector store is unreachable. SAGE
            # must keep working without it, so the group simply disappears
            # rather than offering tools that would fail when called.
            return []

        tools: list[InternalTool] = []

        indexed = self._indexed_documents()
        if indexed:
            tools.append(
                InternalTool(
                    name="SearchDocuments",
                    description=self._search_description(indexed),
                    params={"query": "string"},
                    result="string",
                    function=self.tool_search_documents,
                )
            )

        if self._indexable_files(indexed):
            tools.append(
                InternalTool(
                    name="IndexDocument",
                    description=self._index_description(indexed),
                    params={"file_id": "string"},
                    result="string",
                    function=self.tool_index_document,
                )
            )

        return tools

    # -- descriptions --------------------------------------------------------

    def _search_description(self, indexed: list[dict]) -> str:
        """Name the searchable documents in the tool description.

        Without them the model has to guess whether a question is answerable
        from an uploaded document, and guesses both ways: it searches when
        nothing relevant is indexed, and answers from memory when the answer
        was sitting in a manual. Listing the documents makes the choice a
        lookup instead. PlayBookTools builds LoadPlayBook's description the
        same way.
        """
        listing = "\n".join(
            f"- {document['filename']} ({document['chunks']} sections)"
            for document in indexed
        )
        return dedent(f"""
            Search the user's indexed documents for passages relevant to a question.

            Call this when the user asks something that the documents below could answer.
            Do not call it for questions unrelated to them.

            Searchable documents:
            {listing}

            Returns numbered passages with their source. When you use one, cite it
            as [n] so the user can see where the answer came from.
        """).strip()

    def _index_description(self, indexed: list[dict]) -> str:
        listing = "\n".join(
            f"- file_id: {file.file_id}\n  Name: {file.file_name}"
            for file in self._indexable_files(indexed)
        )
        return dedent(f"""
            Make a large uploaded document searchable. Indexing runs once per document
            and takes a few seconds; afterwards its contents can be searched.

            Only worth doing for documents too large to read directly. Small files are
            refused, with an explanation.

            Files that can be indexed:
            {listing}
        """).strip()

    # -- state ---------------------------------------------------------------

    def _session_id(self) -> str:
        return self.ctx.session.session_id

    def _indexed_documents(self) -> list[dict]:
        return self._indexed

    async def refresh(self) -> None:
        """Reload which documents are indexed.

        `tools()` is synchronous and called on every turn while the vector
        store is async, so the listing is refreshed here and read from a field
        there. RagService caches it per session, so this is a dictionary
        lookup after the first call.
        """
        if self.service is None:
            # Acquired here rather than injected, so registry.py can keep
            # constructing every group the same way.
            self.service = await get_rag_service()
            if self.service is None:
                self._indexed = []
                return
        try:
            self._indexed = await self.service.list_documents(self._session_id())
        except Exception as error:
            # An unreachable vector store must not break the turn. The group
            # goes quiet instead, and SAGE answers without documents.
            logger.warning("Could not list indexed documents: %s", error)
            self._indexed = []

    def _indexable_files(self, indexed: list[dict]) -> list:
        """Uploaded files that could be indexed but are not yet.

        Already-indexed files are excluded so the model is not invited to
        re-index them, and unsupported types are excluded so it is not offered
        a call that can only fail.
        """
        already = {document["file_id"] for document in indexed}
        return [
            file
            for file in self.ctx.session.uploaded_files.values()
            if file.file_id not in already and is_extractable(file.file_name)
        ]

    # -- tools ---------------------------------------------------------------

    async def tool_search_documents(self, query: str) -> str:
        if not query or not query.strip():
            return "No query was given, so nothing was searched."

        try:
            result = await self.service.search(self._session_id(), query)
        except Exception as error:
            logger.warning("Document search failed: %s", error)
            return f"The document search failed: {error}"

        if not result:
            names = ", ".join(d["filename"] for d in self._indexed_documents()) or "none"
            return (
                f"No passage in the indexed documents ({names}) matches that query. "
                "Answer from your own knowledge, or say that the documents do not cover it."
            )

        return result.context

    async def tool_index_document(self, file_id: str) -> str:
        file = self.ctx.session.uploaded_files.get(file_id)
        if file is None:
            available = ", ".join(
                f.file_id for f in self.ctx.session.uploaded_files.values()
            ) or "none"
            return f"No uploaded file with file_id '{file_id}'. Available: {available}."

        path = create_path(self._session_id(), file_id)
        try:
            data = path.read_bytes()
        except OSError as error:
            logger.warning("Could not read %s: %s", path, error)
            return f"Could not read '{file.file_name}' from disk: {error}"

        result = await self.service.index_document(
            self._session_id(), file_id, file.file_name, data
        )
        await self.refresh()

        if not result.indexed:
            return result.reason
        return (
            f"'{result.filename}' is now searchable ({result.chunks} sections). "
            "Use SearchDocuments to answer questions about it."
        )
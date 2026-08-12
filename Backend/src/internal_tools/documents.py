from __future__ import annotations

import logging
from textwrap import dedent

from ..file_utils import create_path
from ..models import DocumentSourcesMessage, InternalTool
from ..rag.factory import get_rag_service
from ..rag.service import RagService
from ..text_extraction import is_extractable
from .context import InternalToolContext

logger = logging.getLogger(__name__)


class DocumentTools:
    """Search over documents the user uploaded in this session.

    One tool, not two. An earlier version exposed IndexDocument and
    SearchDocuments separately, and the split cost more than it saved.
    Observed repeatedly in use: unless the user explicitly said "index the
    document", the model reached for ChatHistory/SearchChats instead --
    indexing looked like a maintenance step rather than part of answering a
    question, and SearchDocuments was not offered at all until something had
    been indexed.

    That failure is expensive in a way the two-tool split is not. SearchChats
    puts the full transcript of every chat in the session into an LLM call
    with no bound (see chats.py), so one wrong choice costs roughly 4k tokens
    on a small history and 40k on a large one -- hundreds of times the ~62
    tokens saved by keeping the two descriptions apart. The user also has to
    send a second message, doubling the turns.

    So indexing became an implementation detail. Asking a question about an
    uploaded document is one action; whether that document happens to be
    indexed yet is not something the model should have to reason about.

    The cost of merging is that the first search on a new document is slower,
    because it indexes first. That is better than the two turns it replaces.

    Management actions -- activate, deactivate, list, remove -- stay off the
    model's tool list entirely. They are user actions with a Files sidebar
    already, and SAGE spends around 13,396 prompt tokens on a single simple
    question before any of this.
    """

    GROUP_NAME = "Documents"

    def __init__(self, ctx: InternalToolContext, service: RagService | None = None):
        self.ctx = ctx
        self.service = service
        self._indexed: list[dict] = []

    def tools(self) -> list[InternalTool]:
        if self.service is None:
            # RAG is not configured, or the vector store is unreachable. SAGE
            # must keep working without it, so the group disappears rather
            # than offering a tool that would fail when called.
            return []

        searchable = self._searchable_files()
        if not searchable:
            return []

        return [
            InternalTool(
                name="SearchDocuments",
                description=self._description(searchable),
                params={"query": "string"},
                result="string",
                function=self.tool_search_documents,
            )
        ]

    # -- description ---------------------------------------------------------

    def _description(self, searchable: list[str]) -> str:
        """Name the documents the tool can answer from.

        Listing them is what stops the model guessing. Without the names it
        guesses in both directions: searching when nothing relevant is
        uploaded, and answering from memory when the answer was in a manual
        sitting right there. PlayBookTools builds LoadPlayBook's description
        the same way.
        """
        listing = "\n".join(f"- {name}" for name in searchable)
        return dedent(f"""
            Search the documents the user uploaded in this conversation, and return the
            passages that are relevant to a question.

            Use this whenever the user asks something the documents below might answer,
            including questions about error codes, part numbers, procedures, settings or
            any other detail they contain. Prefer it over searching past conversations:
            these documents are the source, a chat transcript is not.

            Documents available:
            {listing}

            The first search on a document prepares it and may take a few seconds.
            Subsequent searches are fast.
        """).strip()

    # -- state ---------------------------------------------------------------

    def _session_id(self) -> str:
        return self.ctx.session.session_id

    def _searchable_files(self) -> list[str]:
        """Names of uploaded files this tool could answer from.

        Includes files that are not indexed yet: the tool indexes on demand,
        so from the model's side there is no difference.
        """
        return sorted(
            file.file_name
            for file in self.ctx.session.uploaded_files.values()
            if is_extractable(file.file_name)
        )

    async def refresh(self) -> None:
        """Reload which documents are already indexed.

        `tools()` is synchronous and runs on every turn while the vector store
        is async, so the listing is refreshed here and read from a field
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

    # -- the tool ------------------------------------------------------------

    async def tool_search_documents(self, query: str) -> str:
        if not query or not query.strip():
            return "No query was given, so nothing was searched."

        notes = await self._index_pending()

        try:
            result = await self.service.search(self._session_id(), query)
        except Exception as error:
            logger.warning("Document search failed: %s", error)
            return f"The document search failed: {error}"

        if not result:
            names = ", ".join(self._searchable_files()) or "none"
            return (
                f"{notes}No passage in the uploaded documents ({names}) matches that "
                "query. Say that the documents do not cover it, rather than answering "
                "from general knowledge as though they did."
            )

        await self._publish_sources(result.sources)

        # The instruction to cite lives here, in the tool's result, rather than
        # in its description. The description is read when the model decides
        # which tool to call; by the time it writes the answer, what is in
        # context is this text.
        #
        # It is not sufficient on its own: observed with the instruction in
        # the description alone, correct answers came back with no citations,
        # and moving it here still leaves the final wording to the output
        # generator's own prompt, which says nothing about citing. That is why
        # the sources also go to the frontend as structured data -- so the user
        # sees them whether or not the model complies.
        return (
            f"{notes}Passages found in the user's documents, most relevant first. "
            "Answer only from these passages, and mark each claim with the number of "
            "the passage it came from, like [1]. Do not add facts they do not contain.\n\n"
            f"{result.context}"
        )

    async def _publish_sources(self, sources: list[dict]) -> None:
        """Send the retrieved passages to the frontend as structured data.

        Separate from the tool result on purpose. The result is prose for the
        model; this is data for the UI, which can render it as clickable
        sources without parsing the prose back apart.

        Failure here must not fail the search: the answer is still correct
        without the source panel. scheduling.py pushes websocket messages the
        same way, so this follows an established pattern rather than a new one.
        """
        chat_id = self.ctx.chat_id
        if not chat_id or not sources:
            return
        try:
            await self.ctx.session.websocket_send(
                DocumentSourcesMessage(sources=sources, chat_id=chat_id)
            )
        except Exception as error:
            logger.warning("Could not publish document sources: %s", error)

    async def _index_pending(self) -> str:
        """Index any uploaded document that is not indexed yet.

        Returns a short note for the model when something was indexed, so a
        slow first search is explained rather than silent.
        """
        already = {document["file_id"] for document in self._indexed}
        pending = [
            file
            for file in self.ctx.session.uploaded_files.values()
            if file.file_id not in already and is_extractable(file.file_name)
        ]
        if not pending:
            return ""

        prepared, skipped = [], []
        for file in pending:
            path = create_path(self._session_id(), file.file_id)
            try:
                data = path.read_bytes()
            except OSError as error:
                logger.warning("Could not read %s: %s", path, error)
                skipped.append(f"{file.file_name} (could not be read)")
                continue

            result = await self.service.index_document(
                self._session_id(), file.file_id, file.file_name, data
            )
            if result.indexed:
                prepared.append(file.file_name)
            else:
                # Refused for a reason worth passing on: too small to be worth
                # indexing, no extractable text, an embedding failure. The
                # model should know why a document is missing from the results.
                skipped.append(f"{file.file_name}: {result.reason}")

        await self.refresh()

        note = ""
        if prepared:
            note += f"Prepared for search: {', '.join(prepared)}.\n"
        if skipped:
            note += f"Not searchable -- {'; '.join(skipped)}\n"
        return note + "\n" if note else ""
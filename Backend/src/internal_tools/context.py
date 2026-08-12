from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..code_execution import CodeExecutor
from ..models import Chat, QueryResponse, SessionData

if TYPE_CHECKING:
    from ..abstract_method import AbstractMethod


@dataclass(slots=True)
class InternalToolContext:
    session: SessionData
    agent_method: type["AbstractMethod"]
    code_executor: CodeExecutor

    # Which chat the current turn belongs to, when there is one. Needed by
    # tools that push a websocket message of their own, since every message in
    # that protocol is addressed to a chat. Optional because InternalTools is
    # also constructed outside a turn -- by the /internal-tools route and when
    # resuming scheduled tasks -- and those callers have no chat to name.
    chat_id: str | None = None

    async def query(self, query: str) -> QueryResponse:
        """Call AgentMethod.query without streaming, chat history, or internal tools."""
        self.session.is_notifs_aborted = False
        response = QueryResponse(query=query)
        method_impl = self.agent_method(self.session, Chat(chat_id=''), response, streaming=False)
        return await method_impl.query()
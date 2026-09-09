import logging
from typing import List
import asyncio

import httpx
from litellm.experimental_mcp_client.client import MCPClient
from mcp.types import CallToolRequestParams
from pydantic import BaseModel

from .models import (ToolApprovalState, SessionData, QueryResponse,
                     ToolCall, ToolType, ContainerLoginNotification, ContainerLoginResponse,
                     ToolResultMessage, ConfirmActionNotification, ConfirmActionResponse,
                     ParamSource)
from .internal_tools import InternalTools
from .xai.rationale import extract_rationale
from .opaca_client import actions_blacklist


# list of string-fragments; if any action or agent name contains one of those, SAGE will ask for confirmation before calling the tool
actions_needing_confirmation: List[str] = []

# Every failure path in invoke_tool writes a result starting with one of these.
# A tool that legitimately returns text about an error ("No errors were found")
# does not start with them, which is why this matches a prefix rather than
# searching for the word anywhere in the result.
_FAILURE_PREFIXES = (
    "Failed to invoke tool.",
    "Failed to invoke MCP tool.",
    "Failed to invoke Internal tool.",
    "Failed to invoke OPACA tool.",
)


def _classify_result(result) -> tuple[bool, str | None]:
    """Decide whether a result represents a failure, and extract the error text."""
    if isinstance(result, str) and result.startswith(_FAILURE_PREFIXES):
        _, _, detail = result.partition("\n")
        return False, (detail.strip() or result)
    return True, None

logger = logging.getLogger(__name__)


class ToolCaller:

    def __init__(self, session: SessionData, internal_tools: InternalTools = None, streaming: bool = False, chat_id: str = "none", response: QueryResponse = None) -> None:
        self.session = session
        self.internal_tools = internal_tools
        self.streaming = streaming
        self.chat_id = chat_id
        # Held so the approval dialog can say where a pending call's arguments
        # came from. Optional: a ToolCaller without one still works, it just
        # asks for confirmation with less to go on.
        self.response = response


    def determine_tool_type(self, tool_name: str) -> ToolType:
        """Determine tool type and name based on presence of server label and matching MCP server/tools"""
        provider = tool_name.split('--')[0]
        if self.internal_tools and self.internal_tools.is_internal_tool(provider):
            return ToolType.INTERNAL
        elif provider in self.session.mcp_servers:
            return ToolType.MCP
        else:
            return ToolType.OPACA


    async def invoke_tool(self, tool: ToolCall, login_attempt_retry: bool = False, skip_approval: bool = False) -> ToolCall:
        """
        Invoke tool (OPACA or MCP) matching the given ToolCall. 
        If OPACA invoke fails due to required login, attempt Login (via websocket callback) and try again.
        In any case returns a ToolCall, where "result" can be an error message.
        """
        async def create_result(result, success: bool | None = None, error: str | None = None):
            """Build the finished ToolCall.

            Every path out of invoke_tool goes through here, so recording the
            outcome here is what makes it impossible to forget on one branch.
            When the caller does not say, the outcome is inferred from the
            prefix the failure paths below all write; that inference is the
            fallback, not the mechanism.
            """
            if success is None:
                success, error = _classify_result(result)
            await self.send_to_websocket(ToolResultMessage(id=tool.id, result=result, chat_id=self.chat_id))
            return ToolCall(id=tool.id, type=tool.type, name=tool.name, args=tool.args,
                            result=result, success=success, error=error,
                            rationale=tool.rationale, considered=tool.considered)

        # The rationale rides in with the arguments and comes out here, before
        # anything reads them: the approval prompt, the provenance matcher and
        # the agent container all see the call as the model meant to make it,
        # not as the schema we added a field to.
        tool.args, why, considered = extract_rationale(tool.args)
        tool.rationale = tool.rationale or why
        tool.considered = tool.considered or considered

        # If login_attempt_retry=True, the user has already been asked and allowed tool execution
        if not (login_attempt_retry or skip_approval):
            approval_state = self.resolve_tool_approval(tool)

            if approval_state == ToolApprovalState.DENY:
                # Should not happen due to filtering in get_tools, but double-checking approval status just in case it changes in the future
                return await create_result("Failed to invoke tool.\nExecution denied by user settings, do not attempt again.")
                
            if approval_state == ToolApprovalState.ASK and not await self.check_confirmation(tool):
                return await create_result("Failed to invoke tool.\nExecution declined by user, do not attempt again.")

        # tools are always formatted the same; provider can be MCP server or OPACA agent
        provider, tool_name = tool.name.split('--', maxsplit=1)

        # MCP Tool Execution Flow
        if tool.type == ToolType.MCP:
            server = self.session.mcp_servers.get(provider)
            try:
                client = MCPClient(server_url=server.server_url)
                res = await client.call_tool(CallToolRequestParams(name=tool_name, arguments=tool.args))
                if res.isError:
                    t_result = f"Failed to invoke MCP tool.\nError: {res.content}"
                else:
                    t_result = res.content
            except Exception as e:
                t_result = f"Failed to invoke MCP tool.\nCause: {e}"

        # Internal Tool Execution Flow
        elif tool.type == ToolType.INTERNAL:
            try:
                t_result = await self.internal_tools.call_internal_tool(tool_name, tool.args)
            except Exception as e:
                t_result = f"Failed to invoke Internal tool.\nCause: {e}"

        # OPACA Tool Execution Flow
        else:
            try:
                t_result = await self.session.opaca_client.safe_invoke(tool_name, provider, tool.args)
            except httpx.HTTPStatusError as e:
                res = e.response.json()
                t_result = f"Failed to invoke OPACA tool.\nStatus code: {e.response.status_code}\nResponse: {e.response.text}\nResponse JSON: {res}"
                cause = res.get("cause", {}).get("message", "")
                status = res.get("cause", {}).get("statusCode", -1)
                if self.session.has_websocket() and (status in [401, 403] or ("401" in cause or "403" in cause or "credentials" in cause)):
                    return await self.handle_container_login(provider, tool_name, tool, login_attempt_retry)
            except Exception as e:
                t_result = f"Failed to invoke OPACA tool.\nCause: {e}"

        return await create_result(t_result)


    def resolve_tool_approval(self, tool: ToolCall) -> ToolApprovalState:
        if tool.type == ToolType.MCP:
            server_label, tool_name = tool.name.split('--', maxsplit=1)
            user_approval = self.session.get_mcp_tool(server_label, tool_name).approval
        else:
            user_approval = self.session.get_opaca_tool_approval(tool.name)

        if any(x.lower() in tool.name.lower() for x in actions_blacklist):
            # First the admin blacklist is applied
            return ToolApprovalState.DENY
        if user_approval == ToolApprovalState.DENY:
            # Then the user's approval setting for the tool
            return ToolApprovalState.DENY
        if any(x.lower() in tool.name.lower() for x in actions_needing_confirmation):
            # Then the admin confirmation list
            return ToolApprovalState.ASK
        # Then either the users ask or allow
        return user_approval


    async def check_confirmation(self, tool: ToolCall) -> bool:
        """Use websocket to ask user for confirmation before executing the action.
        Returns whether the action may be executed or not.

        The notification carries where each argument came from. That is the
        whole reason to compute provenance before a call rather than after: the
        user is deciding, and an argument traceable to nothing is the one thing
        that should change their mind.
        """
        if not self.session.has_websocket(): return False
        notification = ConfirmActionNotification(tool=tool.name, params=tool.args,
                                                 rationale=tool.rationale)
        self._add_context(notification, tool)
        # ask user for confirmation, sharing lock-mechanism with container-login
        async with self.session.opaca_client.login_lock:
            await self.session.websocket_send(notification)
            return ConfirmActionResponse(**await self.session.websocket_receive()).allowed

    def _add_context(self, notification: ConfirmActionNotification, tool: ToolCall) -> None:
        """Fill in provenance and what has happened so far. Never fails the call.

        A confirmation prompt that errors out would block a tool the user was
        willing to allow, so a broken explanation degrades to no explanation.
        """
        if self.response is None:
            return
        try:
            from .xai.provenance import edges_for_call
            from .xai.trace import build_trace

            trace = build_trace(self.response)
            notification.prior_calls = len([c for c in trace.calls if c.id != tool.id])
            notification.prior_failures = len(trace.failed_calls)
            notification.sources = [
                ParamSource(
                    param=edge.to_param_path,
                    value=edge.value,
                    source=edge.source,
                    kind=edge.kind,
                    origin=(f"{edge.from_call_name} ({edge.from_path})"
                            if edge.from_call_name else None),
                )
                for edge in edges_for_call(tool.id, tool.args, trace)
            ]
        except Exception as e:
            logger.warning(f"Could not build approval context: {e}")


    async def handle_container_login(self, agent_name: str, action_name: str, tool: ToolCall, login_attempt_retry: bool = False):
        """Handles failed tool invocation due to missing credentials."""

        # If a "missing credentials" error is encountered, initiate container login
        container_id, container_name = await self.session.opaca_client.get_most_likely_container_id(agent_name, action_name)

        # fix out-of-sync logged-in state, otherwise deadlock in retry within login-lock
        if container_id in self.session.opaca_client.container_tokens:
            del self.session.opaca_client.container_tokens[container_id]

        # This lock prevents more than one login-request message being sent to the UI at once. If multiple
        # invokes to actions of not-logged-in containers arrive, the second will wait here until the first
        # has been processed, and then immediately retry if it the same container, otherwise ask the user
        async with self.session.opaca_client.login_lock:
            # might already be logged in on lock-release if two actions of same container were called in parallel
            if container_id in self.session.opaca_client.container_tokens:
                return await self.invoke_tool(tool, True)
            while True:
                # Get credentials from user
                await self.session.websocket_send(ContainerLoginNotification(
                    container_name=container_name,
                    tool_name=tool.name,
                    retry=login_attempt_retry
                ))
                response = ContainerLoginResponse(**await self.session.websocket_receive())
                if not (response.username and response.password):
                    tool.result = "Failed to invoke tool.\nNo credentials provided."
                    tool.success, tool.error = _classify_result(tool.result)
                    return tool
                
                # Attempt to login at container via OPACA (error if immediate login-check fails)
                try:
                    await self.session.opaca_client.container_login(container_id, response.username, response.password)
                    break
                except Exception as e:
                    login_attempt_retry = True

        # login succeeded (or not checked by container) -> try to invoke the tool again
        res = await self.invoke_tool(tool, True)

        # Schedule a deferred logout based on the user-provided timeout
        asyncio.create_task(self.session.opaca_client.deferred_container_logout(container_id, response.timeout))

        return res

    async def send_to_websocket(self, message: BaseModel):
        if self.session.has_websocket() and self.streaming:
            await self.session.websocket_send(message)
# Tools

SAGE integrates three distinct tool types to give LLM agents access to data, files, and external systems. Regardless of their origin, all tools are normalized into standard OpenAI tool definitions so the model can interact with them uniformly.

Every tool passes through a centralized permission layer prior to execution.

## Approval Types

Each tool can be configured with one of three approval types:
- "allow": The tool executes automatically in the background without interrupting the chat flow.
- "ask": Pauses execution and prompts the user via the UI to approve or deny the action before it runs.
- "deny": Completely blocks the tool. SAGE filters these out during listing so the LLM is unaware of their existence.

## Internal Tools

- Built-in utilities implemented directly within the SAGE backend.
- Used for chat searches, task scheduling, and file handling.
- Default approval type is "allow".

## OPACA Tools

- Generated dynamically from OPACA agent tools on the fly when the agents are prompted.
- Default approval type is "allow".

## MCP Tools

- When an MCP server is added to SAGE, the backend scans all server tools, converts them into OpenAPI compatible tools and caches them in a session.
- Default approval type for the tools can be set when adding the MCP server.

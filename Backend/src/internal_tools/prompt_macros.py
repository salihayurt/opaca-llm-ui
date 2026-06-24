from __future__ import annotations

import uuid
from textwrap import dedent

from ..models import InternalTool, PromptMacro
from .context import InternalToolContext


class PromptMacroTools:
    GROUP_NAME = "Prompt Macros"

    def __init__(self, ctx: InternalToolContext):
        self.ctx = ctx

    def tools(self) -> list[InternalTool]:
        tools = [
            InternalTool(
                name="CreatePromptMacro",
                description=(
                    "Create and enable a reusable prompt macro for future user requests. "
                    "Use this when the user explicitly asks to define, remember, or save reusable behavior. "
                    "Describe when the macro should be loaded and the detailed instructions to follow. "
                    "Do not create a macro merely to fulfill a one-time request."
                ),
                params={
                    "name": "string",
                    "when_to_use": "string",
                    "what_to_do": "string",
                },
                result="object",
                function=self.tool_create_prompt_macro,
            ),
        ]

        if self._enabled_macros():
            tools.insert(
                0,
                InternalTool(
                    name="LoadPromptMacro",
                    description=self._loader_description(),
                    params={"macro_id": "string"},
                    result="string",
                    function=self.tool_load_prompt_macro,
                ),
            )

        return tools

    def _enabled_macros(self) -> list[PromptMacro]:
        return self.ctx.session.enabled_prompt_macros()

    def _loader_description(self) -> str:
        macro_lines = []
        for macro in self._enabled_macros():
            macro_lines.append(
                f"- macro_id: {macro.id}\n"
                f"  Name: {macro.name}\n"
                f"  When to use: {macro.when_to_use}"
            )
        macros_text = "\n".join(macro_lines)

        return dedent(f"""
            Load detailed instructions for a user-defined prompt macro.

            Call this tool before answering when the user's request clearly matches one of the available macros.
            Do not call this tool if none of the macros are relevant.

            Available prompt macros:
            {macros_text}

            Pass the exact macro_id of the matching macro.
        """).strip()

    async def tool_create_prompt_macro(self, name: str, when_to_use: str, what_to_do: str) -> dict:
        values = {
            "name": name.strip(),
            "when_to_use": when_to_use.strip(),
            "what_to_do": what_to_do.strip(),
        }
        if not all(values.values()):
            raise ValueError("Prompt macro name, usage description, and instructions must not be empty.")

        prompt_macro = PromptMacro(
            id=str(uuid.uuid4()),
            **values,
        )
        self.ctx.session.set_prompt_macro(prompt_macro)
        return {
            "created": True,
            "prompt_macro": prompt_macro.model_dump(),
        }

    async def tool_load_prompt_macro(self, macro_id: str) -> str:
        macro = next((m for m in self._enabled_macros() if m.id == macro_id), None)
        if macro is None:
            available = ", ".join(m.id for m in self._enabled_macros()) or "none"
            return f"No enabled prompt macro with macro_id '{macro_id}' was found. Available macro IDs: {available}."

        return dedent(f"""
            Prompt macro loaded: {macro.name}

            Follow these user-defined instructions for the current request:
            {macro.what_to_do}
        """).strip()

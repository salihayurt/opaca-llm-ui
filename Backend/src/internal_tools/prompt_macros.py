from __future__ import annotations

from textwrap import dedent

from ..models import InternalTool, PromptMacro
from .context import InternalToolContext


class PromptMacroTools:
    GROUP_NAME = "Prompt Macros"

    def __init__(self, ctx: InternalToolContext):
        self.ctx = ctx

    def tools(self) -> list[InternalTool]:
        if not self._enabled_macros():
            return []

        return [
            InternalTool(
                name="LoadPromptMacro",
                description=self._loader_description(),
                params={"macro_id": "string"},
                result="string",
                function=self.tool_load_prompt_macro,
            ),
        ]

    def _enabled_macros(self) -> list[PromptMacro]:
        return self.ctx.session.enabled_prompt_macros()

    def _loader_description(self) -> str:
        macro_lines = []
        for macro in self._enabled_macros():
            macro_lines.append(
                f"- macro_id: {macro.id}\n"
                f"  Name: {macro.name}\n"
                f"  When to use: {macro.description}"
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

    async def tool_load_prompt_macro(self, macro_id: str) -> str:
        macro = next((m for m in self._enabled_macros() if m.id == macro_id), None)
        if macro is None:
            available = ", ".join(m.id for m in self._enabled_macros()) or "none"
            return f"No enabled prompt macro with macro_id '{macro_id}' was found. Available macro IDs: {available}."

        return dedent(f"""
            Prompt macro loaded: {macro.name}

            Follow these user-defined instructions for the current request:
            {macro.instructions}
        """).strip()

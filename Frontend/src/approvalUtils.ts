// Shared by SidebarAgents and SidebarMcp. Later, when all the tools will be combined, it can be moved to the future SidebarTools.

import { type RestrictedActions, ToolApprovalState } from "./models";

export function matchesRestrictedTool(toolName: string, restrictions: RestrictedActions) {
    const normalizedToolName = toolName.toLowerCase();

    return (restrictions?.forbidden ?? []).some(fragment => normalizedToolName.includes(fragment.toLowerCase())) ||
        (restrictions?.need_confirmation ?? []).some(fragment => normalizedToolName.includes(fragment.toLowerCase()));
}

export function isForbiddenTool(toolName: string, restrictions: RestrictedActions) {
    const normalizedToolName = toolName.toLowerCase();

    return (restrictions?.forbidden ?? []).some(fragment => normalizedToolName.includes(fragment.toLowerCase()));
}

export function isConfirmationTool(toolName: string, restrictions: RestrictedActions) {
    const normalizedToolName = toolName.toLowerCase();

    return (restrictions?.need_confirmation ?? []).some(fragment => normalizedToolName.includes(fragment.toLowerCase()));
}

export function getEffectiveApproval(toolName: string, approval: ToolApprovalState | undefined, restrictions: RestrictedActions): ToolApprovalState {
    const normalizedToolName = toolName.toLowerCase();
    const normalizedApproval = approval || ToolApprovalState.ALLOW;

    if (restrictions.forbidden.some(fragment => normalizedToolName.includes(fragment.toLowerCase()))) {
        return ToolApprovalState.DENY;
    }

    if (normalizedApproval === ToolApprovalState.DENY) {
        return ToolApprovalState.DENY;
    }

    if (restrictions.need_confirmation.some(fragment => normalizedToolName.includes(fragment.toLowerCase()))) {
        return ToolApprovalState.ASK;
    }

    if (normalizedApproval === ToolApprovalState.ASK) {
        return ToolApprovalState.ASK;
    }

    return ToolApprovalState.ALLOW;
}
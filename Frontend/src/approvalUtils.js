// Shared by SidebarAgents and SidebarMcp. Later, when all the tools will be combined, it can be moved to the future SidebarTools.

export function matchesRestrictedTool(toolName, restrictions) {
    const normalizedToolName = toolName.toLowerCase();

    return (restrictions?.forbidden ?? []).some(fragment => normalizedToolName.includes(fragment.toLowerCase())) ||
        (restrictions?.need_confirmation ?? []).some(fragment => normalizedToolName.includes(fragment.toLowerCase()));
}

export function isForbiddenTool(toolName, restrictions) {
    const normalizedToolName = toolName.toLowerCase();

    return (restrictions?.forbidden ?? []).some(fragment => normalizedToolName.includes(fragment.toLowerCase()));
}

export function isConfirmationTool(toolName, restrictions) {
    const normalizedToolName = toolName.toLowerCase();

    return (restrictions?.need_confirmation ?? []).some(fragment => normalizedToolName.includes(fragment.toLowerCase()));
}

export function getEffectiveApproval(toolName, approval, restrictions) {
    const normalizedToolName = toolName.toLowerCase();
    const normalizedApproval = approval || 'allow';

    if ((restrictions?.forbidden ?? []).some(fragment => normalizedToolName.includes(fragment.toLowerCase()))) {
        return 'deny';
    }

    if (normalizedApproval === 'deny') {
        return 'deny';
    }

    if ((restrictions?.need_confirmation ?? []).some(fragment => normalizedToolName.includes(fragment.toLowerCase()))) {
        return 'ask';
    }

    if (normalizedApproval === 'ask') {
        return 'ask';
    }

    return 'allow';
}
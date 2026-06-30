import { isBrowserDarkTheme } from "./ColorThemes.js";

type DesktopNotificationClickHandler = (
    event: Event,
    notification: Notification,
) => void | Promise<void>;

interface DesktopNotificationOptions {
    body?: string | null;
    onClick?: DesktopNotificationClickHandler | null;
    focusWindowOnClick?: boolean;
}

const missedResponseChatIds = new Set<string>();
const notificationFavicons: Record<"light" | "dark", string> = {
    light: "/sage-logo-small-light-notification.png",
    dark: "/sage-logo-small-dark-notification.png",
};

export async function showDesktopNotification(
    title: string,
    {body = null, onClick = null, focusWindowOnClick = true}: DesktopNotificationOptions = {},
): Promise<Notification | null> {
    if (typeof window === "undefined" || !("Notification" in window)) return null;

    const permission = Notification.permission === "default"
        ? await Notification.requestPermission()
        : Notification.permission;
    if (permission !== "granted") return null;

    const notification = new Notification(title, body ? {body: body} : undefined);
    if (focusWindowOnClick || onClick) {
        notification.onclick = async event => {
            if (focusWindowOnClick) {
                window.focus();
            }
            if (onClick) {
                await onClick(event, notification);
            }
        };
    }
    return notification;
}

export function markMissedChatResponse(chatId: string | null | undefined): void {
    if (!chatId) return;
    missedResponseChatIds.add(chatId);
    updateTabNotificationFavicon();
}

export function clearMissedChatResponse(chatId: string | null = null): void {
    if (chatId) {
        missedResponseChatIds.delete(chatId);
    } else {
        missedResponseChatIds.clear();
    }
    updateTabNotificationFavicon();
}

function updateTabNotificationFavicon(): void {
    if (typeof document === "undefined") return;

    const hasMissedResponse = missedResponseChatIds.size > 0;
    const icons = document.querySelectorAll<HTMLLinkElement>('link[rel~="icon"]');

    if (!hasMissedResponse) {
        icons.forEach(icon => {
            if (icon.dataset.originalHref) {
                icon.href = icon.dataset.originalHref;
                delete icon.dataset.originalHref;
            }
        });
        return;
    }

    icons.forEach(icon => {
        if (!icon.dataset.originalHref) {
            icon.dataset.originalHref = icon.href;
        }
        icon.href = getNotificationFavicon(icon);
    });
}

function getNotificationFavicon(icon: HTMLLinkElement): string {
    const media = icon?.media?.toLowerCase() ?? "";
    if (media.includes("prefers-color-scheme: dark")) return notificationFavicons.dark;
    if (media.includes("prefers-color-scheme: light")) return notificationFavicons.light;
    return isBrowserDarkTheme() ? notificationFavicons.dark : notificationFavicons.light;
}

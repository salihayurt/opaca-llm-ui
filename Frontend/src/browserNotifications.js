const missedResponseChatIds = new Set();
const redFaviconCache = {};

export async function showDesktopNotification(title, {body = null, onClick = null, focusWindowOnClick = true} = {}) {
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

export function markMissedChatResponse(chatId) {
    if (!chatId || missedResponseChatIds.has(chatId)) return;
    missedResponseChatIds.add(chatId);
    updateTabNotificationFavicon();
}

export function clearMissedChatResponse(chatId = null) {
    if (chatId) {
        missedResponseChatIds.delete(chatId);
    } else {
        missedResponseChatIds.clear();
    }
    updateTabNotificationFavicon();
}

async function updateTabNotificationFavicon() {
    if (typeof document === "undefined") return;

    const hasMissedResponse = missedResponseChatIds.size > 0;
    const icons = document.querySelectorAll('link[rel~="icon"]');

    if (!hasMissedResponse) {
        icons.forEach(icon => {
            if (icon.dataset.originalHref) {
                icon.href = icon.dataset.originalHref;
                delete icon.dataset.originalHref;
            }
        });
        return;
    }

    await Promise.all(Array.from(icons).map(async icon => {
        if (!icon.dataset.originalHref) {
            icon.dataset.originalHref = icon.href;
        }
        const redHref = await createRedFavicon(icon.dataset.originalHref);
        if (missedResponseChatIds.size > 0) {
            icon.href = redHref;
        }
    }));
}

function createRedFavicon(sourceHref) {
    if (redFaviconCache[sourceHref]) {
        return Promise.resolve(redFaviconCache[sourceHref]);
    }

    return new Promise(resolve => {
        const img = new Image();
        img.onload = () => {
            const size = Math.max(img.naturalWidth, img.naturalHeight, 32);
            const canvas = document.createElement("canvas");
            canvas.width = size;
            canvas.height = size;

            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0, size, size);
            ctx.globalCompositeOperation = "source-atop";
            ctx.fillStyle = "#dc3545";
            ctx.fillRect(0, 0, size, size);

            const redHref = canvas.toDataURL("image/png");
            redFaviconCache[sourceHref] = redHref;
            resolve(redHref);
        };
        img.onerror = () => resolve(sourceHref);
        img.src = sourceHref;
    });
}

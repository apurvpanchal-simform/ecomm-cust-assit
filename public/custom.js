// Inject custom CSS to hide avatars for tool calls
const style = document.createElement('style');
style.textContent = `
    .step-message .message-avatar,
    div[data-test="step"] .message-avatar,
    .step-avatar {
        display: none !important;
    }
`;
document.head.appendChild(style);

// Setup customer typing indicator
let lastCustomerTypingTime = 0;

function getConversationId() {
    // The welcome message injects a hidden zero-width link: [​](http://conversation-id/{conv_id})
    // This is always the most reliable source of the conversation ID
    const links = document.querySelectorAll('a[href^="http://conversation-id/"]');
    if (links.length > 0) {
        const parts = links[0].href.split('/');
        return parts[parts.length - 1];
    }

    // Fallback: try the URL path for /thread/<uuid> style Chainlit URLs
    const pathParts = window.location.pathname.split('/');
    const threadIdx = pathParts.indexOf('thread');
    if (threadIdx !== -1 && pathParts[threadIdx + 1]) {
        return pathParts[threadIdx + 1];
    }

    return null;
}

function getBackendUrl() {
    // If already on port 8000 (direct API access) use same origin
    // Otherwise swap the chainlit port (8501) for the backend port (8000)
    const origin = window.location.origin;
    if (origin.includes(':8501')) {
        return origin.replace(':8501', ':8000');
    }
    // In production both are on the same origin
    return origin;
}

document.addEventListener('input', (e) => {
    // Chainlit uses a textarea for the chat input
    if (e.target.tagName.toLowerCase() === 'textarea') {
        const now = Date.now();
        if (now - lastCustomerTypingTime < 2000) return; // Throttle to 2s
        lastCustomerTypingTime = now;
        
        const convId = getConversationId();

        // Only send if we have a conversation id, meaning it's an established chat
        if (convId && convId !== '') {
            console.debug("Customer typing detected for conversation:", convId);
            fetch(`${getBackendUrl()}/chat/conversations/${convId}/typing/customer`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            }).catch(err => {
                console.debug("Failed to send typing event", err);
            });
        } else {
            console.debug("Customer typing: no conversation ID found yet, skipping.");
        }
    }
});

// Fallback for human agent avatars: if an avatar image fails to load (e.g. because the agent
// has a custom name and no file exists), fallback to support_agent.png.
document.addEventListener('error', function(e) {
    if (e.target.tagName && e.target.tagName.toLowerCase() === 'img') {
        if (e.target.src.includes('/public/avatars/')) {
            // Don't loop infinitely if support_agent.png itself is missing
            if (!e.target.src.endsWith('support_agent.png')) {
                e.target.src = '/public/avatars/support_agent.png';
            }
        }
    }
}, true); // Use capture phase because error events on images do not bubble

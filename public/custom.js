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

document.addEventListener('input', (e) => {
    // Chainlit uses a textarea with id "chat-input" or similar class for input
    if (e.target.tagName.toLowerCase() === 'textarea') {
        const now = Date.now();
        if (now - lastCustomerTypingTime < 2000) return; // Throttle to 2s
        lastCustomerTypingTime = now;
        
        let convId = null;
        const pathParts = window.location.pathname.split('/');
        if (pathParts.includes('thread')) {
            convId = pathParts[pathParts.length - 1];
        } else {
            const links = document.querySelectorAll('a[href^="http://conversation-id/"]');
            if (links.length > 0) {
                const parts = links[0].href.split('/');
                convId = parts[parts.length - 1];
            }
        }

        // Only send if we have a conversation id, meaning it's an established chat
        if (convId && convId !== '') {
            console.debug("Customer typing detected for conversation:", convId);
            // Note: API_BASE is the backend server, typically running on 8000
            // Since custom.js runs on 8501 (Chainlit), we need to send to port 8000
            const backendUrl = window.location.origin.replace('8501', '8000');
            fetch(`${backendUrl}/chat/conversations/${convId}/typing/customer`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            }).catch(err => {
                // Ignore errors (could be CORS if not configured, but should be fine locally)
                console.debug("Failed to send typing event", err);
            });
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

// Hide the Rename thread option in the chainlit UI
const hideRenameObserver = new MutationObserver((mutations) => {
    document.querySelectorAll('li[role="menuitem"], button').forEach(el => {
        if (el.textContent.trim() === 'Rename' || el.getAttribute('aria-label') === 'Rename') {
            el.style.display = 'none';
        }
    });
});

hideRenameObserver.observe(document.body, {
    childList: true,
    subtree: true
});

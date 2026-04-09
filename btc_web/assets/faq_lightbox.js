/* FAQ image lightbox — click to enlarge, click again to close */
document.addEventListener('click', function(e) {
    var img = e.target;
    // Click on accordion image → enlarge
    if (img.tagName === 'IMG' && img.closest('.accordion-body') && !img.classList.contains('faq-enlarged')) {
        e.preventDefault();
        var overlay = document.createElement('div');
        overlay.className = 'faq-overlay';
        document.body.appendChild(overlay);
        img.classList.add('faq-enlarged');
        img._origStyle = img.getAttribute('style') || '';
        img.style.maxWidth = '95vw';
        img.style.width = 'auto';
        overlay.onclick = function() { closeEnlarged(img, overlay); };
        return;
    }
    // Click on enlarged image → close
    if (img.tagName === 'IMG' && img.classList.contains('faq-enlarged')) {
        var overlay = document.querySelector('.faq-overlay');
        closeEnlarged(img, overlay);
        return;
    }
});

function closeEnlarged(img, overlay) {
    img.classList.remove('faq-enlarged');
    if (img._origStyle !== undefined) {
        img.setAttribute('style', img._origStyle);
    }
    if (overlay) overlay.remove();
}

// ESC key closes
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        var img = document.querySelector('.faq-enlarged');
        var overlay = document.querySelector('.faq-overlay');
        if (img) closeEnlarged(img, overlay);
    }
});

/* Diagnostic: track bub-qs checkbox mutations to distinguish reconciliation
   clobber vs payload drop. Per reviewer 2026-04-24. */
(function () {
  window.__qsMut = window.__qsMut || [];
  function start() {
    if (!document.body) { setTimeout(start, 50); return; }
    var obs = new MutationObserver(function (muts) {
      for (var i = 0; i < muts.length; i++) {
        var t = muts[i].target;
        if (t && (t.name === 'bub-qs' || (t.id && t.id.indexOf('bub-qs') >= 0))) {
          window.__qsMut.push({
            t: Math.round(performance.now()),
            type: muts[i].type,
            attr: muts[i].attributeName,
            checked: t.checked,
            value: t.value,
          });
        }
      }
    });
    obs.observe(document.body, {subtree: true, attributes: true, childList: true,
                                attributeFilter: ['checked', 'value']});
  }
  start();
})();

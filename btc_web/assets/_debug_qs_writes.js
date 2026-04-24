/* Temporary diagnostic — capture any response containing bub-qs value */
(function () {
  window.__qsLog = window.__qsLog || [];
  window.__dashResps = window.__dashResps || [];
  if (window.__fetchWrapped) return;
  window.__fetchWrapped = true;
  var orig = window.fetch;
  window.fetch = async function() {
    var url = typeof arguments[0] === 'string' ? arguments[0] : arguments[0]?.url || '';
    var resp = await orig.apply(this, arguments);
    if (url.indexOf('_dash-update-component') !== -1) {
      try {
        var body = await resp.clone().text();
        if (body.indexOf('bub-qs.value') !== -1) {
          window.__dashResps.push({
            t: Math.round(performance.now()),
            body: body.slice(0, 1500),
          });
        }
      } catch(e) {}
    }
    return resp;
  };
})();

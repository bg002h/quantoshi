/* Temporary diagnostic — capture any write to bub-qs */
(function () {
  window.__qsLog = window.__qsLog || [];
  function install() {
    if (!window.dash_clientside || !window.dash_clientside.set_props) {
      setTimeout(install, 50);
      return;
    }
    if (window.__qsWrapped) return;
    window.__qsWrapped = true;
    var orig = window.dash_clientside.set_props;
    window.dash_clientside.set_props = function(id, props) {
      if (id === 'bub-qs' && 'value' in props) {
        window.__qsLog.push({
          t: Math.round(performance.now()),
          v: JSON.stringify(props.value),
          stack: (new Error()).stack.split('\n').slice(1, 5).join(' | ')
        });
      }
      return orig.apply(this, arguments);
    };
  }
  install();
})();

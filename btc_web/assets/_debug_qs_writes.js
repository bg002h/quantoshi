/* Diagnostic: poll bub-qs checked state every 20ms to detect any transient write. */
(function () {
  window.__qsPoll = window.__qsPoll || [];
  function start() {
    var read = function() {
      var inputs = document.querySelectorAll('input[name="bub-qs"]');
      if (!inputs.length) return null;
      return Array.from(inputs).filter(function(i){return i.checked;}).map(function(i){return i.value;});
    };
    var last = JSON.stringify([]);
    var id = setInterval(function() {
      var v = read();
      if (v === null) return;
      var s = JSON.stringify(v);
      if (s !== last) {
        window.__qsPoll.push({
          t: Math.round(performance.now()),
          v: v
        });
        last = s;
      }
    }, 20);
    // Auto-stop after 25s to avoid memory churn
    setTimeout(function(){ clearInterval(id); }, 25000);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();

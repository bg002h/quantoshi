/* ── Ambient toasts & footer block height ─────────────────────────────────────
   1. Bitcoin birthday toast (Jan 3)
   2. Night owl toast (10 PM – 5 AM, once per session)
   3. Footer: live block height + halving countdown
   All self-contained, no Dash callbacks needed. */
(function() {
    "use strict";

    var WIZ_KEY = "wizard-flags";
    function _wizFlags() {
        try { return JSON.parse(localStorage.getItem(WIZ_KEY)) || {}; }
        catch(e) { return {}; }
    }
    function _setWizFlag(flag) {
        var f = _wizFlags();
        f[flag] = true;
        localStorage.setItem(WIZ_KEY, JSON.stringify(f));
    }

    /* ── Ambient toast helper ──────────────────────────────────────────────── */
    function showAmbientToast(text) {
        var el = document.createElement("div");
        el.className = "ambient-toast";
        el.textContent = text;
        document.body.appendChild(el);
        setTimeout(function() {
            if (el.parentNode) el.parentNode.removeChild(el);
        }, 6500);
    }

    /* ── 1. Bitcoin birthday: January 3 ────────────────────────────────────── */
    var now = new Date();
    if (now.getMonth() === 0 && now.getDate() === 3) {
        var genesisYear = 2009;
        var age = now.getFullYear() - genesisYear;
        var todayKey = "bday_" + now.getFullYear();
        if (!_wizFlags()[todayKey]) {
            _setWizFlag(todayKey);
            setTimeout(function() {
                showAmbientToast("Happy Birthday, Bitcoin! Year " + age +
                                 " \u2014 Genesis block: Jan 3, 2009");
            }, 4000);
        }
    }

    /* ── 2. Night owl: 10 PM \u2013 5 AM local time (once per month) ─────────── */
    var hour = now.getHours();
    if (hour >= 22 || hour < 5) {
        var owlMonth = now.getFullYear() + "-" + (now.getMonth() + 1);
        var wf = _wizFlags();
        if (wf.night_owl_month !== owlMonth) {
            var f = _wizFlags();
            f.night_owl_month = owlMonth;
            localStorage.setItem(WIZ_KEY, JSON.stringify(f));
            setTimeout(function() {
                showAmbientToast("Burning the midnight oil? Stay humble, stack sats.");
            }, 6000);
        }
    }

    /* ── 3. Footer: block height + halving countdown ───────────────────────── */
    var HALVING_INTERVAL = 210000;
    /* Known reference: halving #4 at block 840,000 (April 2024) */
    var NEXT_HALVING_BLOCK = 1050000; /* halving #5 */

    var currentBlockHeight = null;

    function updateFooter(height) {
        if (!height || isNaN(height)) return;
        currentBlockHeight = height;
        var footerHeight = document.getElementById("footer-block-height");
        var footerHalving = document.getElementById("footer-halving-countdown");
        if (footerHeight) {
            footerHeight.textContent = "Block #" + height.toLocaleString();
        }
        if (footerHalving) {
            var remaining = NEXT_HALVING_BLOCK - height;
            if (remaining <= 0) {
                /* Already past this halving — show next one */
                var nextH = NEXT_HALVING_BLOCK;
                while (nextH <= height) nextH += HALVING_INTERVAL;
                remaining = nextH - height;
            }
            /* Estimate time: ~10 min per block */
            var days = Math.round(remaining * 10 / 1440);
            footerHalving.textContent = "Next halving in ~" +
                remaining.toLocaleString() + " blocks (~" + days.toLocaleString() + " days)";
        }
    }

    /* Listen for block height updates from blockdrop.js WebSocket */
    /* blockdrop.js stores lastHeight but doesn't expose it, so we
       piggyback on the same APIs it uses. */

    /* Endpoint: own mempool onion for Tor, clearnet otherwise */
    var _isOnion = location.hostname.endsWith(".onion");
    var _mempoolHTTP = _isOnion
        ? "http://jxnpv6ef3yo2kqpeu6u3nmv343k7vpyn7katlfdoc3n7hgvz7l5woqid.onion"
        : "https://mempool.space";
    var _fallbackHTTP = _isOnion
        ? "http://explorerzydxu5ecjrkwceayqybizmpjjznk5izmitf2modhcusuqlid.onion"
        : "https://blockstream.info";

    /* Fetch current height on load (delay to let Dash render footer) */
    function fetchHeight() {
        fetch(_mempoolHTTP + "/api/blocks/tip/height")
            .then(function(r) { return r.text(); })
            .then(function(t) {
                var h = parseInt(t, 10);
                if (h && !isNaN(h)) updateFooter(h);
            })
            .catch(function() {
                return fetch(_fallbackHTTP + "/api/blocks/tip/height")
                    .then(function(r) { return r.text(); })
                    .then(function(t) {
                        var h = parseInt(t, 10);
                        if (h && !isNaN(h)) updateFooter(h);
                    })
                    .catch(function() {});
            });
    }
    setTimeout(fetchHeight, 2000);

    /* Poll every 2 minutes for updated height */
    setInterval(fetchHeight, 120000);

})();

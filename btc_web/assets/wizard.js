/* ── Magic Internet Money Wizard ─────────────────────────────────────────────
   A pixelated wizard flies across the screen trailing Bitcoin sparkles.
   Triggers: price milestone, first share, all tabs, 1/250 random, 1/10 after
   block crack, Konami code, stack milestones (0.01/0.1/0.5/1.0 BTC).
   Dev mode: always after block crack. */
(function() {
    "use strict";

    var FLIGHT_MS     = 4000;
    var SPARKLE_COUNT = 12;
    var wizardActive  = false;

    /* Wizard one-time flags live in their own localStorage key so Dash's
       dcc.Store (which owns "journey-store") can never overwrite them. */
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

    /* ── Wizard animation ────────────────────────────────────────────────── */
    function summonWizard(reason) {
        if (wizardActive) return;
        wizardActive = true;

        var goRight = Math.random() > 0.5;
        var startX  = goRight ? -120 : window.innerWidth + 20;
        var endX    = goRight ? window.innerWidth + 120 : -120;
        var yPos    = 15 + Math.random() * 40; /* 15-55% from top */

        var overlay = document.createElement("div");
        overlay.className = "wizard-overlay";

        /* Wizard character */
        var wiz = document.createElement("div");
        wiz.className = "wizard-sprite" + (goRight ? "" : " wizard-flip");
        wiz.innerHTML =
            '<svg class="wizard-svg" viewBox="0 0 140 200" xmlns="http://www.w3.org/2000/svg">' +
            /* Hat */
            '<polygon points="70,2 45,60 95,60" fill="#3a2d7c"/>' +
            '<polygon points="70,2 45,60 95,60" fill="url(#hatShine)" opacity="0.3"/>' +
            '<rect x="35" y="55" width="70" height="10" rx="3" fill="#2d2266"/>' +
            /* Stars on hat */
            '<text x="62" y="30" font-size="10" fill="#ffd700" text-anchor="middle">&#x2605;</text>' +
            '<text x="72" y="45" font-size="7" fill="#ffd700" text-anchor="middle">&#x2605;</text>' +
            /* Face */
            '<circle cx="70" cy="75" r="16" fill="#f5d6a8"/>' +
            /* Eyes */
            '<circle cx="64" cy="72" r="2" fill="#222"/>' +
            '<circle cx="76" cy="72" r="2" fill="#222"/>' +
            /* Smile */
            '<path d="M65,79 Q70,84 75,79" stroke="#222" stroke-width="1.5" fill="none"/>' +
            /* Beard */
            '<path d="M58,80 Q55,100 60,110 Q65,115 70,118 Q75,115 80,110 Q85,100 82,80" fill="#ddd" opacity="0.9"/>' +
            '<path d="M62,85 Q65,95 68,105" stroke="#ccc" stroke-width="0.5" fill="none"/>' +
            '<path d="M72,85 Q75,95 78,105" stroke="#ccc" stroke-width="0.5" fill="none"/>' +
            /* Robe */
            '<path d="M50,88 L35,175 Q70,185 105,175 L90,88 Q70,95 50,88Z" fill="#4a3a9c"/>' +
            '<path d="M50,88 L35,175 Q70,185 105,175 L90,88 Q70,95 50,88Z" fill="url(#robeShine)" opacity="0.2"/>' +
            /* Robe center line */
            '<line x1="70" y1="95" x2="70" y2="178" stroke="#3a2d7c" stroke-width="1.5"/>' +
            /* Belt */
            '<ellipse cx="70" cy="115" rx="22" ry="4" fill="#ffd700" opacity="0.7"/>' +
            /* Arms */
            '<path d="M50,95 L25,130 L30,132 L52,102" fill="#4a3a9c"/>' +
            '<path d="M90,95 L110,120 L112,118 L92,92" fill="#4a3a9c"/>' +
            /* Hands */
            '<circle cx="25" cy="131" r="5" fill="#f5d6a8"/>' +
            '<circle cx="111" cy="119" r="5" fill="#f5d6a8"/>' +
            /* Staff */
            '<line x1="24" y1="126" x2="18" y2="50" stroke="#8B4513" stroke-width="3" stroke-linecap="round"/>' +
            /* Bitcoin orb on staff */
            '<circle cx="18" cy="45" r="10" fill="#f7931a"/>' +
            '<circle cx="18" cy="45" r="10" fill="url(#orbGlow)" opacity="0.4"/>' +
            '<text x="18" y="50" font-size="14" font-weight="bold" fill="#fff" ' +
            '  text-anchor="middle" font-family="monospace">&#x20bf;</text>' +
            /* Feet */
            '<ellipse cx="52" cy="178" rx="10" ry="4" fill="#2d2266"/>' +
            '<ellipse cx="88" cy="178" rx="10" ry="4" fill="#2d2266"/>' +
            /* "MAGIC INTERNET MONEY" text */
            '<text x="70" y="196" font-size="9" font-weight="bold" fill="#f7931a" ' +
            '  text-anchor="middle" font-family="monospace, sans-serif" ' +
            '  letter-spacing="0.5">MAGIC INTERNET MONEY</text>' +
            /* Defs for gradients */
            '<defs>' +
            '<linearGradient id="hatShine" x1="0" y1="0" x2="1" y2="1">' +
            '  <stop offset="0%" stop-color="#fff"/>' +
            '  <stop offset="100%" stop-color="transparent"/>' +
            '</linearGradient>' +
            '<linearGradient id="robeShine" x1="0" y1="0" x2="0.3" y2="1">' +
            '  <stop offset="0%" stop-color="#fff"/>' +
            '  <stop offset="100%" stop-color="transparent"/>' +
            '</linearGradient>' +
            '<radialGradient id="orbGlow">' +
            '  <stop offset="0%" stop-color="#fff"/>' +
            '  <stop offset="100%" stop-color="#f7931a"/>' +
            '</radialGradient>' +
            '</defs>' +
            '</svg>';
        wiz.style.left = startX + "px";
        wiz.style.top = yPos + "vh";
        overlay.appendChild(wiz);

        /* Reason toast */
        if (reason) {
            var toast = document.createElement("div");
            toast.className = "wizard-toast";
            toast.textContent = reason;
            toast.style.top = (yPos + 12) + "vh";
            overlay.appendChild(toast);
        }

        document.body.appendChild(overlay);

        /* Animate wizard flight */
        var startTime = null;
        function fly(ts) {
            if (!startTime) startTime = ts;
            var progress = (ts - startTime) / FLIGHT_MS;
            if (progress >= 1) {
                cleanup();
                return;
            }
            /* Sinusoidal bob */
            var x = startX + (endX - startX) * progress;
            var yOff = Math.sin(progress * Math.PI * 3) * 15;
            wiz.style.left = x + "px";
            wiz.style.top = "calc(" + yPos + "vh + " + yOff + "px)";

            /* Drop sparkles periodically */
            if (Math.random() < 0.15) {
                spawnSparkle(overlay, x + (goRight ? -20 : 20),
                             parseFloat(wiz.style.top) || (yPos / 100 * window.innerHeight));
            }

            requestAnimationFrame(fly);
        }
        requestAnimationFrame(fly);

        /* Auto-cleanup safety net */
        setTimeout(cleanup, FLIGHT_MS + 500);

        function cleanup() {
            if (overlay && overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
            wizardActive = false;
        }
    }

    function spawnSparkle(parent, x, y) {
        var sp = document.createElement("div");
        sp.className = "wizard-sparkle";
        var symbols = ["\u20bf", "\u2728", "\u2b50", "\u2734\ufe0f", "\u26a1"];
        sp.textContent = symbols[Math.floor(Math.random() * symbols.length)];
        sp.style.left = x + "px";
        /* Parse y: could be "calc(...)" string, use viewport center as fallback */
        var yNum = (typeof y === "number") ? y :
                   (window.innerHeight * 0.35);
        sp.style.top = yNum + "px";
        sp.style.setProperty("--drift", (Math.random() * 60 - 30) + "px");
        parent.appendChild(sp);
        setTimeout(function() {
            if (sp.parentNode) sp.parentNode.removeChild(sp);
        }, 1500);
    }

    /* Expose globally so blockdrop.js can call it */
    window._summonWizard = summonWizard;

    /* ── Dev: test streak-unlocked orange Q after wizard flies ────────────── */
    var _isDev = (location.hostname !== "quantoshi.xyz" &&
                  !location.hostname.endsWith(".onion"));
    if (_isDev) {
        var _origSummon = summonWizard;
        summonWizard = function(reason) {
            _origSummon(reason);
            setTimeout(function() {
                try {
                    var j = JSON.parse(localStorage.getItem("journey-store")) || {};
                    j.streak_unlocked = true;
                    localStorage.setItem("journey-store", JSON.stringify(j));
                    var els = document.querySelectorAll(".brand-q");
                    for (var i = 0; i < els.length; i++) {
                        els[i].classList.add("streak-unlocked");
                    }
                } catch(e) {}
            }, FLIGHT_MS + 300);
        };
        window._summonWizard = summonWizard;
    }

    /* ── Trigger 6: Random 1/250 on page load ────────────────────────────── */
    if (Math.random() < 1 / 250) {
        setTimeout(function() { summonWizard("Magic Internet Money!"); }, 3000);
    }

    /* ── Trigger 8: Konami code ──────────────────────────────────────────── */
    var konamiSeq = [38,38,40,40,37,39,37,39,66,65]; /* ↑↑↓↓←→←→BA */
    var konamiPos = 0;
    document.addEventListener("keydown", function(e) {
        if (e.keyCode === konamiSeq[konamiPos]) {
            konamiPos++;
            if (konamiPos >= konamiSeq.length) {
                konamiPos = 0;
                summonWizard("Konami Wizard Unlocked!");
            }
        } else {
            konamiPos = 0;
        }
    });

    /* ── Trigger 1: Price crosses round number ───────────────────────────── */
    var lastTickerPrice = null;
    var PRICE_MILESTONES = [50000, 100000, 150000, 200000, 250000, 300000,
                            400000, 500000, 750000, 1000000];

    function parseTickerPrice(el) {
        if (!el) return null;
        var m = (el.textContent || "").match(/\$([\d.]+)(K|M)?/);
        if (!m) return null;
        var p = parseFloat(m[1]);
        if (m[2] === "K") p *= 1000;
        else if (m[2] === "M") p *= 1000000;
        return p;
    }

    /* Watch for ticker text changes */
    var tickerObs = null;
    function watchTicker() {
        var el = document.getElementById("price-ticker");
        if (!el) { setTimeout(watchTicker, 2000); return; }
        lastTickerPrice = parseTickerPrice(el);
        tickerObs = new MutationObserver(function() {
            var newPrice = parseTickerPrice(el);
            if (newPrice && lastTickerPrice) {
                for (var i = 0; i < PRICE_MILESTONES.length; i++) {
                    var m = PRICE_MILESTONES[i];
                    if (lastTickerPrice < m && newPrice >= m) {
                        var label = m >= 1000000 ? "$" + (m/1e6) + "M" : "$" + (m/1e3) + "K";
                        summonWizard("BTC crossed " + label + "!");
                        break;
                    }
                }
            }
            lastTickerPrice = newPrice;
        });
        tickerObs.observe(el, {childList: true, characterData: true, subtree: true});
    }
    watchTicker();

    /* ── Trigger 2: First share link generated ───────────────────────────── */
    var shareWatched = false;
    function watchShare() {
        var el = document.getElementById("share-link-area");
        if (!el) { setTimeout(watchShare, 2000); return; }
        if (shareWatched) return;
        shareWatched = true;
        new MutationObserver(function() {
            if (!_wizFlags().first_share) {
                _setWizFlag("first_share");
                summonWizard("First Share Link Created!");
            }
        }).observe(el, {childList: true, characterData: true, subtree: true});
    }
    watchShare();

    /* ── Trigger 3: All 7 tabs explored (once only) ─────────────────────── */
    /* Note: streak.js also wraps setItem — chain them properly */
    var _prevSetItem = Storage.prototype.setItem;
    Storage.prototype.setItem = function(key, val) {
        _prevSetItem.call(this, key, val);
        if (key === "journey-store" && !_wizFlags().all_tabs) {
            try {
                var j = JSON.parse(val);
                if (j && j.tabs_seen && j.tabs_seen.length >= 7) {
                    _setWizFlag("all_tabs");
                    setTimeout(function() {
                        summonWizard("All 7 Tabs Explored!");
                    }, 500);
                }
            } catch(e) {}
        }
    };

    /* ── Trigger 9: Stack milestones (0.01, 0.1, 0.5, 1.0 BTC) ─────────── */
    var STACK_MILESTONES = [0.01, 0.1, 0.5, 1.0];
    var lastStackTotal = null;
    function checkStackMilestone() {
        try {
            var lots = JSON.parse(localStorage.getItem("lots-store"));
            if (!Array.isArray(lots) || lots.length === 0) return;
            var total = 0;
            for (var i = 0; i < lots.length; i++) {
                total += parseFloat(lots[i].amount || lots[i].amt || 0);
            }
            var wf = _wizFlags();
            var passed = wf.stack_milestones || [];
            for (var m = 0; m < STACK_MILESTONES.length; m++) {
                var ms = STACK_MILESTONES[m];
                if (total >= ms && passed.indexOf(ms) === -1) {
                    passed.push(ms);
                    wf.stack_milestones = passed;
                    localStorage.setItem(WIZ_KEY, JSON.stringify(wf));
                    summonWizard(ms >= 1 ? "1 Whole Coin! \u20bf" : "\u20bf " + ms + " BTC Stacked!");
                    break;
                }
            }
            lastStackTotal = total;
        } catch(e) {}
    }
    /* Check on lots-store changes via the setItem wrapper */
    var _prevSetItem2 = Storage.prototype.setItem;
    Storage.prototype.setItem = function(key, val) {
        _prevSetItem2.call(this, key, val);
        if (key === "lots-store") {
            setTimeout(checkStackMilestone, 300);
        }
    };

})();

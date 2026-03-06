/* ── Knighting ceremony: awarded when the Orange Q streak unlocks ─────────────
   Detects the moment streak_unlocked transitions to true in journey-store.
   Plays a full-screen ceremony: dim → staff descends → taps Q → golden burst →
   Q turns orange → "Rise, Anon. The Orange Q is yours." */
(function() {
    "use strict";

    var WIZ_KEY = "wizard-flags";
    var ceremonyPlayed = false;

    function _wizFlags() {
        try { return JSON.parse(localStorage.getItem(WIZ_KEY)) || {}; }
        catch(e) { return {}; }
    }

    /* Check if ceremony was already shown (skip on dev — replay every load) */
    var _isDev = (location.hostname !== "quantoshi.xyz" &&
                  !location.hostname.endsWith(".onion"));
    if (_wizFlags().knighted && !_isDev) ceremonyPlayed = true;

    var NOBLE_TITLES = [
        "Lord of the Blockchain",
        "Duke of Satoshi's Keep",
        "Baron von HODL",
        "Sovereign of the Mempool",
        "Knight of the Orange Coin",
        "Archduke of Cold Storage",
        "Warden of the Private Keys",
        "Sentinel of the Timechain",
        "Guardian of the Genesis Block",
        "Marshal of the Lightning Network",
        "Viscount of Unspent Outputs",
        "Earl of the Final Settlement",
        "Protector of Sound Money",
        "Chancellor of the Hash Rate",
        "Keeper of the Orange Flame",
    ];

    /* ── The ceremony ──────────────────────────────────────────────────────── */
    function playKnighting() {
        if (ceremonyPlayed) return;
        ceremonyPlayed = true;

        /* Mark as played */
        var f = _wizFlags();
        f.knighted = true;
        localStorage.setItem(WIZ_KEY, JSON.stringify(f));

        /* Find the Q element to position effects around it */
        var qEl = document.querySelector(".brand-q");
        if (!qEl) return;
        var qRect = qEl.getBoundingClientRect();
        var qCx = qRect.left + qRect.width / 2;
        var qCy = qRect.top + qRect.height / 2;

        /* ── Overlay ──────────────────────────────────────────────────────── */
        var overlay = document.createElement("div");
        overlay.className = "knight-overlay";
        document.body.appendChild(overlay);

        /* ── Staff (descends from above) ──────────────────────────────────── */
        var staff = document.createElement("div");
        staff.className = "knight-staff";
        staff.innerHTML =
            '<svg viewBox="0 0 40 300" xmlns="http://www.w3.org/2000/svg">' +
            '<line x1="20" y1="0" x2="20" y2="260" stroke="#8B4513" stroke-width="5" stroke-linecap="round"/>' +
            '<circle cx="20" cy="10" r="12" fill="#f7931a"/>' +
            '<circle cx="20" cy="10" r="12" fill="url(#kOrbGlow)" opacity="0.5"/>' +
            '<text x="20" y="15" font-size="14" font-weight="bold" fill="#fff" ' +
            '  text-anchor="middle" font-family="monospace">&#x20bf;</text>' +
            '<defs><radialGradient id="kOrbGlow">' +
            '<stop offset="0%" stop-color="#fff"/>' +
            '<stop offset="100%" stop-color="#f7931a"/>' +
            '</radialGradient></defs></svg>';
        staff.style.left = Math.max(4, qCx - 20) + "px";
        overlay.appendChild(staff);

        /* ── Phase 1 (0–1.5s): Staff descends ─────────────────────────────── */
        /* Staff starts above viewport, CSS animation brings it down */

        /* ── Phase 2 (1.5s): Staff touches Q → golden burst ──────────────── */
        setTimeout(function() {
            /* Flash the Q → turn orange */
            qEl.classList.add("knight-flash");
            var allQs = document.querySelectorAll(".brand-q");
            for (var qi = 0; qi < allQs.length; qi++) {
                allQs[qi].classList.add("streak-orange");
            }
            /* Add shimmer to brand name */
            var brands = document.querySelectorAll(".brand-uantoshi");
            for (var b = 0; b < brands.length; b++) {
                brands[b].classList.add("streak-shimmer");
            }
            /* Hat appears after color settles */
            setTimeout(function() {
                var qs = document.querySelectorAll(".brand-q");
                for (var h = 0; h < qs.length; h++) {
                    qs[h].classList.add("streak-hat");
                }
            }, 800);

            /* Golden burst particles */
            for (var i = 0; i < 24; i++) {
                spawnGoldParticle(overlay, qCx, qCy);
            }

            /* Radiating rings */
            for (var r = 0; r < 3; r++) {
                var ring = document.createElement("div");
                ring.className = "knight-ring";
                ring.style.left = qCx + "px";
                ring.style.top = qCy + "px";
                ring.style.animationDelay = (r * 0.3) + "s";
                overlay.appendChild(ring);
            }
        }, 1500);

        /* ── Phase 3 (2.5s): Text appears ─────────────────────────────────── */
        setTimeout(function() {
            var text = document.createElement("div");
            text.className = "knight-text";
            var title = NOBLE_TITLES[Math.floor(Math.random() * NOBLE_TITLES.length)];
            text.innerHTML = "Rise, <em>" + title + "</em>.<br>The Orange Q is yours.";
            overlay.appendChild(text);
        }, 2500);

        /* ── Phase 4 (3s): Staff retracts ─────────────────────────────────── */
        setTimeout(function() {
            staff.classList.add("knight-staff-retract");
        }, 3500);

        /* ── Phase 5 (5.5s): Gold particle shower ─────────────────────────── */
        setTimeout(function() {
            for (var s = 0; s < 30; s++) {
                setTimeout(function() {
                    var x = Math.random() * window.innerWidth;
                    spawnFallingGold(overlay, x);
                }, Math.random() * 1200);
            }
        }, 3000);

        /* ── Cleanup (7s) ─────────────────────────────────────────────────── */
        setTimeout(function() {
            overlay.classList.add("knight-fade-out");
            setTimeout(function() {
                if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
                qEl.classList.remove("knight-flash");
            }, 1000);
        }, 6500);
    }

    function spawnGoldParticle(parent, cx, cy) {
        var p = document.createElement("div");
        p.className = "knight-particle";
        var angle = Math.random() * 360;
        var dist = 60 + Math.random() * 140;
        var dx = Math.cos(angle * Math.PI / 180) * dist;
        var dy = Math.sin(angle * Math.PI / 180) * dist;
        p.style.left = cx + "px";
        p.style.top = cy + "px";
        p.style.setProperty("--dx", dx + "px");
        p.style.setProperty("--dy", dy + "px");
        var symbols = ["\u2605", "\u2726", "\u2727", "\u2736", "\u2605"];
        p.textContent = symbols[Math.floor(Math.random() * symbols.length)];
        parent.appendChild(p);
        setTimeout(function() {
            if (p.parentNode) p.parentNode.removeChild(p);
        }, 1800);
    }

    function spawnFallingGold(parent, x) {
        var p = document.createElement("div");
        p.className = "knight-confetti";
        p.textContent = "\u2605";
        p.style.left = x + "px";
        p.style.animationDuration = (2 + Math.random() * 2) + "s";
        parent.appendChild(p);
        setTimeout(function() {
            if (p.parentNode) p.parentNode.removeChild(p);
        }, 4500);
    }

    /* ── Hook into streak.js setItem chain ─────────────────────────────────── */
    /* We detect when streak_unlocked transitions from falsy to true */
    var _prevSetItem = Storage.prototype.setItem;
    Storage.prototype.setItem = function(key, val) {
        _prevSetItem.call(this, key, val);
        if (key === "journey-store" && !ceremonyPlayed) {
            try {
                var j = JSON.parse(val);
                if (j && j.streak_unlocked) {
                    /* On dev, only fire after wizard has flown */
                    if (_isDev && !window._wizardFlown) return;
                    /* Small delay so the Q class gets applied first */
                    setTimeout(playKnighting, 800);
                }
            } catch(e) {}
        }
    };

    /* Expose for dev testing */
    window._playKnighting = function() {
        ceremonyPlayed = false;
        playKnighting();
    };

    /* Dev: reset Q to white (no hat) on page load so we see the full transition.
       wizard.js sets streak_unlocked after the wizard flies,
       which triggers the setItem wrapper above → playKnighting(). */
    if (_isDev) {
        function resetQ() {
            var els = document.querySelectorAll(".brand-q");
            for (var i = 0; i < els.length; i++) {
                els[i].classList.remove("streak-unlocked", "streak-orange", "streak-hat");
            }
            var brands = document.querySelectorAll(".brand-uantoshi");
            for (var b = 0; b < brands.length; b++) {
                brands[b].classList.remove("streak-shimmer");
            }
        }
        resetQ();
        /* Dash renders late — reset again after a delay */
        setTimeout(resetQ, 1000);
        setTimeout(resetQ, 2500);
    }

})();

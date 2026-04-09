/* ── Knighting ceremony: awarded when the Orange Q streak unlocks ─────────────
   Detects the moment streak_unlocked transitions to true in journey-store.
   Plays a full-screen ceremony: dim → staff descends → taps Q → golden burst →
   Q turns orange → "Rise, Anon. The Orange Q is yours."
   Onion variant: triggered by easter egg button for Tor users. */
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

    var STAFF_SVG =
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

    /* Shared: apply burst effects (Q flash, orange, shimmer, hat, particles, rings) */
    function _applyBurst(overlay, qEl, qCx, qCy) {
        qEl.classList.add("knight-flash");
        var allQs = document.querySelectorAll(".brand-q");
        for (var qi = 0; qi < allQs.length; qi++) {
            allQs[qi].classList.add("streak-orange");
        }
        var brands = document.querySelectorAll(".brand-uantoshi");
        for (var b = 0; b < brands.length; b++) {
            brands[b].classList.add("streak-shimmer");
        }
        setTimeout(function() {
            var qs = document.querySelectorAll(".brand-q");
            for (var h = 0; h < qs.length; h++) {
                qs[h].classList.add("streak-hat");
            }
        }, 800);
        for (var i = 0; i < 24; i++) {
            spawnGoldParticle(overlay, qCx, qCy);
        }
        for (var r = 0; r < 3; r++) {
            var ring = document.createElement("div");
            ring.className = "knight-ring";
            ring.style.left = qCx + "px";
            ring.style.top = qCy + "px";
            ring.style.animationDelay = (r * 0.3) + "s";
            overlay.appendChild(ring);
        }
    }

    /* Shared: create overlay + staff, return {overlay, staff, qEl, qCx, qCy} */
    function _setupCeremony() {
        var qEl = document.querySelector(".brand-q");
        if (!qEl) return null;
        var qRect = qEl.getBoundingClientRect();
        var qCx = qRect.left + qRect.width / 2;
        var qCy = qRect.top + qRect.height / 2;
        var overlay = document.createElement("div");
        overlay.className = "knight-overlay";
        document.body.appendChild(overlay);
        var staff = document.createElement("div");
        staff.className = "knight-staff";
        staff.innerHTML = STAFF_SVG;
        staff.style.left = Math.max(4, qCx - 20) + "px";
        overlay.appendChild(staff);
        return {overlay: overlay, staff: staff, qEl: qEl, qCx: qCx, qCy: qCy};
    }

    /* Shared: pick a title (deterministic) and persist it + knighted flag + streak_unlocked */
    function _knightAndTitle() {
        var f = _wizFlags();

        /* Keep existing title (backward compat for already-knighted users) */
        if (f.noble_title) {
            f.knighted = true;
            localStorage.setItem(WIZ_KEY, JSON.stringify(f));
            /* Also set streak_unlocked so orange Q persists on future loads */
            try {
                var j2 = JSON.parse(localStorage.getItem("journey-store")) || {};
                j2.streak_unlocked = true;
                localStorage.setItem("journey-store", JSON.stringify(j2));
            } catch(e2) {}
            return f.noble_title;
        }

        /* Generate stable seed on first visit (cryptographic randomness) */
        if (!f.user_seed) {
            f.user_seed = crypto.getRandomValues(new Uint32Array(1))[0];
        }

        /* Deterministic selection from seed */
        var title = NOBLE_TITLES[f.user_seed % NOBLE_TITLES.length];
        f.knighted = true;
        f.noble_title = title;
        localStorage.setItem(WIZ_KEY, JSON.stringify(f));
        /* Also set streak_unlocked so orange Q persists on future loads */
        try {
            var j = JSON.parse(localStorage.getItem("journey-store")) || {};
            j.streak_unlocked = true;
            localStorage.setItem("journey-store", JSON.stringify(j));
        } catch(e) {}
        return title;
    }

    /* ── Standard ceremony (7-day streak) ────────────────────────────────── */
    function playKnighting() {
        if (ceremonyPlayed) return;
        ceremonyPlayed = true;

        var title = _knightAndTitle();
        var c = _setupCeremony();
        if (!c) return;

        /* Phase 1 (0–1.5s): Staff descends (CSS animation) */

        /* Phase 2 (1.5s): Staff touches Q → golden burst */
        setTimeout(function() { _applyBurst(c.overlay, c.qEl, c.qCx, c.qCy); }, 1500);

        /* Phase 3 (2.5s): Text appears */
        setTimeout(function() {
            var text = document.createElement("div");
            text.className = "knight-text";
            text.innerHTML = "Rise, <em>" + title + "</em>.<br>The Orange Q is yours.";
            c.overlay.appendChild(text);
        }, 2500);

        /* Phase 4 (3.5s): Staff retracts */
        setTimeout(function() { c.staff.classList.add("knight-staff-retract"); }, 3500);

        /* Phase 5 (3s): Gold particle shower */
        setTimeout(function() {
            for (var s = 0; s < 30; s++) {
                setTimeout(function() {
                    var x = Math.random() * window.innerWidth;
                    spawnFallingGold(c.overlay, x);
                }, Math.random() * 1200);
            }
        }, 3000);

        /* Cleanup (7s) */
        setTimeout(function() {
            c.overlay.classList.add("knight-fade-out");
            setTimeout(function() {
                if (c.overlay.parentNode) c.overlay.parentNode.removeChild(c.overlay);
                c.qEl.classList.remove("knight-flash");
            }, 1000);
        }, 6500);
    }

    /* ── Onion ceremony (dark web variant for Tor users) ─────────────────── */
    function playOnionKnighting() {
        if (ceremonyPlayed) return;
        ceremonyPlayed = true;

        var title = _knightAndTitle();
        var c = _setupCeremony();
        if (!c) return;

        /* Phase 1 (0–1.5s): Staff descends */

        /* Phase 2 (1.5s): Golden burst */
        setTimeout(function() { _applyBurst(c.overlay, c.qEl, c.qCx, c.qCy); }, 1500);

        /* Phase 3 (2.5s): Text 1 — dark web journey */
        var text1;
        setTimeout(function() {
            text1 = document.createElement("div");
            text1.className = "knight-text";
            text1.innerHTML = "Through the treacherous depths of the dark web,<br>" +
                              "you found your way to Quantoshi.";
            c.overlay.appendChild(text1);
        }, 2500);

        /* Phase 4 (3.5s): Staff retracts */
        setTimeout(function() { c.staff.classList.add("knight-staff-retract"); }, 3500);

        /* Phase 5 (5.5s): Text 1 fades → Text 2 — Rise */
        var text2;
        setTimeout(function() {
            if (text1) {
                text1.style.transition = "opacity 0.6s";
                text1.style.opacity = "0";
                setTimeout(function() {
                    if (text1.parentNode) text1.parentNode.removeChild(text1);
                }, 600);
            }
            text2 = document.createElement("div");
            text2.className = "knight-text";
            text2.innerHTML = "Rise, <em>" + title + "</em>.<br>The Orange Q is yours.";
            c.overlay.appendChild(text2);
        }, 5500);

        /* Phase 6 (6s): Gold shower */
        setTimeout(function() {
            for (var s = 0; s < 30; s++) {
                setTimeout(function() {
                    var x = Math.random() * window.innerWidth;
                    spawnFallingGold(c.overlay, x);
                }, Math.random() * 1200);
            }
        }, 6000);

        /* Phase 7 (8.5s): Text 2 fades → Text 3 — pretend not to know */
        setTimeout(function() {
            if (text2) {
                text2.style.transition = "opacity 0.6s";
                text2.style.opacity = "0";
                setTimeout(function() {
                    if (text2.parentNode) text2.parentNode.removeChild(text2);
                }, 600);
            }
            var text3 = document.createElement("div");
            text3.className = "knight-text";
            text3.innerHTML = "We shall pretend not to know each other next time&hellip;<br>" +
                              "but the Orange Q will always be yours.";
            c.overlay.appendChild(text3);
        }, 8500);

        /* Cleanup (12.5s) */
        setTimeout(function() {
            c.overlay.classList.add("knight-fade-out");
            setTimeout(function() {
                if (c.overlay.parentNode) c.overlay.parentNode.removeChild(c.overlay);
                c.qEl.classList.remove("knight-flash");
            }, 1000);
        }, 12000);
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

    /* Expose for dev testing + onion ceremony trigger */
    window._playKnighting = function() {
        ceremonyPlayed = false;
        playKnighting();
    };
    window._playOnionKnighting = function() {
        ceremonyPlayed = false;
        playOnionKnighting();
    };

    /* Replay ceremony (for FAQ link / easter egg) */
    window._replayKnighting = function() {
        var f = _wizFlags();
        if (!f.knighted) return;  /* only replay if previously knighted */
        ceremonyPlayed = false;
        if (location.hostname.endsWith(".onion")) {
            playOnionKnighting();
        } else {
            playKnighting();
        }
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

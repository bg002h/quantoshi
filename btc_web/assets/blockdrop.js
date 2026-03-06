/* ── Block Drop: new-block animation ─────────────────────────────────────────
   Connects to mempool.space WebSocket (blockstream.info polling fallback).
   On new block: screen shake → block falls → user taps to crack → tx scatter.
   Self-contained, no Dash callbacks needed. ~10-min cadence, negligible cost. */
(function() {
    "use strict";

    /* ── Config ──────────────────────────────────────────────────────────── */
    var TAPS_TO_BREAK = 4;
    var TX_COUNT      = 16;       /* scattered transaction elements       */
    var FRAG_COUNT    = 6;        /* block fragments on shatter           */
    var SHAKE_MS      = 400;
    var DROP_MS       = 1100;
    var SCATTER_MS    = 1800;
    var POLL_SEC      = 30;       /* blockstream fallback poll interval   */

    var lastHeight    = null;
    var animating     = false;
    var ws            = null;
    var pollTimer     = null;
    var wsRetryDelay  = 2000;

    /* ── Mempool.space WebSocket (primary) ───────────────────────────────── */
    function connectWS() {
        try {
            ws = new WebSocket("wss://mempool.space/api/v1/ws");
        } catch(e) { startPolling(); return; }

        ws.onopen = function() {
            wsRetryDelay = 2000;
            ws.send(JSON.stringify({action:"want", data:["blocks"]}));
            /* Stop polling if it was running as fallback */
            if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        };
        ws.onmessage = function(evt) {
            try {
                var d = JSON.parse(evt.data);
                if (d.block && d.block.height) {
                    var h = d.block.height;
                    if (lastHeight !== null && h > lastHeight) {
                        onNewBlock(h, d.block.tx_count || 0);
                    }
                    lastHeight = h;
                }
            } catch(e) {}
        };
        ws.onclose = function() {
            ws = null;
            setTimeout(function() {
                wsRetryDelay = Math.min(wsRetryDelay * 2, 60000);
                connectWS();
            }, wsRetryDelay);
            /* Start polling as interim fallback */
            if (!pollTimer) startPolling();
        };
        ws.onerror = function() { if (ws) ws.close(); };
    }

    /* ── Blockstream.info polling (fallback) ─────────────────────────────── */
    function startPolling() {
        if (pollTimer) return;
        pollTimer = setInterval(pollBlock, POLL_SEC * 1000);
        pollBlock();
    }
    function pollBlock() {
        fetch("https://blockstream.info/api/blocks/tip/height")
            .then(function(r) { return r.text(); })
            .then(function(t) {
                var h = parseInt(t, 10);
                if (!h || isNaN(h)) return;
                if (lastHeight !== null && h > lastHeight) {
                    onNewBlock(h, 0);
                }
                lastHeight = h;
            })
            .catch(function() {});
    }

    /* ── New block handler ───────────────────────────────────────────────── */
    function onNewBlock(height, txCount) {
        if (animating) return;
        animating = true;
        screenShake();
        setTimeout(function() { dropBlock(height, txCount); }, SHAKE_MS);
    }

    /* ── Screen shake ────────────────────────────────────────────────────── */
    function screenShake() {
        document.body.classList.add("block-shake");
        setTimeout(function() {
            document.body.classList.remove("block-shake");
        }, SHAKE_MS);
    }

    /* ── Drop block from sky ─────────────────────────────────────────────── */
    function dropBlock(height, txCount) {
        var overlay = document.createElement("div");
        overlay.className = "blockdrop-overlay";

        var block = document.createElement("div");
        block.className = "blockdrop-cube blockdrop-fall";
        block.innerHTML = '<div class="blockdrop-label">#' + height.toLocaleString() + '</div>' +
                          '<div class="blockdrop-sublabel">' + (txCount > 0 ? txCount + ' tx' : 'new block') + '</div>';

        /* Crack overlay layers */
        for (var c = 1; c <= TAPS_TO_BREAK; c++) {
            var crack = document.createElement("div");
            crack.className = "blockdrop-crack blockdrop-crack-" + c;
            block.appendChild(crack);
        }

        overlay.appendChild(block);
        document.body.appendChild(overlay);

        var taps = 0;

        /* Wait for drop to finish before enabling taps */
        setTimeout(function() {
            block.classList.remove("blockdrop-fall");
            block.classList.add("blockdrop-landed");

            /* Bounce hint */
            block.classList.add("blockdrop-bounce");
            setTimeout(function() { block.classList.remove("blockdrop-bounce"); }, 400);

            function onTap(e) {
                e.preventDefault();
                taps++;
                /* Micro-shake on each tap */
                block.classList.remove("blockdrop-hit");
                void block.offsetWidth; /* reflow */
                block.classList.add("blockdrop-hit");

                /* Show progressive cracks */
                block.setAttribute("data-cracks", taps);

                if (taps >= TAPS_TO_BREAK) {
                    overlay.removeEventListener("click", onTap);
                    overlay.removeEventListener("touchstart", onTap);
                    shatterBlock(overlay, block, txCount, height);
                }
            }
            overlay.addEventListener("click", onTap);
            overlay.addEventListener("touchstart", onTap, {passive: false});

            /* Auto-dismiss after 12s if user doesn't tap */
            setTimeout(function() {
                if (taps < TAPS_TO_BREAK && overlay.parentNode) {
                    overlay.style.opacity = "0";
                    overlay.style.transition = "opacity 0.5s";
                    setTimeout(function() { cleanup(overlay); }, 500);
                }
            }, 12000);
        }, DROP_MS);
    }

    /* ── Shatter block + scatter transactions ────────────────────────────── */
    function shatterBlock(overlay, block, txCount, height) {
        /* Get position BEFORE hiding */
        var rect = block.getBoundingClientRect();
        var cx = rect.left + rect.width / 2;
        var cy = rect.top + rect.height / 2;
        block.style.display = "none";

        /* Fragments */
        for (var i = 0; i < FRAG_COUNT; i++) {
            var frag = document.createElement("div");
            frag.className = "blockdrop-fragment";
            var angle = (i / FRAG_COUNT) * 360 + (Math.random() * 40 - 20);
            var dist = 80 + Math.random() * 120;
            var dx = Math.cos(angle * Math.PI / 180) * dist;
            var dy = Math.sin(angle * Math.PI / 180) * dist;
            var rot = Math.random() * 360 - 180;
            frag.style.left = cx + "px";
            frag.style.top = cy + "px";
            frag.style.setProperty("--dx", dx + "px");
            frag.style.setProperty("--dy", dy + "px");
            frag.style.setProperty("--rot", rot + "deg");
            overlay.appendChild(frag);
        }

        /* Transaction elements */
        var n = Math.min(TX_COUNT, Math.max(8, txCount || TX_COUNT));
        for (var t = 0; t < n; t++) {
            var tx = document.createElement("div");
            tx.className = "blockdrop-tx";
            tx.textContent = "tx";
            var a = Math.random() * 360;
            var d = 120 + Math.random() * 200;
            var tdx = Math.cos(a * Math.PI / 180) * d;
            var tdy = Math.sin(a * Math.PI / 180) * d - 60; /* bias upward */
            tx.style.left = cx + "px";
            tx.style.top = cy + "px";
            tx.style.setProperty("--dx", tdx + "px");
            tx.style.setProperty("--dy", tdy + "px");
            tx.style.animationDelay = (Math.random() * 0.3) + "s";
            overlay.appendChild(tx);
        }

        /* Cleanup after scatter finishes */
        setTimeout(function() { cleanup(overlay); }, SCATTER_MS + 500);

        /* Trigger 7: wizard after block crack (1/10 chance, always in dev) */
        if (typeof window._summonWizard === "function") {
            var wizChance = isDev ? 1 : 0.1;
            if (Math.random() < wizChance) {
                setTimeout(function() {
                    window._summonWizard("A wizard emerges from block #" + height + "!");
                }, SCATTER_MS + 200);
            }
        }
    }

    function cleanup(overlay) {
        if (overlay && overlay.parentNode) {
            overlay.parentNode.removeChild(overlay);
        }
        animating = false;
    }

    /* ── Dev/test mode: fake block 10s after load on non-production ─────── */
    var isDev = (location.hostname !== "quantoshi.xyz" &&
                 !location.hostname.endsWith(".onion"));
    if (isDev) {
        setTimeout(function() {
            var fakeHeight = (lastHeight || 890000) + 1;
            lastHeight = fakeHeight - 1; /* ensure delta triggers */
            onNewBlock(fakeHeight, 2847);
        }, 10000);
    }

    /* ── Bootstrap ───────────────────────────────────────────────────────── */
    /* Seed lastHeight so first WS message doesn't trigger animation */
    fetch("https://mempool.space/api/blocks/tip/height")
        .then(function(r) { return r.text(); })
        .then(function(t) {
            var h = parseInt(t, 10);
            if (h && !isNaN(h)) lastHeight = h;
        })
        .catch(function() {
            return fetch("https://blockstream.info/api/blocks/tip/height")
                .then(function(r) { return r.text(); })
                .then(function(t) {
                    var h = parseInt(t, 10);
                    if (h && !isNaN(h)) lastHeight = h;
                })
                .catch(function() {});
        })
        .finally(function() {
            connectWS();
        });
})();

"""Registration surface for spl.

Every check here guards a step that fails SILENTLY -- the model renders fine
whether or not the step was done. Do not replace these with a visual check.
"""
import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]


def _src(rel):
    return (REPO / rel).read_text()


class TestPipelineIntegration:
    def test_in_build_bm_model_instances(self):
        """Step 19. update_prices.py runs build_bm_model.py after every price
        append; a model missing here is absent from every pkl rebuild."""
        assert 'instances["spl"]' in _src("tools/build_bm_model.py")

    def test_in_refit_all_ppl(self):
        """Step 20. Otherwise spl is never refit and its params freeze.

        Substring-matching the source is not enough. refit_all_ppl.py
        os.path.exists()-checks the whole path string, so an entry written
        ("SatPL", "tools/fit_spl.py --update") satisfies `"fit_spl.py" in
        src` while resolving to nothing -- and the script is silently
        skipped on every live run, which is the exact freeze this test
        exists to catch. That failure mode is LIVE a few lines away in the
        same list, so parse the entry and assert its path resolves.
        """
        tree = ast.parse(_src("tools/refit_all_ppl.py"))
        scripts = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SCRIPTS"
                for t in node.targets
            ):
                scripts = ast.literal_eval(node.value)
        assert scripts is not None, "SCRIPTS not found in refit_all_ppl.py"

        paths = [p for _, p in scripts if "fit_spl.py" in p]
        assert len(paths) == 1, f"expected exactly one spl entry, got {paths}"
        # Deliberately scoped to the spl entry only. Generalising this to
        # every SCRIPTS entry would go red on a pre-existing unrelated
        # entry whose path does not resolve; that one is out of scope.
        assert (REPO / paths[0]).exists(), (
            f"SCRIPTS entry {paths[0]!r} does not resolve on disk; "
            "refit_all_ppl.py would print 'NOT FOUND -- skipping' and "
            "never refit spl"
        )

    def test_in_fit_shrinking_sigma(self):
        """Step 21: both the branch and the models list."""
        s = _src("tools/fit_shrinking_sigma.py")
        assert 'model_name == "spl"' in s
        assert re.search(r"models\s*=\s*\[[^\]]*\"spl\"", s, re.S)

    def test_shrinking_sigma_branch_uses_the_class_mask(self):
        """The branch must use t >= T_MIN, matching the constructor. A
        different mask optimises a different residual set than the one
        _init_shrinking_bands derives sigma from.

        Match the executable mask, not the identifier. A bare `"T_MIN" in
        <window>` is satisfied by the import line and by prose comments
        that merely mention T_MIN -- it stays green with the mask deleted
        or replaced by a wrong literal, i.e. green through exactly the
        "cleanup" it claims to prevent.
        """
        s = _src("tools/fit_shrinking_sigma.py")
        i = s.index('model_name == "spl"')
        assert re.search(r"np\.where\(\s*t\s*>=\s*T_MIN", s[i:])


class TestAppWiring:
    def test_registered_in_price_models(self):
        from conftest import _app_ctx
        assert "spl" in _app_ctx.PRICE_MODELS
        assert _app_ctx.PRICE_MODELS["spl"].short_name == "spl"

    def test_colour_in_every_palette(self):
        from colors import PALETTES, MODEL_TRACE_COLORS
        assert "spl" in MODEL_TRACE_COLORS
        for name, pal in PALETTES.items():
            assert "spl" in pal["model_colors"], f"missing in {name}"

    def test_checklist_options_append_only(self):
        """Bitmask positions are positional; the newest model must be LAST in
        each list. s2f_inst was appended after spl (2026-08-20) — spl is no
        longer the tail, which is exactly what append-only should show."""
        from snapshot import _CHECKLIST_OPTIONS
        lists = [k for k in _CHECKLIST_OPTIONS if k.endswith("-model-show")]
        assert len(lists) == 5, lists
        for k in lists:
            assert _CHECKLIST_OPTIONS[k][-1] == "s2f_inst", \
                f"{k}: newest model must be appended last, not inserted"
            assert "spl" in _CHECKLIST_OPTIONS[k], f"{k}: spl still present"

    def test_deprioritized(self):
        from layout.display_models import _DEPRIORITIZED
        assert "spl" in _DEPRIORITIZED

    def test_heatmap_pill_base_and_label(self):
        from layout.heatmap import _HM_PILL_MODELS_BASE, _HM_PILL_LABELS
        assert "spl" in _HM_PILL_MODELS_BASE
        assert _HM_PILL_LABELS["spl"] == "SatPL"

    def test_in_ticker_cycle(self):
        """spl IS in the navbar ticker percentile cycle (user-requested). The
        other diagnostic models (gomp/bpl/plo/sexp/logi) stay excluded."""
        from callbacks.ticker import _MODEL_CYCLE
        assert "spl" in _MODEL_CYCLE
        assert not ({"gomp", "bpl", "plo", "sexp", "logi"} & set(_MODEL_CYCLE))


class TestModelInfoCard:
    def test_newest_mi_card_is_last_in_both_lists(self):
        """Deep links are positional: _MODEL_INFO_ITEMS[n-1]. Inserting
        mid-list silently breaks every existing /mi.N link, so the newest card
        must be appended last. mi-s2f-inst was appended after mi-spl."""
        from layout.model_info import _MODEL_INFO_ITEM_IDS
        from callbacks.routing import _MODEL_INFO_ITEMS
        assert _MODEL_INFO_ITEM_IDS[-1] == "mi-s2f-inst"
        assert _MODEL_INFO_ITEMS[-1] == "mi-s2f-inst"
        assert "mi-spl" in _MODEL_INFO_ITEM_IDS

    def test_the_two_lists_agree(self):
        """layout/model_info is the SSOT (so stated at routing.py:535);
        routing.py holds a hand-maintained mirror. Nothing guarded this
        three-way sync before spl."""
        from layout.model_info import _MODEL_INFO_ITEM_IDS
        from callbacks.routing import _MODEL_INFO_ITEMS
        assert list(_MODEL_INFO_ITEM_IDS) == list(_MODEL_INFO_ITEMS)

    def test_accordion_order_matches_the_id_list(self):
        from layout.model_info import _MODEL_INFO_ITEM_IDS
        from layout.model_info._items import _build_accordion_items
        rendered = [getattr(i, "item_id", None) for i in _build_accordion_items()]
        assert rendered == list(_MODEL_INFO_ITEM_IDS)

    def test_pre_existing_deep_links_still_resolve(self):
        """The positional hazard, pinned from the other side. These are the
        item_ids /mi.23../mi.29 resolved to BEFORE spl was added; appending
        mi-spl last must leave every one of them where it was."""
        from callbacks.routing import _mi_item_for_pathname
        for n, expected in enumerate(
            ["mi-s2f", "mi-mc", "mi-ef", "mi-u1",
             "mi-compare", "mi-regimes", "mi-citadel"], start=23
        ):
            assert _mi_item_for_pathname(f"/mi.{n}") == expected
            assert _mi_item_for_pathname(f"/9.{n}") == expected
        assert _mi_item_for_pathname("/mi.30") == "mi-spl"

    def test_every_card_image_exists_on_disk(self):
        """A card image that is not committed is an HTTP 500 on a public page,
        and nothing else fails: the layout builds, every test passes, and only
        a reader sees it. Both spl figures shipped exactly that way until
        tools/render_spl_figure.py rendered them."""
        from layout.model_info._items import _build_accordion_items
        srcs = set(re.findall(r"/assets/[A-Za-z0-9_.-]+\.(?:png|jpg|jpeg|svg)",
                              str([str(i) for i in _build_accordion_items()])))
        assert {"/assets/spl_linear_vs_log.png", "/assets/spl_profile.png"} <= srcs
        missing = [s for s in sorted(srcs)
                   if not (REPO / "btc_web" / s.lstrip("/")).exists()]
        assert not missing, f"referenced but not committed: {missing}"


class TestOffChecklistRegistries:
    """Neither of these is in workflow_new_model.md's 26 steps."""

    def test_scanner_order(self):
        """Benign -- _scanner_sort_key sinks unknown keys via except ValueError
        -- but it is a real registry, and an unlisted model sorts to the bottom
        instead of into its Display Models slot."""
        from callbacks.scanner import _SCANNER_ORDER
        assert "spl" in _SCANNER_ORDER
        assert _SCANNER_ORDER.index("spl") > _SCANNER_ORDER.index("bpl")

    def test_architecture_doc_model_table(self):
        """docs/architecture.md is SERVED at /docs/architecture (api.py), so
        staleness here is publicly visible."""
        row = [ln for ln in _src("docs/architecture.md").splitlines()
               if ln.startswith("| `spl` |")]
        assert len(row) == 1, "expected exactly one spl row in the model table"
        assert "SaturatingPowerLawModel" in row[0]

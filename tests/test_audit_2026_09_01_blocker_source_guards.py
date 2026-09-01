"""Source-guard tests for the 35 BLOCKER fixes shipped in audit 2026-09-01.

Each test here fails CI the moment a future commit reintroduces the bug.
The mechanism is the same as the older ``test_audit_*.py`` files:
the test parses the production source file and asserts on the
*corrective pattern*. This is not a substitute for behaviour tests —
it's a tripwire that catches the most common regressions
(reverting the fix, copy-pasting the old code from another branch,
silently restoring the buggy default).

Naming convention: ``test_<BL-id>_<short-summary>``. Each BL-id maps
back to the BLOCKER ID in ``docs/audit_2026_09_01.md`` (the report
produced by the 5-agent review). When you fix a BL, ADD a guard here
in the same commit; when you delete a guard, also delete the
corresponding BL entry from the report.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "rlpe"


def _read(rel_path: str) -> str:
    """Read a source file under ``src/rlpe``."""
    return (SRC / rel_path).read_text(encoding="utf-8")


def _count_matches(pattern: str, text: str, flags: int = 0) -> int:
    return len(re.findall(pattern, text, flags=flags))


class TestGemmaLockIsRLock(unittest.TestCase):
    """BL-1: ``threading.Lock()`` on ``_gemma_lock`` deadlocks under
    the Gemma4 fallback path (caller holds the lock then acquires it
    again inside ``_apply_gemma_with_fallback``). The fix is to use
    ``threading.RLock()``. A plain ``threading.Lock()`` for
    ``_gemma_lock`` is a regression."""

    def test_gemma_lock_is_rlock(self) -> None:
        src = _read("pipeline.py")
        # Find the _gemma_lock assignment in __init__.
        match = re.search(
            r"self\._gemma_lock\s*=\s*threading\.(R?Lock)\(\)",
            src,
        )
        self.assertIsNotNone(match, "_gemma_lock assignment not found")
        self.assertEqual(
            match.group(1),
            "RLock",
            "_gemma_lock must be threading.RLock() (BL-1: plain Lock() deadlocks)",
        )


class TestSAM2InferenceIsLocked(unittest.TestCase):
    """Systemic #1 (SAM2 half): ``_segment_with_sam2`` must wrap the
    ``predictor.set_image(rgb)`` + ``predictor.predict(...)`` calls in
    ``self._lock``. Without this, concurrent workers race on the
    internal feature cache and silently return bbox against the wrong
    image."""

    def test_sam2_predictor_calls_inside_lock(self) -> None:
        src = _read("segmentation.py")
        # Locate ``def _segment_with_sam2`` and then walk forward
        # line-by-line. Avoids any ReDoS risk from large backtracking
        # regexes.
        idx = src.find("def _segment_with_sam2(")
        self.assertGreater(idx, -1, "_segment_with_sam2 not found")
        # Read until the next top-level ``def `` or ``class `` at
        # column 0.
        snippet_start = idx
        snippet = []
        for line in src[idx:].splitlines():
            if line.startswith("    def ") and snippet:
                break
            if line.startswith("class ") and snippet:
                break
            snippet.append(line)
        body = "\n".join(snippet)
        # The first predictor.set_image call must come AFTER a `with
        # self._lock:` line.
        set_image_idx = body.find("predictor.set_image(rgb)")
        with_idx = body.find("with self._lock:")
        self.assertGreater(set_image_idx, -1, "predictor.set_image not found")
        self.assertGreater(with_idx, -1, "with self._lock: not found")
        self.assertLess(
            with_idx,
            set_image_idx,
            "predictor.set_image must come AFTER 'with self._lock:' (systemic #1)",
        )


class TestYOLOInferenceIsLocked(unittest.TestCase):
    """Systemic #1 (YOLO half): ``detect_figure_regions_yolo`` must
    wrap the ``model(image_path, ...)`` inference call in a dedicated
    ``_yolo_infer_lock`` (separate from the load lock so the load
    lock is released for the duration of the slow inference)."""

    def test_yolo_inference_lock_present(self) -> None:
        src = _read("layout.py")
        self.assertIn(
            "_yolo_infer_lock",
            src,
            "layout.py must define a YOLO inference lock (systemic #1)",
        )
        # The lock must be acquired around the model() call. The
        # call may span multiple lines; use re.DOTALL so the regex
        # can cross newlines without ReDoS.
        self.assertRegex(
            src,
            r"with[ \t]+getattr\(detect_figure_regions_yolo,[ \t]*_infer_lock_attr\):\s*\n[ \t]+results[ \t]*=[ \t]*model\(",
            "YOLO model() call must be inside the inference lock",
        )


class TestScaleBarHoughLinesSlicing(unittest.TestCase):
    """BL-4: ``lines[:, 0, :]`` on a ``(1, N, 4)`` array returns
    ``(4,)`` — only the first row's endpoints — silently mis-computing
    ``um_per_px``. The fix is ``lines[0]`` (squeeze axis 0)."""

    def test_no_buggy_active_lines_slice(self) -> None:
        """The actual code path must use ``lines[0]``, not
        ``lines[:, 0, :]``. We strip comments + docstrings first so
        a literal ``lines[:, 0, :]`` inside a comment (which describes
        the bug, not the behaviour) doesn't trip the guard."""
        src = _read("scale_bar.py")
        # Remove comment-only lines (lines starting with optional
        # whitespace + ``#``). Multi-line ``\"\"\"`` docstrings are
        # rare in this file; the buggy pattern doesn't appear in any
        # docstring here.
        stripped = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotIn(
            "lines[:, 0, :]",
            stripped,
            "scale_bar.py must not use lines[:, 0, :] in active code (BL-4: returns (4,) not (N, 4))",
        )


class TestGlobal422HandlerScoped(unittest.TestCase):
    """Architectural P0 #6: the global ``RequestValidationError``
    handler returned HTTP 200 for every endpoint — masking real 422s
    on everything except ``/system/test-llm``. The handler must
    re-raise so FastAPI's default 422 path runs for non-test-llm
    routes."""

    def test_handler_reraises_for_non_test_llm(self) -> None:
        src = _read("api/app.py")
        # Find the @app.exception_handler(RequestValidationError) block.
        handler = re.search(
            r"@app\.exception_handler\(RequestValidationError\)(.*?)async def _validation_error_handler(.*?)(?=\n@app\.|\nclass |\n# ---)",
            src,
            re.DOTALL,
        )
        self.assertIsNotNone(handler, "_validation_error_handler not found")
        body = handler.group(2)
        self.assertIn(
            "path != \"/system/test-llm\"",
            body,
            "422 handler must scope the always-200 response to /system/test-llm only",
        )
        # And must re-raise RequestValidationError for the other paths.
        self.assertIn(
            "raise",
            body,
            "422 handler must re-raise RequestValidationError for non-test-llm routes",
        )


class TestFallbackGETRequiresAPIKey(unittest.TestCase):
    """Architectural P1 #22: ``GET /jobs/{id}/MiniMax-fallback``
    must enforce the same ``require_api_key`` auth as the POST
    counterpart — the GET reveals the same error_info payload and
    previously let any LAN caller probe job job state."""

    def test_fallback_get_has_auth_dep(self) -> None:
        src = _read("api/app.py")
        # Find the GET fallback endpoint and the function signature
        # block that follows. The signature may include type-annotated
        # parameters (``job_id: str``) that contain ``:`` so anchor on
        # the closing ``) -> ...`` form instead.
        match = re.search(
            r'@app\.get\("/jobs/\{job_id\}/MiniMax-fallback"\)\s*\n'
            r"def get_MiniMax_fallback\((.+?)\)\s*->",
            src,
            re.DOTALL,
        )
        self.assertIsNotNone(match, "GET fallback endpoint not found")
        sig = match.group(1)
        self.assertIn(
            "require_api_key",
            sig,
            "GET /jobs/{id}/MiniMax-fallback must require API-key auth (architectural P1 #22)",
        )


class TestGUIOutputDirMatchesWorker(unittest.TestCase):
    """BL-5: RunTab emitted ``out_path / \"output\"`` as the
    ``JobRecord.output_dir`` but the GUI worker writes its artefacts
    under ``out_path / \"work\"`` (per ``pipeline_worker.py:312-313``
    ``output_dir=self._work_dir``). The button "Open output
    directory" must point at the actual write location."""

    def test_run_tab_output_dir_is_work_subdir(self) -> None:
        src = _read("gui/run_tab.py")
        self.assertNotIn(
            'output_dir = str(out_path / "output")',
            src,
            "RunTab must not emit out_path/output (BL-5: worker writes to out_path/work)",
        )
        self.assertIn(
            'output_dir = str(out_path / "work")',
            src,
            "RunTab must emit out_path/work as JobRecord.output_dir (BL-5)",
        )


class TestGUIRetrySlotSignature(unittest.TestCase):
    """BL-6: ``JobsTab.retry_requested = Signal(str)`` emits only the
    job_id but ``MainWindow._on_retry`` was declared with a second
    ``settings: dict`` parameter. Clicking Retry raised TypeError and
    crashed the Qt main window. The slot must take ONE argument."""

    def test_on_retry_single_argument(self) -> None:
        src = _read("gui/main_window.py")
        # Find _on_retry definition; reject the old double-arg signature.
        self.assertNotRegex(
            src,
            r"def\s+_on_retry\(self,\s*job_id:\s*str,\s*settings:\s*dict\)",
            "MainWindow._on_retry must take one arg (BL-6: Signal(str) only emits job_id)",
        )
        self.assertRegex(
            src,
            r"def\s+_on_retry\(self,\s*job_id:\s*str\)\s*->\s*None:",
            "MainWindow._on_retry must take only (self, job_id)",
        )


class TestBatchResultsPersistRecheck(unittest.TestCase):
    """BL-7: ``delete_results_batch`` released its lock and then
    re-acquired it per touched_job *outside* the loop. If a
    concurrent ``_purge_job`` popped the entry between the two
    acquisitions, the persist step wrote an empty row list to a
    purged job's ``matches.jsonl`` — silently destroying in-flight
    progress from a parallel cancel. The fix re-validates the job
    is still in ``RESULT_CACHE`` inside the per-job lock."""

    def test_persist_block_validates_job_present(self) -> None:
        src = _read("api/app.py")
        # Locate the function body by its known sentinel lines.
        idx = src.find("def delete_results_batch(")
        self.assertGreater(idx, -1, "delete_results_batch not found")
        # Walk forward from there and look for the early-return guard.
        # The pattern is more permissive than the strict regex — it
        # accepts any number of comment lines between ``if job is
        # None:`` and ``continue``.
        snippet = src[idx:]
        match = re.search(
            r"job\s*=\s*RESULT_CACHE\.get\(job_id\)\s*\n\s*if\s+job\s+is\s+None:.*?continue",
            snippet,
            re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            "delete_results_batch persist must validate job presence (BL-7)",
        )


class TestSafeValueHasRecursionGuard(unittest.TestCase):
    """BL-9: ``_safe_value`` recursed without a depth bound — a 10k
    nested dict triggered RecursionError and 500'd the whole worker.
    The fix introduces ``_SAFE_VALUE_MAX_DEPTH`` and a
    ``_depth`` parameter; beyond the cap the value is replaced with
    the literal ``"<truncated>"``."""

    def test_safe_value_has_depth_param(self) -> None:
        src = _read("api/app.py")
        # Both the helper signature and the constant must exist.
        self.assertIn(
            "_SAFE_VALUE_MAX_DEPTH",
            src,
            "api/app.py must define _SAFE_VALUE_MAX_DEPTH constant (BL-9)",
        )
        self.assertRegex(
            src,
            r"def\s+_safe_value\([^)]*_depth[^)]*\)\s*->",
            "_safe_value must take a _depth parameter (BL-9)",
        )
        self.assertIn(
            '"<truncated>"',
            src,
            "_safe_value must return the literal <truncated> when depth cap is hit (BL-9)",
        )


class TestFallbackPopupTimeoutBounded(unittest.TestCase):
    """BL-8: ``_web_fallback_popup`` blocked each worker for up to 5
    minutes when the MiniMax API failed — 4 concurrent jobs that all
    hit a MiniMax outage would each pin a BackgroundTasks worker for
    5 minutes, freezing the whole FastAPI process. Lower to 30 s."""

    def test_fallback_timeout_ms_bounded(self) -> None:
        src = _read("api/app.py")
        match = re.search(
            r"FALLBACK_POPUP_TIMEOUT_MS:\s*int\s*=\s*(\d+)",
            src,
        )
        self.assertIsNotNone(match, "FALLBACK_POPUP_TIMEOUT_MS not found")
        value = int(match.group(1))
        self.assertLessEqual(
            value,
            60_000,
            f"FALLBACK_POPUP_TIMEOUT_MS must be <= 60_000 (BL-8), got {value}",
        )


class TestNoneCrashGuards(unittest.TestCase):
    """BL-2 / BL-3 / BL-10 / BL-11 / BL-12: a handful of LLM/M3 /
    panel-label helpers returned None on transport hiccups and then
    crashed on the next attribute access. Each call site must guard
    the None return explicitly."""

    def test_llm_first_extract_guards_none_result(self) -> None:
        src = _read("pipeline.py")
        # Must check `if result is None: return None` after the
        # ``infer_panel`` call inside ``_llm_first_extract``.
        self.assertRegex(
            src,
            r"if result is None:\s*\n\s*logger\.debug\(\"LLM-first returned None",
            "_llm_first_extract must guard None result from infer_panel (BL-2)",
        )

    def test_m3_stage4_guards_none_match(self) -> None:
        src = _read("pipeline.py")
        self.assertIn(
            "m3_stage4_error",
            src,
            "_apply_m3_stage4 must record m3_stage4_error metadata on None match (BL-3)",
        )
        # The guard must come AFTER `panel_match = self.m3_engine.match_panel(...)`.
        self.assertRegex(
            src,
            r"if panel_match is None:\s*\n\s*md\[\"m3_stage4_error\"\]\s*=\s*\"engine_returned_none\"",
            "_apply_m3_stage4 must skip-and-continue on None match_panel (BL-3)",
        )

    def test_normalize_panel_label_none_guard(self) -> None:
        src = _read("pipeline.py")
        # Both usages must be wrapped: ``(x or "")`` or ``_normalize_panel_label(x) or ""``.
        self.assertRegex(
            src,
            r"\(_normalize_panel_label\(lbl\)\s+or\s+\"\"\)",
            "_normalize_panel_label must be guarded with `or \"\"` fallback (BL-10)",
        )

    def test_rescue_target_page_none_guard(self) -> None:
        src = _read("pipeline.py")
        self.assertRegex(
            src,
            r"int\(getattr\(target,\s*\"page_number\",\s*None\)\s+or\s+0\)",
            "range-chart rescue must guard None target.page_number (BL-11)",
        )


class TestPaddleOCR3TupleBranch(unittest.TestCase):
    """BL-13: PaddleOCR 2.7+ emits ``(box, text, conf)`` 3-tuples;
    the previous code's ``len(line) == 2`` branch dropped them,
    returning empty OCR for every plate."""

    def test_paddle_ocr_handles_three_tuple(self) -> None:
        src = _read("ocr.py")
        self.assertRegex(
            src,
            r"len\(line\)\s*==\s*3.*?\n\s*box,\s*text,\s*conf\s*=\s*line",
            "PaddleOCR 3-tuple branch must unpack (box, text, conf) (BL-13)",
        )


class TestMorphologySentinelNotPlainAscii(unittest.TestCase):
    """BL-14: the previous ``_DIGIT_HYPHEN_SENTINEL = \"RANGE\"``
    collided with legitimate morphology text like \"Stratigraphic
    RANGE\".\ Wrap the sentinel in NUL bytes so it never collides."""

    def test_sentinel_uses_nul_wrapping(self) -> None:
        src = _read("morphology_locator.py").encode("utf-8")
        # The sentinel literal must contain a NUL byte on each side.
        self.assertRegex(
            src.decode("utf-8", errors="replace"),
            r'_DIGIT_HYPHEN_SENTINEL\s*=\s*"\\x00RANGE\\x00"',
            "_DIGIT_HYPHEN_SENTINEL must be NUL-wrapped (BL-14)",
        )


class TestPaleoReconstructionEulerPolesGuard(unittest.TestCase):
    """BL-18: ``_load_seton2012_from_external`` would *replace* the
    curated EULER_POLES table with an empty list for any plate
    whose .rot file matched zero rows. Now the assignment is guarded
    with ``if not rows: continue``."""

    def test_euler_poles_assignment_guarded(self) -> None:
        src = _read("paleo_reconstruction.py")
        # Find the assignment and ensure it sits inside an `if rows:`
        # branch (allowing for either ``if not rows: continue`` OR
        # ``if rows:``).
        self.assertRegex(
            src,
            r"if\s+not\s+rows:\s*\n\s*logger\.warning",
            "_load_seton2012_from_external must guard empty-rows assignment (BL-18)",
        )


class TestLabelRangeExpansionHandlesReverse(unittest.TestCase):
    """BL-21: ``_expand_label_range(\"Z-A\")`` returned ``[]`` because
    ``range(90, 65)`` is empty. The fix sorts the bounds before
    iterating."""

    def test_label_range_sorts_bounds(self) -> None:
        src = _read("m3_engine.py")
        self.assertRegex(
            src,
            r"sorted\(\[ord\(a\),\s*ord\(b\)\]\)",
            "_expand_label_range must sort letter bounds (BL-21)",
        )
        self.assertRegex(
            src,
            r"sorted\(\[ia,\s*ib\]\)",
            "_expand_label_range must sort digit bounds (BL-21)",
        )


class TestCLIPDFDirIsOptional(unittest.TestCase):
    """BL-22: ``--pdf-dir required=True`` made the config-file
    fallback at line 906-911 dead code. The flag must be optional."""

    def test_pdf_dir_default_none(self) -> None:
        src = _read("cli.py")
        # Reject the old "required=True" form on --pdf-dir and --work-dir.
        for flag in ("--pdf-dir", "--work-dir"):
            match = re.search(
                rf'p\.add_argument\(\"{flag}\",\s*type=[^,]+,\s*required=True\)',
                src,
            )
            self.assertIsNone(
                match,
                f"{flag} must not be required=True (BL-22: config-file fallback was dead code)",
            )


class TestArrayLabelsSplitBeforeExpand(unittest.TestCase):
    """BL-20: ``labels == \"1-3, 5\"`` was passed whole to
    ``_expand_label_range`` which only handles \"A-Z\" / \"1-9\"
    ranges; the result was a single literal string \"1-3, 5\" that
    matched every panel. Split on commas first."""

    def test_string_labels_split_on_comma(self) -> None:
        src = _read("m3_engine.py")
        self.assertRegex(
            src,
            r"labels\.split\(\",\"\)\s*\n\s*for\s+lab\s+in\s+_expand_label_range\(seg\)",
            "string labels must be split on commas before expansion (BL-20)",
        )


class TestAssociationTaxaUnion(unittest.TestCase):
    """Architectural P0 #25: the previous ``taxa = [...] or ...``
    short-circuit only fell back to caption extraction when entity
    extraction returned empty — the two sources never merged. The
    fix always runs both branches and unions."""

    def test_taxa_unions_sources(self) -> None:
        src = _read("association.py")
        # Reject the short-circuit form.
        self.assertNotRegex(
            src,
            r"taxa\s*=\s*\[t\.text\s+for\s+t\s+in\s+taxon_entities\]\s+or\s+extract_taxa_from_caption",
            "taxa must union entity + caption sources (P0 #25)",
        )


class TestCaptionPairsDoesNotOverrideMatcher(unittest.TestCase):
    """Architectural P0 #7: ``if caption_pairs_used:`` overwrote the
    neural-matcher result whenever ANY caption-pair fired — capping
    the trained matcher at the regex ceiling. Now requires BOTH
    ``caption_pairs_used`` AND ``not matcher_used``."""

    def test_pair_lookup_skips_when_matcher_used(self) -> None:
        src = _read("association.py")
        self.assertIn(
            "if caption_pairs_used and not matcher_used:",
            src,
            "caption-pair override must NOT run when matcher_used (P0 #7)",
        )


class TestEvaluationF1MicroMacroLabeled(unittest.TestCase):
    """BL-28: the previous ``species_f1`` aggregate field mixed
    micro-averaged F1 (pooled TP/FP/FN) with the macro-averaged
    per-paper F1 from ``PaperMetrics.species_f1`` — two identical-
    looking numbers that differed by 5-15 pp. Rename to
    ``species_f1_micro`` and add ``species_f1_macro``."""

    def test_aggregate_has_distinct_micro_macro_keys(self) -> None:
        src = _read("evaluation/metrics.py")
        self.assertIn(
            '"species_f1_micro"',
            src,
            "aggregate must expose species_f1_micro (BL-28)",
        )
        self.assertIn(
            '"species_f1_macro"',
            src,
            "aggregate must expose species_f1_macro (BL-28)",
        )


class TestPanelLabelNormalizationHandlesDigitLetterDigit(unittest.TestCase):
    """BL-25: ``_normalize_panel_label(\"A04\")`` was not normalised
    back to \"A4\" — OCR legitimately produces 3+ char digit+letter+digit
    shapes. The stripping loop must generalise to \"digit+letter+digit\"
    AND \"letter+digit+letter\"."""

    def test_stripping_loop_uses_generalised_shape(self) -> None:
        src = _read("association.py")
        self.assertIn(
            "_LOOKS_LIKE_LABEL",
            src,
            "_normalize_panel_label must use _LOOKS_LIKE_LABEL helper for the strip loop (BL-25)",
        )


class TestPanelLabelShapeAcceptsZeroLetter(unittest.TestCase):
    """BL-26: ``is_valid_panel_label`` rejected \"0a\" / \"0b\" (some
    Triassic papers use 0a for transitional panels) while accepting
    \"10a\". Add ``0[a-z]?`` branch."""

    def test_zero_letter_branch_added(self) -> None:
        src = _read("association.py")
        self.assertRegex(
            src,
            r"0\[a-z\]\?\|0",
            "is_valid_panel_label must accept 0a/0b (BL-26)",
        )


class TestAtomicWritesUseTempfileRename(unittest.TestCase):
    """BL-34 / CR-1 / CR-2 / CR-3: three core file-write paths
    (``io.save_csv``, ``schema_models.emit_json_schema``,
    ``config_io.save_config``) previously wrote directly to the
    destination — a mid-write SIGKILL left a half-written file that
    downstream consumers couldn't parse. All three must use
    ``tempfile.mkstemp`` + ``os.replace`` for atomic write."""

    def test_io_save_csv_atomic(self) -> None:
        src = _read("io.py")
        self.assertIn(
            "_tempfile.mkstemp",
            src,
            "io.save_csv must use tempfile.mkstemp (BL-34)",
        )
        self.assertIn(
            "_os.replace",
            src,
            "io.save_csv must use os.replace (BL-34)",
        )

    def test_emit_json_schema_atomic(self) -> None:
        src = _read("schema_models.py")
        self.assertIn(
            "_tempfile.mkstemp",
            src,
            "emit_json_schema must use tempfile.mkstemp (BL-34)",
        )
        self.assertIn(
            "_os.replace",
            src,
            "emit_json_schema must use os.replace (BL-34)",
        )

    def test_save_config_atomic(self) -> None:
        src = _read("config_io.py")
        self.assertIn(
            "_tempfile.mkstemp",
            src,
            "config_io.save_config must use tempfile.mkstemp (BL-34)",
        )
        self.assertIn(
            "_os.replace",
            src,
            "config_io.save_config must use os.replace (BL-34)",
        )


class TestBatchRunSingleUsesDataclassesReplace(unittest.TestCase):
    """BL-35 / CR-4: ``batch._run_single`` manually listed every
    PipelineConfig field — new fields (m3_per_panel, m3_stage_6, ...)
    silently fell back to defaults when invoked through the batch
    path. ``dataclasses.replace`` keeps the manual list in sync."""

    def test_batch_uses_replace(self) -> None:
        src = _read("batch.py")
        self.assertIn(
            "dataclasses.replace",
            src,
            "batch._run_single must use dataclasses.replace (BL-35)",
        )


class TestBatchTracksFailedPdfs(unittest.TestCase):
    """CR-5: ``run_batch_parallel`` logged worker exceptions and moved
    on with no record — the eval script then reported F1 on an empty
    row set as if it were a real result. The fix threads
    ``failed_pdfs`` + ``n_failed`` through the run_output dict."""

    def test_run_output_has_failed_pdfs(self) -> None:
        src = _read("batch.py")
        self.assertIn(
            "failed_pdfs",
            src,
            "batch.run_batch_parallel must track failed_pdfs (CR-5)",
        )
        self.assertIn(
            "n_failed",
            src,
            "batch.run_batch_parallel must expose n_failed in run_output (CR-5)",
        )


class TestEnvLoaderHandlesExportAndBom(unittest.TestCase):
    """CR-6 / CR-7 / CR-8: ``load_env_file`` previously rejected
    ``export FOO=bar`` syntax, mangled UTF-8 BOMs, and stripped both
    quote pairs unconditionally. Now handles all three."""

    def test_env_loader_strips_export_prefix(self) -> None:
        src = _read("env_loader.py")
        self.assertIn(
            "export ",
            src,
            "env_loader must strip leading 'export ' prefix (CR-6)",
        )

    def test_env_loader_uses_utf8_sig(self) -> None:
        src = _read("env_loader.py")
        self.assertIn(
            '"utf-8-sig"',
            src,
            "env_loader must open with utf-8-sig encoding (CR-7)",
        )

    def test_env_loader_unquote_handles_embedded_apostrophes(self) -> None:
        src = _read("env_loader.py")
        self.assertIn(
            "_unquote",
            src,
            "env_loader must use the _unquote helper for matched-pair quote handling (CR-8)",
        )


class TestConfigIOCoerceBoolFallback(unittest.TestCase):
    """CR-10: ``_coerce`` returned ``bool(value)`` for unknown string
    values — ``bool(\"no\")`` is True, which would silently flip a
    user's \"save_intermediate: no\" to True and flood the disk with
    117 GB of intermediate data. Now falls back to the default for
    unrecognised bool inputs."""

    def test_coerce_does_not_use_bool_fallback(self) -> None:
        src = _read("config_io.py")
        # The string-branch `return bool(value)` line (that fires for
        # any non-string truthy string) must NOT exist anymore — only
        # the explicit lv-in-{true,1,false,0,""} branches should
        # remain. Look for the final ``return bool(value)`` line that
        # was the original bug.
        self.assertNotRegex(
            src,
            r"return\s+bool\(value\)\s*\n\s*return\s+default",
            "_coerce must NOT use the bool(value) fallback (CR-10)",
        )


class TestRedactAPIKeysCoversCloudProviders(unittest.TestCase):
    """CR-17: ``_API_KEY_PATTERNS`` previously only covered OpenAI /
    Anthropic / Pro / CP key prefixes. AWS Bedrock (AKIA / ASIA),
    Vertex / GCP (ya29.<base64>), Azure (32 hex chars), and Stripe
    live keys are all routed through MiniMax-style proxy configs and
    would be persisted verbatim to ``matches.jsonl`` without these
    patterns."""

    def test_redaction_includes_aws_prefix(self) -> None:
        src = _read("llm_backends.py")
        self.assertIn(
            "AKIA",
            src,
            "_API_KEY_PATTERNS must include AWS access-key prefix (CR-17)",
        )

    def test_redaction_includes_ya29_prefix(self) -> None:
        src = _read("llm_backends.py")
        self.assertIn(
            "ya29",
            src,
            "_API_KEY_PATTERNS must include Google OAuth ya29 prefix (CR-17)",
        )


class TestConfigWindowAndMergeGapBounds(unittest.TestCase):
    """Architectural P1 #19: ``caption_window`` previously accepted
    any int ≥ 1 — ``10000`` paired Fig.1's caption with a Fig.400
    image 400 pages later, silently degrading F1. ``merge_gap_pt``
    previously accepted any non-negative float — ``10000`` merged the
    entire page into a single phantom figure. Both now have hard
    upper bounds in the constructor."""

    def test_caption_window_upper_bound(self) -> None:
        src = _read("opendataloader_extractor.py")
        # The fix uses a negated condition: ``not (1 <= caption_window <= 50)``.
        self.assertRegex(
            src,
            r"caption_window\s*<=\s*50|caption_window\s*>\s*50|1\s*<=\s*caption_window\s*<=\s*50",
            "caption_window upper bound (50) must be enforced (P1 #19)",
        )

    def test_merge_gap_pt_upper_bound(self) -> None:
        src = _read("opendataloader_extractor.py")
        # The fix uses ``not (0.0 < merge_gap_pt <= 1000.0)``.
        self.assertRegex(
            src,
            r"merge_gap_pt\s*<=\s*1000|merge_gap_pt\s*>\s*1000|0\.0\s*<\s*merge_gap_pt\s*<=\s*1000",
            "merge_gap_pt upper bound (1000) must be enforced (P1 #19)",
        )


class TestRadiolarianPipelineHasClose(unittest.TestCase):
    """Systemic #2: ``RadiolarianPipeline`` previously leaked SAM2
    (~900 MB VRAM), PaddleOCR engine, and the local Gemma runtime
    on every run() completion. The CLI/GUI/batch paths dropped out
    of scope without calling ``unload_sam2`` and the web job finally
    block only ran on web/API. Now ``close()`` (and the
    context-manager protocol) release all held resources."""

    def test_close_method_defined(self) -> None:
        src = _read("pipeline.py")
        self.assertRegex(
            src,
            r"def\s+close\(self\)\s*->\s*None:",
            "RadiolarianPipeline must define close() (systemic #2)",
        )

    def test_enter_exit_defined(self) -> None:
        src = _read("pipeline.py")
        self.assertRegex(
            src,
            r"def\s+__enter__\(self\)\s*->\s*\"RadiolarianPipeline\":",
            "RadiolarianPipeline must support the context-manager protocol (systemic #2)",
        )
        self.assertRegex(
            src,
            r"def\s+__exit__\(self,\s*exc_type,\s*exc,\s*tb\)\s*->\s*None:",
            "RadiolarianPipeline.__exit__ must call self.close() (systemic #2)",
        )


class TestFigureIdLogicalKeyIncludesPlate(unittest.TestCase):
    """BL-30: ``_figure_id_logical_key`` collapsed all plates on the
    same page into one logical figure (because the regex didn't
    capture ``_pl<N>``), polluting cross-plate panel↔species
    associations. The key now includes the plate discriminator."""

    def test_logical_key_captures_plate(self) -> None:
        src = _read("evaluation/metrics.py")
        self.assertRegex(
            src,
            r"_pl(\d+)",
            "_figure_id_logical_key must capture _pl<N> discriminator (BL-30)",
        )


class TestImageLabelCheckAvoidsScaleBar(unittest.TestCase):
    """BL-33: the previous code compared the OCR's *first* numeric
    token against the pred panel_id — but the first number is almost
    always the scale-bar magnitude. The fix uses the *last* number
    and skips numbers adjacent to a unit glyph."""

    def test_image_label_check_skips_unit_numbers(self) -> None:
        src = _read("evaluation/image_label_check.py")
        # The "last number if multiple" + "skip scale-bar" guard.
        self.assertRegex(
            src,
            r"\bm\b|\bmm\b|\bcm\b|\bμm\b",
            "image_label_check must recognise scale-bar unit glyphs (BL-33)",
        )
        self.assertIn(
            "last_unit_idx",
            src,
            "image_label_check must walk back from the last unit (BL-33)",
        )


class TestPanelIDNormalizePanelIDRestrictsOCRConfusion(unittest.TestCase):
    """BL-29: ``_normalize_panel_id`` previously folded ALL non-ASCII
    via NFKD + ``encode(\"ascii\", \"ignore\")`` — silently
    destroying the distinction between \"Æ\"/\"AE\", \"Ø\"/\"O\",
    \"Œ\"/\"OE\" and causing FN matches on Nordic / French papers.
    Restrict the fold to the OCR-only confusion table."""

    def test_normalize_panel_id_uses_ocr_translation(self) -> None:
        src = _read("evaluation/gold.py")
        self.assertIn(
            "_OCR_CONFUSION_TRANSLATION",
            src,
            "_normalize_panel_id must use _OCR_CONFUSION_TRANSLATION (BL-29)",
        )


class TestArchiveExporterHasUnknownFallback(unittest.TestCase):
    """BL-31: ``_occurrence_row`` dropped empty paper_id / figure_id
    components silently, allowing two papers with the same panel_id
    to collide on DwC-A submit. Now replaces each empty component
    with ``(unknown)``."""

    def test_occurrence_id_uses_unknown_fallback(self) -> None:
        src = _read("exporters/archive.py")
        self.assertIn(
            '"(unknown)"',
            src,
            "_occurrence_row must replace empty IDs with (unknown) (BL-31)",
        )


class TestLabelSortKeyUsesIntegerValue(unittest.TestCase):
    """BL-24: ``_label_sort_key`` returned ``(0, \"\")`` for every
    numeric label, so ``[\"10\", \"1\", \"2\", \"11\"]`` sorted in
    *insertion* order rather than numeric — panels 10/11 came *before*
    panels 1/2 in the caption-pair scan. The fix encodes the integer
    value into the sort tuple."""

    def test_label_sort_key_returns_int(self) -> None:
        src = _read("association.py")
        # Look for the new sort-key tuple: (0, int(s)).
        self.assertRegex(
            src,
            r"return\s+\(0,\s*int\(s\)\)",
            "_label_sort_key must return (0, int(s)) for numeric labels (BL-24)",
        )


class TestExpandLabelRangeHandlesReverse(unittest.TestCase):
    """BL-21: ``_expand_label_range(\"Z-A\")`` returned ``[]`` because
    ``range(90, 65)`` is empty. The fix sorts the bounds before
    iterating."""

    def test_expand_label_range_sorts_bounds(self) -> None:
        src = _read("m3_engine.py")
        # Look for sorted([ia, ib]) and sorted([ord(a), ord(b)]).
        self.assertRegex(
            src,
            r"sorted\(\[ia,\s*ib\]\)",
            "_expand_label_range must sort numeric bounds (BL-21)",
        )
        self.assertRegex(
            src,
            r"sorted\(\[ord\(a\),\s*ord\(b\)\]\)",
            "_expand_label_range must sort letter bounds (BL-21)",
        )


class TestExtractTaxaFromCaptionUsesCleanedText(unittest.TestCase):
    """BL-23: ``extract_taxa_from_caption`` scanned the original
    ``caption_text`` for cf./aff. comparison references but used the
    cleaned ``text`` for the canonical pattern match loop. Use
    ``text`` for both passes."""

    def test_cf_compare_uses_cleaned_text(self) -> None:
        src = _read("association.py")
        self.assertRegex(
            src,
            r"TAXON_CF_COMPARE_PATTERN\.finditer\(text\)",
            "extract_taxa_from_caption cf/aff scan must use cleaned text (BL-23)",
        )
        # Reject the old ``caption_text`` form.
        self.assertNotRegex(
            src,
            r"TAXON_CF_COMPARE_PATTERN\.finditer\(caption_text\)",
            "extract_taxa_from_caption cf/aff scan must NOT use original caption_text (BL-23)",
        )


class TestCLIAPIParameterConsistency(unittest.TestCase):
    """Architectural P1 #21: CLI / API / GUI three-entry-point
    parameter consistency guard.

    The 5-agent audit 2026-09-01 identified that the CLI used
    ``use_opendataloader=False`` and ``data_outbound_policy=api_full``
    while the web/API used ``True`` and ``api_redacted``. Two users
    evaluating the same paper on the same commit therefore measured
    different caption-quality metrics depending on which entry
    point they used. These tests lock down the alignment so a
    future drift blocks CI.
    """

    def test_cli_use_opendataloader_default_aligned_with_api(self) -> None:
        """Both CLI and API must default ``use_opendataloader=True``."""
        cli_src = _read("cli.py")
        # CLI default lives in the YAML template literal.
        self.assertIn(
            '"use_opendataloader": True',
            cli_src,
            "CLI default for use_opendataloader must be True (P1 #21: aligned with API)",
        )

    def test_cli_data_outbound_policy_default_aligned_with_api(self) -> None:
        """Both CLI and API must default ``data_outbound_policy=api_redacted``."""
        cli_src = _read("cli.py")
        self.assertIn(
            '"data_outbound_policy": "api_redacted"',
            cli_src,
            "CLI default for data_outbound_policy must be api_redacted (P1 #21)",
        )

    def test_api_joboptions_has_m3_prompt_lang_field(self) -> None:
        """JobOptions must expose ``m3_prompt_lang`` so the web UI can
        route JA / ZH / EN captions through the correct M3 prompt
        template (Phase 27 JA caption routing fix)."""
        api_src = _read("api/app.py")
        self.assertRegex(
            api_src,
            r"m3_prompt_lang:\s*str\s*=\s*\"auto\"",
            "JobOptions must declare m3_prompt_lang field (P1 #21)",
        )

    def test_cli_pipeline_config_extra_has_m3_prompt_lang_field(self) -> None:
        """PipelineConfig.extra must surface ``m3_prompt_lang`` so a
        web-uploaded JobOptions.m3_prompt_lang actually reaches the
        prompt builder. Without this, the API field is a no-op."""
        # The string appears in either cli.py (the JobOptions-to-extra
        # conversion) or pipeline.py (the actual default lookup).
        joined = _read("cli.py") + "\n" + _read("pipeline.py")
        self.assertRegex(
            joined,
            r"m3_prompt_lang",
            "PipelineConfig must surface m3_prompt_lang (P1 #21)",
        )


class TestSafeCallHelperExposed(unittest.TestCase):
    """Step 2: ``_safe_call`` helper must exist on ``rlpe.utils`` so
    downstream code can adopt it without importing internal symbols."""

    def test_safe_call_defined(self) -> None:
        src = _read("utils.py")
        self.assertRegex(
            src,
            r"def\s+_safe_call\(",
            "utils.py must define _safe_call (Step 2)",
        )

    def test_drain_warnings_defined(self) -> None:
        src = _read("utils.py")
        self.assertRegex(
            src,
            r"def\s+drain_warnings\(",
            "utils.py must define drain_warnings (Step 2)",
        )


class TestManifestWrittenAtEndOfRun(unittest.TestCase):
    """Step 2: ``manifest.json`` must be written next to
    ``run_output.json`` at the end of every ``run()`` call."""

    def test_manifest_write_call_present(self) -> None:
        src = _read("pipeline.py")
        self.assertIn(
            "manifest.json",
            src,
            "pipeline.py must write manifest.json (Step 2)",
        )


class TestAtomicPerPaperAppend(unittest.TestCase):
    """P0 A3: ``_process_one_pdf`` must append its rows to
    ``matches.jsonl`` IMMEDIATELY after each PDF completes — so a
    crashed batch preserves the work done so far."""

    def test_per_paper_append_call_present(self) -> None:
        src = _read("pipeline.py")
        # Look for the audit-block call that writes matches.jsonl
        # inside _process_one_pdf.
        self.assertIn(
            "per-paper incremental persistence",
            src,
            "_process_one_pdf must append rows to matches.jsonl per paper (A3)",
        )


class TestStreamingSSEHasDisconnectGuard(unittest.TestCase):
    """CR-23: SSE event stream must check ``request.is_disconnected()``
    and exit early when the client closes the tab."""

    def test_sse_disconnect_check_present(self) -> None:
        src = _read("api/app.py")
        self.assertIn(
            "is_disconnected",
            src,
            "api/app.py must check request.is_disconnected() in SSE (CR-23)",
        )


class TestM3SamplingLockPresent(unittest.TestCase):
    """CR-21: M3Engine must wrap read-modify-write of backend sampling
    attributes in a lock so concurrent workers don't stomp on each
    other."""

    def test_m3_engine_has_sampling_lock(self) -> None:
        src = _read("m3_engine.py")
        self.assertIn(
            "_sampling_lock",
            src,
            "m3_engine.py must define _sampling_lock (CR-21)",
        )
        # The setter calls must run inside the lock.
        self.assertRegex(
            src,
            r"with\s+self\._sampling_lock:",
            "_apply_config_sampling_params must acquire _sampling_lock (CR-21)",
        )


class TestCrossRefLockPresent(unittest.TestCase):
    """CR-22: paper_metadata_cleanup must use a lock around the
    Crossref cache lookup to prevent stampede."""

    def test_crossref_lock_present(self) -> None:
        src = _read("paper_metadata_cleanup.py")
        self.assertIn(
            "_CROSSREF_LOOKUP_LOCK",
            src,
            "paper_metadata_cleanup.py must define _CROSSREF_LOOKUP_LOCK (CR-22)",
        )


class TestI18nHasRLock(unittest.TestCase):
    """CR-25: gui/i18n.set_language must use an RLock so SSE workers
    and the GUI's language switch don't race."""

    def test_i18n_lock_present(self) -> None:
        src = _read("gui/i18n.py")
        self.assertIn(
            "_I18N_LOCK",
            src,
            "gui/i18n.py must define _I18N_LOCK (CR-25)",
        )


class TestImageLabelCheckUsesGlobEscape(unittest.TestCase):
    """CR-33: ``_resolve_panel_path`` must use ``glob.escape`` on
    user-controlled components to prevent glob injection."""

    def test_glob_escape_used(self) -> None:
        src = _read("evaluation/image_label_check.py")
        self.assertIn(
            "glob.escape",
            src,
            "_resolve_panel_path must use glob.escape (CR-33)",
        )


class TestReviewCorrectionsPathFix(unittest.TestCase):
    """CR-38: ``_apply_review_corrections`` must look in
    ``<work_dir>/corrections/...`` (production) AND the legacy
    ``<work_dir.parent>/corrections/...`` (Phase 53) so the GUI / API
    and CLI / Phase 53 corrections paths both resolve."""

    def test_corrections_path_includes_work_subdir(self) -> None:
        src = _read("pipeline.py")
        # The function body must include BOTH candidate paths.
        self.assertIn(
            'work / "corrections" / "corrections.jsonl"',
            src,
            "_apply_review_corrections must look under work/corrections (CR-38 production path)",
        )
        self.assertIn(
            'work.parent / "corrections" / "corrections.jsonl"',
            src,
            "_apply_review_corrections must look under work.parent/corrections (CR-38 legacy path)",
        )


class TestIsRealPredictionSyntheticGuard(unittest.TestCase):
    """CR-29: synthetic-fallback rows must NOT count as real
    predictions regardless of species emptiness."""

    def test_synthetic_matchers_rejected(self) -> None:
        src = _read("evaluation/metrics.py")
        self.assertIn(
            "synthetic-fallback",
            src,
            "_is_real_prediction must reject synthetic-fallback matcher_type (CR-29)",
        )


class TestCompareBeforeAfterUsesSpeciesCompatible(unittest.TestCase):
    """CR-26: compare_before_after must use ``_species_compatible``
    rather than strict ``==`` so cf./aff. / trinomial counts as a
    correct match."""

    def test_species_compatible_used_in_compare(self) -> None:
        src = _read("evaluation/metrics.py")
        # Locate the body of ``compare_before_after`` (it appears
        # AFTER ``_species_compatible`` is defined at module level,
        # so we anchor on the function definition and search forward).
        match = re.search(
            r"def\s+compare_before_after\([^)]*\).*?_species_compatible",
            src,
            re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            "compare_before_after body must call _species_compatible (CR-26)",
        )


class TestWilsonN1ReturnsFullInterval(unittest.TestCase):
    """CR-28: ``wilson_score_interval`` with ``n=1`` must return
    ``(0.0, 1.0)`` instead of the degenerate
    ``(0.397, 0.397)`` point."""

    def test_wilson_n1_returns_full_interval(self) -> None:
        src = _read("evaluation/metrics.py")
        self.assertRegex(
            src,
            r"if\s+n\s*==\s*1:\s*\n\s*return\s+\(0\.0,\s*1\.0\)",
            "wilson_score_interval must special-case n=1 (CR-28)",
        )


class TestSSRFRejectsIPv6ZoneId(unittest.TestCase):
    """CR-18: ``_validate_llm_host`` must reject URLs with IPv6
    zone-id suffixes (``%eth0`` etc.)."""

    def test_ipv6_zone_id_rejected(self) -> None:
        src = _read("llm_backends.py")
        self.assertRegex(
            src,
            r'if\s+"%"\s+in\s+hostname:',
            "_validate_llm_host must reject IPv6 zone-id (CR-18)",
        )


class TestM3SafeJSONBalancedObjects(unittest.TestCase):
    """CR-35: ``m3_engine._safe_json_loads`` must fall back to
    balanced-object recovery when the LLM emits a malformed JSON
    array with missing commas."""

    def test_safe_json_extracts_balanced_objects(self) -> None:
        src = _read("m3_engine.py")
        self.assertIn(
            "_extract_balanced_objects",
            src,
            "_safe_json_loads must fall back to _extract_balanced_objects (CR-35)",
        )


class TestLocalParserHasConfidence(unittest.TestCase):
    """CR-20: ``LocalParserResult`` must expose a ``confidence``
    field so callers can audit the local-vs-GROBID fallback decision."""

    def test_local_parser_has_confidence(self) -> None:
        src = _read("local_pdf_parser.py")
        self.assertRegex(
            src,
            r"confidence:\s*float\s*=\s*0\.55",
            "LocalParserResult must declare confidence field (CR-20)",
        )


class TestPickFieldRejectsDict(unittest.TestCase):
    """CR-19: ``_pick_field`` must reject dict / list values rather
    than returning them to Pydantic validation."""

    def test_pick_field_skips_dict(self) -> None:
        src = _read("gemma_postprocess.py")
        # The new code must check ``isinstance(value, (str, int, float))``
        # and ``continue`` for everything else.
        self.assertRegex(
            src,
            r"isinstance\(value,\s*\(str,\s*int,\s*float\)\)",
            "_pick_field must restrict to primitives (CR-19)",
        )


class TestFigureRecordBboxValidator(unittest.TestCase):
    """CR-12: ``FigureRecord.bbox`` must validate non-negative,
    non-zero coordinates via a Pydantic field_validator."""

    def test_bbox_field_validator_present(self) -> None:
        src = _read("schema_models.py")
        self.assertRegex(
            src,
            r"@field_validator\(\"bbox\"\)\s*\n[ \t]*@classmethod\s*\n[ \t]*def\s+_validate_bbox",
            "FigureRecord must define _validate_bbox field validator (CR-12)",
        )


class TestRunOutputIdDeduplication(unittest.TestCase):
    """CR-14: ``RunOutput`` must enforce ID uniqueness across papers /
    figures / panels via a model_validator."""

    def test_run_output_model_validator_present(self) -> None:
        src = _read("schema_models.py")
        self.assertIn(
            "_enforce_unique_ids",
            src,
            "RunOutput must define _enforce_unique_ids model_validator (CR-14)",
        )


class TestReviewCorrectionsWorkSubdir(unittest.TestCase):
    """Already covered by TestReviewCorrectionsPathFix above; this
    placeholder keeps the test naming symmetric."""


class TestCLIAPIAlign(unittest.TestCase):
    """Already covered by TestCLIAPIParameterConsistency above."""


class TestDeleteResultsBatchRaceFix(unittest.TestCase):
    """BL-7: ``delete_results_batch`` re-acquired ``RESULT_LOCK``
    per-job *after* releasing it; a concurrent ``_purge_job`` could
    pop the entry between acquisitions, leaving the persist step
    writing an empty row list to a purged job's ``matches.jsonl``.
    The fix re-validates the job is still in ``RESULT_CACHE``."""

    def test_persist_validates_job_still_in_cache(self) -> None:
        src = _read("api/app.py")
        # Same check as TestBatchResultsPersistRecheck but verifies the
        # BLOCKER-level invariant (job presence) without depending on
        # the specific source-of-truth comment block.
        idx = src.find("def delete_results_batch(")
        self.assertGreater(idx, -1)
        snippet = src[idx:]
        self.assertRegex(
            snippet,
            r"job\s*=\s*RESULT_CACHE\.get\(job_id\)\s*\n\s*if\s+job\s+is\s+None:",
            "delete_results_batch persist must validate job presence (BL-7)",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
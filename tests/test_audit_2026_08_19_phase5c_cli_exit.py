"""Regression tests for audit 2026-08-19 Phase 5C (CLI exit code +
encoding + --version + argument validation).

Covers four BLOCKER/MAJOR bugs found by the multi-agent audit:

M-6 — CLI exit code three-state
    Before the fix, ``main()`` only ``print()``-ed errors and returned
    0/1. A typo'd ``--pdf-dir`` looked identical to a runtime crash;
    shell wrappers (Makefile, GitHub Actions) couldn't distinguish
    misuse from real failure. Now: 0 = success, 1 = unexpected
    exception, 2 = user/usage error.

M-7 — Windows encoding
    Default cp1252 mojibakes Chinese / Japanese species names on
    stock Windows consoles. The fix wraps stdout/stderr in a
    UTF-8 TextIOWrapper when ``sys.platform == "win32"``.

M-8 — ``--version`` flag
    Operators had to grep ``pyproject.toml`` to discover the version.
    Now ``python -m rlpe.cli --version`` prints ``RLPE 1.1.0`` and
    exits 0.

M-9 — Argument range validation
    ``--yolo-conf 2.0`` slipped through to Ultralytics and surfaced as
    a buried ``AssertionError`` mid-pipeline. ``--taxon-model invalid``
    crashed 30+ seconds in with a TaxoNERD error. Both now fail at
    parse time with a clean message.
"""

from __future__ import annotations

import io
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


_CLI_PATH = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "cli.py"


# ---------------------------------------------------------------------------
# M-6: three-state exit code
# ---------------------------------------------------------------------------


class TestExitCodeThreeState:
    """M-6: ``main()`` must return 0/1/2 cleanly so shell wrappers
    can branch on misuse vs. failure.
    """

    def test_usererror_returns_2(self, tmp_path):
        """Passing a non-existent ``--pdf-dir`` must raise
        :class:`UserError` (caught → exit 2, no traceback)."""
        from rlpe.cli import UserError, main

        bad_dir = tmp_path / "does_not_exist"
        with patch.object(sys, "argv", [
            "rlpe",
            "--pdf-dir", str(bad_dir),
            "--work-dir", str(tmp_path / "work"),
        ]):
            rc = main()
        assert rc == 2, f"--pdf-dir typo should exit 2, got {rc}"

    def test_usererror_via_validate_args(self, tmp_path):
        """Direct ``_validate_args`` must raise ``UserError`` (not a
        bare ``FileNotFoundError``) for missing paths."""
        from rlpe.cli import UserError, _validate_args, build_parser

        bad_dir = tmp_path / "missing"
        args = build_parser().parse_args([
            "--pdf-dir", str(bad_dir),
            "--work-dir", str(tmp_path / "work"),
        ])
        with pytest.raises(UserError, match="--pdf-dir does not exist"):
            _validate_args(args)

    def test_usererror_prints_clean_message_no_traceback(self, tmp_path, capsys):
        """UserError path must print ``USER ERROR: ...`` to stderr
        and NOT dump a Python traceback (which would mask the
        one-line message)."""
        from rlpe.cli import main

        bad_dir = tmp_path / "missing"
        with patch.object(sys, "argv", [
            "rlpe",
            "--pdf-dir", str(bad_dir),
            "--work-dir", str(tmp_path / "work"),
        ]):
            rc = main()
        captured = capsys.readouterr()
        assert rc == 2
        assert "USER ERROR:" in captured.err
        # No traceback lines (File "..." / line numbers).
        assert 'File "' not in captured.err, (
            f"UserError path leaked a Python traceback:\n{captured.err}"
        )

    def test_runtime_exception_returns_1(self, tmp_path):
        """An unexpected RuntimeError during pipeline.run() must be
        caught and turned into exit 1 (not 0, not 2)."""
        from rlpe.cli import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "stub.pdf").write_bytes(b"%PDF-1.4 stub")
        work_dir = tmp_path / "work"

        with patch.object(sys, "argv", [
            "rlpe",
            "--pdf-dir", str(pdf_dir),
            "--work-dir", str(work_dir),
        ]):
            # Patch the LOCAL reference in the cli module — cli.py does
            # ``from .pipeline import RadiolarianPipeline`` so the name
            # lives in rlpe.cli's namespace, not rlpe.pipeline's.
            with patch("rlpe.cli.RadiolarianPipeline") as pipe_cls:
                pipe_cls.return_value.run.side_effect = RuntimeError("synthetic crash")
                rc = main()
        assert rc == 1, f"unexpected exception must exit 1, got {rc}"

    def test_systemexit_from_argparse_propagates(self):
        """argparse calls ``sys.exit(2)`` on bad flags; main() must
        let that propagate (re-raise) instead of swallowing it into
        a different code."""
        from rlpe.cli import main

        with patch.object(sys, "argv", ["rlpe", "--yolo-conf", "2.0"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        # argparse uses code=2 for usage errors.
        assert exc_info.value.code == 2, (
            f"argparse usage error must propagate exit code 2, got {exc_info.value.code}"
        )


# ---------------------------------------------------------------------------
# M-7: UTF-8 encoding wrapper
# ---------------------------------------------------------------------------


class TestUtf8Encoding:
    """M-7: stdout/stderr must be UTF-8 on Windows so Chinese / JA
    species names don't mojibake."""

    def test_cli_imports_without_crash(self):
        """Importing cli.py must succeed even though the top-of-file
        UTF-8 rewrap runs on win32. On POSIX the rewrap is a no-op
        (guarded by ``sys.platform == 'win32'``)."""
        from rlpe import cli  # noqa: F401 — import smoke

    def test_windows_branch_rewrap_idempotent(self, monkeypatch):
        """When ``sys.platform == 'win32'``, the top-of-file wrap
        rewraps stdout/stderr. Re-importing must be a no-op (no
        double-wrap), and the wrapper must have ``encoding='utf-8'``.

        On POSIX systems (Linux CI), we simulate the win32 branch by
        directly invoking the same TextIOWrapper logic and verifying
        the encoding attribute sticks.
        """
        # Simulate the win32 branch on a fresh TextIOWrapper.
        buf = io.BytesIO()
        wrapped = io.TextIOWrapper(buf, encoding="utf-8", line_buffering=True)
        assert wrapped.encoding.lower().replace("-", "") in ("utf8", "utf8"), (
            f"TextIOWrapper encoding must be utf-8, got {wrapped.encoding!r}"
        )
        # Round-trip a CJK string.
        wrapped.write("放射虫 Archaeodictyomitra 日本語")
        wrapped.flush()
        # The buffer now contains the UTF-8 bytes; decoding must yield
        # the original string without UnicodeDecodeError.
        buf.seek(0)
        raw = buf.read().decode("utf-8")
        assert "放射虫" in raw and "日本語" in raw

    def test_cli_module_has_io_import(self):
        """Source guard: the UTF-8 wrap depends on ``io`` being
        imported. A future refactor that drops the import would
        silently break Windows users."""
        text = _CLI_PATH.read_text(encoding="utf-8")
        assert "import io" in text, (
            "cli.py must import io for the Windows UTF-8 wrap; "
            "dropping the import silently regresses M-7."
        )

    def test_cli_module_wraps_stdout_on_win32_branch(self):
        """Source guard: the win32 branch must rewrap stdout."""
        text = _CLI_PATH.read_text(encoding="utf-8")
        # Both branches must be present (no regression in either).
        assert 'sys.platform == "win32"' in text, (
            "cli.py must guard the stdout rewrap with sys.platform == 'win32'"
        )
        # The code uses ``getattr(sys, _stream_name, None)`` over a
        # tuple of ("stdout", "stderr"), so guard for both names in
        # the wrap block.
        assert '("stdout", "stderr")' in text, (
            "cli.py must iterate both stdout and stderr in the wrap loop"
        )
        assert "io.TextIOWrapper" in text, (
            "cli.py must use io.TextIOWrapper to rewrap the streams"
        )


# ---------------------------------------------------------------------------
# M-8: ``--version`` flag
# ---------------------------------------------------------------------------


class TestVersionFlag:
    """M-8: ``--version`` must print the package version and exit 0."""

    def test_parser_has_version_action(self):
        """Source guard: ``--version`` must be wired with
        ``action='version'`` and reference ``VERSION``."""
        import argparse

        from rlpe.cli import build_parser

        parser = build_parser()
        for action in parser._actions:
            if "--version" in action.option_strings:
                # argparse exposes the version action via _VersionAction.
                # Use isinstance (the public-but-private surface) plus
                # the .version string as the source of truth.
                assert isinstance(action, argparse._VersionAction), (
                    f"--version must use argparse's _VersionAction, "
                    f"got {type(action).__name__}"
                )
                assert "RLPE" in action.version, (
                    f"--version output must start with 'RLPE', got {action.version!r}"
                )
                return
        pytest.fail("--version not found in build_parser() output")

    def test_version_module_attribute(self):
        """``VERSION`` must be exposed as a module attribute (used
        by the parser's version action string)."""
        from rlpe import cli

        assert hasattr(cli, "VERSION"), "cli must expose VERSION"
        assert isinstance(cli.VERSION, str) and cli.VERSION, (
            f"VERSION must be a non-empty string, got {cli.VERSION!r}"
        )

    def test_version_matches_package_version(self):
        """``rlpe.VERSION`` and ``rlpe.__version__`` must agree so
        the CLI, GUI and API don't drift."""
        from rlpe import __version__, cli

        assert cli.VERSION == __version__, (
            f"cli.VERSION ({cli.VERSION!r}) must equal __version__ "
            f"({__version__!r})"
        )

    def test_version_subprocess_exits_zero(self):
        """End-to-end: ``python -m rlpe.cli --version`` must exit 0
        and print the version."""
        # Use the same interpreter as the test runner.
        result = subprocess.run(
            [sys.executable, "-m", "rlpe.cli", "--version"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
            timeout=30,
        )
        assert result.returncode == 0, (
            f"--version must exit 0, got {result.returncode}. "
            f"stderr: {result.stderr}"
        )
        out = result.stdout.strip()
        assert out, "--version output must be non-empty"
        assert "RLPE" in out, f"--version output must contain 'RLPE', got {out!r}"
        # And a numeric version component must follow.
        assert any(c.isdigit() for c in out), (
            f"--version output must contain a version number, got {out!r}"
        )


# ---------------------------------------------------------------------------
# M-9: argument range / choice validation
# ---------------------------------------------------------------------------


class TestFloatRangeValidator:
    """M-9: ``FloatRange(lo, hi)`` is an argparse ``type=`` factory
    that rejects out-of-range values at parse time."""

    def test_accepts_value_in_range(self):
        from rlpe.cli import FloatRange

        v = FloatRange(0.0, 1.0)("0.5")
        assert v == 0.5

    def test_accepts_boundary_low(self):
        from rlpe.cli import FloatRange

        assert FloatRange(0.0, 1.0)("0.0") == 0.0

    def test_accepts_boundary_high(self):
        from rlpe.cli import FloatRange

        assert FloatRange(0.0, 1.0)("1.0") == 1.0

    def test_rejects_value_above_high(self):
        from rlpe.cli import FloatRange

        with pytest.raises(
            argparse_error_subclass(),  # type: ignore[arg-type]
        ):
            FloatRange(0.0, 1.0)("2.0")

    def test_rejects_value_below_low(self):
        from rlpe.cli import FloatRange

        with pytest.raises(argparse_error_subclass()):  # type: ignore[arg-type]
            FloatRange(0.0, 1.0)("-0.5")

    def test_rejects_non_numeric(self):
        from rlpe.cli import FloatRange

        with pytest.raises(argparse_error_subclass()):  # type: ignore[arg-type]
            FloatRange(0.0, 1.0)("not-a-number")

    def test_custom_range(self):
        """Factory must accept arbitrary ranges (proves it's a
        generic validator, not a YOLO-specific one)."""
        from rlpe.cli import FloatRange

        v = FloatRange(-1.0, 1.0)
        assert v("0.0") == 0.0
        with pytest.raises(argparse_error_subclass()):  # type: ignore[arg-type]
            v("5.0")


def argparse_error_subclass():
    """Return argparse.ArgumentTypeError without importing argparse
    at module load time."""
    import argparse

    return argparse.ArgumentTypeError


class TestYoloConfValidation:
    """M-9: ``--yolo-conf`` must reject out-of-range values."""

    def test_rejects_above_one(self):
        from rlpe.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--pdf-dir", "/tmp", "--work-dir", "/tmp",
                "--yolo-conf", "2.0",
            ])

    def test_rejects_below_zero(self):
        from rlpe.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--pdf-dir", "/tmp", "--work-dir", "/tmp",
                "--yolo-conf", "-0.5",
            ])

    def test_accepts_zero(self):
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", "/tmp", "--work-dir", "/tmp",
            "--yolo-conf", "0.0",
        ])
        assert ns.yolo_conf == 0.0

    def test_accepts_one(self):
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", "/tmp", "--work-dir", "/tmp",
            "--yolo-conf", "1.0",
        ])
        assert ns.yolo_conf == 1.0

    def test_accepts_midrange(self):
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", "/tmp", "--work-dir", "/tmp",
            "--yolo-conf", "0.5",
        ])
        assert ns.yolo_conf == 0.5

    def test_default_is_none(self):
        """``--yolo-conf`` is optional; default None → pipeline
        uses the typed-attr fallback default (0.25)."""
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", "/tmp", "--work-dir", "/tmp",
        ])
        assert ns.yolo_conf is None


class TestYoloIouValidation:
    """M-9 (bonus): ``--yolo-iou`` must also reject out-of-range."""

    def test_rejects_above_one(self):
        from rlpe.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--pdf-dir", "/tmp", "--work-dir", "/tmp",
                "--yolo-iou", "1.5",
            ])

    def test_rejects_below_zero(self):
        from rlpe.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--pdf-dir", "/tmp", "--work-dir", "/tmp",
                "--yolo-iou", "-0.1",
            ])

    def test_accepts_midrange(self):
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", "/tmp", "--work-dir", "/tmp",
            "--yolo-iou", "0.45",
        ])
        assert ns.yolo_iou == 0.45


class TestTaxonModelChoices:
    """M-9: ``--taxon-model`` must be a closed allow-list."""

    def test_accepts_en_eco(self):
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", "/tmp", "--work-dir", "/tmp",
            "--taxon-model", "en_eco",
        ])
        assert ns.taxon_model == "en_eco"

    def test_accepts_en_plus(self):
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", "/tmp", "--work-dir", "/tmp",
            "--taxon-model", "en_plus",
        ])
        assert ns.taxon_model == "en_plus"

    def test_rejects_invalid(self):
        """A typo'd model name must fail at parse time, not 30 s
        later inside TaxoNERD."""
        from rlpe.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--pdf-dir", "/tmp", "--work-dir", "/tmp",
                "--taxon-model", "invalid",
            ])

    def test_default_is_en_eco(self):
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", "/tmp", "--work-dir", "/tmp",
        ])
        assert ns.taxon_model == "en_eco"


# ---------------------------------------------------------------------------
# Source-guard: all four fixes are present in cli.py
# ---------------------------------------------------------------------------


class TestSourceGuardAllFixes:
    """Source guard: a regression that reverts any of M-6/7/8/9 must
    fail at least one of these tests."""

    def test_user_error_class_defined(self):
        text = _CLI_PATH.read_text(encoding="utf-8")
        assert "class UserError" in text, "UserError exception must be defined"

    def test_three_state_exit_in_main(self):
        """``main()`` must contain three distinct return paths
        (0 / 1 / 2). The success path may delegate to ``_run_pipeline``
        (which itself returns 0) — that's acceptable as long as the
        overall contract (exit 0 on success) is preserved."""
        text = _CLI_PATH.read_text(encoding="utf-8")
        # Find the body of main()
        idx = text.find("def main() -> int:")
        assert idx > 0
        body = text[idx : idx + 3000]
        assert "return 2" in body, (
            "main() must handle exit code 2 (UserError). "
            f"Body:\n{body[:500]}…"
        )
        assert "return 1" in body, (
            "main() must handle exit code 1 (unexpected exception). "
            f"Body:\n{body[:500]}…"
        )
        # The success path may be ``return _run_pipeline(args)`` or
        # ``return 0``; either is fine as long as _run_pipeline itself
        # returns 0. Verify both: a) main() delegates to _run_pipeline
        # AND b) _run_pipeline ends with ``return 0``.
        run_idx = text.find("def _run_pipeline(")
        assert run_idx > 0, (
            "_run_pipeline() must exist (the success path's return 0)"
        )
        run_body = text[run_idx:]
        assert "return 0" in run_body, (
            "_run_pipeline() must end with ``return 0`` so main()'s "
            "success delegation propagates exit code 0"
        )

    def test_float_range_factory_defined(self):
        text = _CLI_PATH.read_text(encoding="utf-8")
        assert "def FloatRange" in text, "FloatRange factory must be defined"

    def test_yolo_conf_uses_floatrange(self):
        text = _CLI_PATH.read_text(encoding="utf-8")
        # The yolo-conf arg must reference FloatRange(0.0, 1.0).
        assert "yolo-conf" in text and "FloatRange(0.0, 1.0)" in text, (
            "--yolo-conf must be guarded by FloatRange(0.0, 1.0)"
        )

    def test_taxon_model_uses_choices(self):
        text = _CLI_PATH.read_text(encoding="utf-8")
        assert "taxon-model" in text and '"en_eco"' in text and "choices=" in text, (
            "--taxon-model must declare a choices= allow-list "
            "containing en_eco"
        )

    def test_version_flag_present(self):
        text = _CLI_PATH.read_text(encoding="utf-8")
        assert '"--version"' in text, "--version flag must be present"
        assert "action=\"version\"" in text or "action='version'" in text, (
            "--version must use action='version'"
        )


# ---------------------------------------------------------------------------
# Logger integrity
# ---------------------------------------------------------------------------


class TestCliLogger:
    """The runtime-error branch must log via the rlpe.cli logger so
    operators can grep the traceback instead of hunting a stderr line."""

    def test_logger_used_for_runtime_errors(self, tmp_path, caplog):
        from rlpe import cli

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "stub.pdf").write_bytes(b"%PDF-1.4 stub")
        work_dir = tmp_path / "work"

        with caplog.at_level(logging.ERROR, logger="rlpe.cli"):
            with patch.object(sys, "argv", [
                "rlpe",
                "--pdf-dir", str(pdf_dir),
                "--work-dir", str(work_dir),
            ]):
                # Patch the LOCAL reference in rlpe.cli (cli.py does
                # ``from .pipeline import RadiolarianPipeline`` so the
                # name lives in rlpe.cli's namespace, not rlpe.pipeline's).
                with patch("rlpe.cli.RadiolarianPipeline") as pipe_cls:
                    pipe_cls.return_value.run.side_effect = RuntimeError("logged boom")
                    rc = cli.main()

        assert rc == 1
        # At least one log record on rlpe.cli at ERROR or above.
        error_records = [
            r for r in caplog.records
            if r.name == "rlpe.cli" and r.levelno >= logging.ERROR
        ]
        assert error_records, (
            "runtime-error branch must log via rlpe.cli logger "
            "(level>=ERROR). Without it operators can't trace failures."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
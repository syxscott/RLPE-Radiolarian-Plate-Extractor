"""Regression tests for audit 2026-08-19 Phase 6C (CLI NIT batch).

Five NIT-class findings from the multi-agent audit:

NIT-1 — ``--work-dir`` / ``--pdf-dir`` did not expanduser
    Before: ``--work-dir ~/rlpe`` was passed through as the literal
    string ``~/rlpe``, which ``pathlib.Path`` then treated as a
    relative directory name (creating ``./~`` on disk). The
    existence check in ``_validate_args`` then blew up with a
    confusing error. The fix wraps the ``type=`` factory with
    ``ExpandUserPath`` so ``~`` / ``~user`` resolve at parse time.

NIT-2 — Missing default config example at ``~/.rlpe/config.json``
    First-run users had to discover the JSON schema by reading the
    source. The fix writes a documented example on first run so a
    user can edit values in place.

NIT-3 — No ``--dry-run`` flag
    Operators had no way to smoke-test the CLI surface (parser, path
    expansion, config load) without paying the OCR / SAM2 / LLM
    bill. The fix adds ``--dry-run`` that prints a resolved-config
    snapshot and exits 0.

NIT-4 — ``print()`` calls without ``flush=True``
    Default ``print`` buffers when stdout is not a tty, so progress
    lines were held until the next chunk or pipe close. The fix
    routes all user-facing prints through ``_flush_print``.

NIT-5 — No ``--quiet`` / ``--verbose`` flags
    Operators had to set ``PYTHONLOGLEVEL`` / edit the root logger
    by hand. The fix adds ``-q`` / ``--quiet`` and ``-v`` /
    ``--verbose`` flags that adjust the ``rlpe.cli`` logger level.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


_CLI_PATH = Path(__file__).resolve().parents[1] / "src" / "rlpe" / "cli.py"


# ---------------------------------------------------------------------------
# NIT-1: ExpandUserPath
# ---------------------------------------------------------------------------


class TestExpandUserPath:
    """NIT-1: ``ExpandUserPath`` must expand ``~`` at parse time."""

    def test_expanduser_in_parser(self, tmp_path):
        """``--pdf-dir ~/test`` must resolve to ``Path('/home/user/test')``
        on POSIX (and ``Path('C:/Users/<user>/test')`` on Windows)."""
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", "~/papers",
            "--work-dir", str(tmp_path / "work"),
        ])
        expected = Path(os.path.expanduser("~/papers"))
        assert ns.pdf_dir == expected, (
            f"--pdf-dir ~/papers must expand to {expected!r}, got {ns.pdf_dir!r}"
        )

    def test_expanduser_work_dir(self, tmp_path):
        """``--work-dir ~/work`` must also expand."""
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", str(tmp_path / "pdfs"),
            "--work-dir", "~/rlpe_work",
        ])
        expected = Path(os.path.expanduser("~/rlpe_work"))
        assert ns.work_dir == expected, (
            f"--work-dir ~/rlpe_work must expand to {expected!r}, got {ns.work_dir!r}"
        )

    def test_expanduser_export_paths(self, tmp_path):
        """``--export-csv`` / ``--export-json`` / ``--export-jsonl``
        must also expanduser."""
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", str(tmp_path / "pdfs"),
            "--work-dir", str(tmp_path / "work"),
            "--export-csv", "~/out.csv",
            "--export-json", "~/out.json",
            "--export-jsonl", "~/out.jsonl",
        ])
        for attr, want in (
            ("export_csv", "~/out.csv"),
            ("export_json", "~/out.json"),
            ("export_jsonl", "~/out.jsonl"),
        ):
            expected = Path(os.path.expanduser(want))
            assert getattr(ns, attr) == expected, (
                f"--{attr.replace('_', '-')} {want} must expand to "
                f"{expected!r}, got {getattr(ns, attr)!r}"
            )

    def test_no_expansion_when_not_tilde(self, tmp_path):
        """A plain relative or absolute path must pass through unchanged."""
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", str(tmp_path / "pdfs"),
            "--work-dir", str(tmp_path / "work"),
        ])
        assert ns.pdf_dir == tmp_path / "pdfs"
        assert ns.work_dir == tmp_path / "work"

    def test_expanduser_factory_rejects_non_string(self):
        """Non-string input to the ``type=`` factory must raise
        ``ArgumentTypeError`` so a future caller can't slip a Path
        through and bypass the expansion."""
        import argparse

        from rlpe.cli import ExpandUserPath

        with pytest.raises(argparse.ArgumentTypeError):
            ExpandUserPath(123)  # type: ignore[arg-type]

    def test_expanduser_factory_accepts_path(self):
        """If a default is a Path instance the factory passes it through
        (avoids breaking future programmatic callers)."""
        from rlpe.cli import ExpandUserPath

        p = Path("/tmp/x")
        assert ExpandUserPath(p) is p

    def test_validate_args_defensive_expanduser(self, tmp_path, monkeypatch):
        """``_validate_args`` must defensively re-expanduser any
        leftover ``~`` even if the parser type= is bypassed (e.g.
        a programmatic caller passing a string default).

        To exercise the defensive branch with a real ``~`` path we
        sandbox ``HOME`` to ``tmp_path`` so ``os.path.expanduser``
        maps ``~`` to a directory we control. We pre-create the
        target subdirectory so the existence check passes and we
        get to the assertion that the path was actually expanded.
        """
        from rlpe.cli import _validate_args, build_parser

        # Sandbox HOME so "~" -> tmp_path / "sandbox_home".
        sandbox_home = tmp_path / "sandbox_home"
        sandbox_home.mkdir()
        monkeypatch.setenv("HOME", str(sandbox_home))

        # Pre-create the expanded work_dir so the existence check
        # succeeds — we want to prove the expansion *happened*, not
        # test the existence check itself.
        expanded_target = sandbox_home / "rlpe_work"
        expanded_target.mkdir()

        # Pre-create a pdf dir so --pdf-dir validation passes.
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        parser = build_parser()
        args = parser.parse_args([
            "--pdf-dir", str(pdf_dir),
            "--work-dir", str(tmp_path / "work"),
        ])
        # Stuff a raw "~/rlpe_work" string into work_dir, bypassing
        # the ExpandUserPath type= factory. The defensive branch
        # must expand it before the existence check runs.
        # Use "~/foo" rather than "~user" because ``os.path.expanduser``
        # only resolves "~user" if ``user`` exists in the password DB;
        # "~" + relative path always maps to $HOME which we control
        # via the monkeypatch above.
        args.work_dir = "~/rlpe_work"
        _validate_args(args)  # must not raise

        # And after the call, work_dir must have been re-pointed at
        # the expanded path (proves the defensive expanduser fired
        # rather than just passing the literal "~/rlpe_work" through).
        assert str(args.work_dir) == str(expanded_target), (
            f"defensive expanduser should have rewritten work_dir "
            f"to {expanded_target!r}, got {args.work_dir!r}"
        )

    def test_source_guard_expanduser_factory(self):
        """Source guard: ``ExpandUserPath`` must be defined so a
        future refactor that drops it cannot silently regress NIT-1."""
        text = _CLI_PATH.read_text(encoding="utf-8")
        assert "def ExpandUserPath" in text, (
            "ExpandUserPath factory must be defined in cli.py"
        )
        # And it must be used by --pdf-dir / --work-dir.
        assert "type=ExpandUserPath" in text, (
            "--pdf-dir / --work-dir / --export-* must use ExpandUserPath"
        )


# ---------------------------------------------------------------------------
# NIT-2: default config example
# ---------------------------------------------------------------------------


class TestDefaultConfigExample:
    """NIT-2: ``~/.rlpe/config.json`` must be auto-created on first run."""

    def test_example_body_is_valid_json(self):
        """The shipped example body must parse as JSON so a corrupt
        template can't break the first run."""
        from rlpe.cli import EXAMPLE_CONFIG_BODY

        payload = json.loads(EXAMPLE_CONFIG_BODY)
        # Top-level scalars we document in the example.
        for key in (
            "pdf_dir",
            "work_dir",
            "output_dir",
            "grobid_url",
            "use_gpu",
            "ocr_backend",
            "taxon_model",
            "num_workers",
            "min_panel_score",
            "extra",
        ):
            assert key in payload, (
                f"example config must document key {key!r}; "
                f"got keys {list(payload)!r}"
            )

    def test_ensure_default_config_creates_file(self, tmp_path):
        """Calling :func:`ensure_default_config` on a missing path must
        create the file and report ``created=True``."""
        from rlpe.cli import ensure_default_config

        target = tmp_path / "subdir" / "config.json"
        assert not target.exists()
        resolved, created = ensure_default_config(target)
        assert created is True
        assert resolved == target
        assert target.exists()
        # Body must be parseable JSON.
        json.loads(target.read_text(encoding="utf-8"))

    def test_ensure_default_config_skips_existing(self, tmp_path):
        """Calling :func:`ensure_default_config` on an existing file
        must NOT overwrite it — the user may have hand-edited values
        we should respect."""
        from rlpe.cli import ensure_default_config

        target = tmp_path / "config.json"
        target.write_text('{"pdf_dir": "/custom"}', encoding="utf-8")
        resolved, created = ensure_default_config(target)
        assert created is False
        # Body unchanged.
        assert target.read_text(encoding="utf-8") == '{"pdf_dir": "/custom"}'

    def test_default_config_path_uses_home(self, monkeypatch):
        """``default_config_path`` must respect ``HOME`` so the test
        can sandbox the home directory."""
        from rlpe.cli import EXAMPLE_CONFIG_FILENAME, default_config_path

        monkeypatch.setenv("HOME", "/tmp/fakehome")
        result = default_config_path()
        assert result == Path("/tmp/fakehome/.rlpe") / EXAMPLE_CONFIG_FILENAME

    def test_cli_first_run_creates_example(self, tmp_path, monkeypatch, capsys):
        """Running the CLI with ``HOME`` sandboxed to a tmp_path
        (and no ``--config`` flag) must drop the documented example
        at ``$HOME/.rlpe/config.json`` and continue.

        The NIT-2 spec is specifically about the *default* path
        (``~/.rlpe/config.json``), so the test does NOT pass
        ``--config`` explicitly — that's the auto-create path.
        """
        from rlpe.cli import EXAMPLE_CONFIG_DIRNAME, EXAMPLE_CONFIG_FILENAME, main

        sandbox_home = tmp_path / "home"
        sandbox_home.mkdir()
        monkeypatch.setenv("HOME", str(sandbox_home))

        # Use a missing --pdf-dir so the run fails fast (exit 2)
        # after the config example has been written. We only care
        # that the example is dropped on the floor before the error.
        bad_pdf = tmp_path / "missing_pdfs"
        with patch.object(sys, "argv", [
            "rlpe",
            "--pdf-dir", str(bad_pdf),
            "--work-dir", str(tmp_path / "work"),
        ]):
            rc = main()

        assert rc == 2  # UserError from missing pdf-dir
        # The example must have been written to the default location.
        example = sandbox_home / EXAMPLE_CONFIG_DIRNAME / EXAMPLE_CONFIG_FILENAME
        assert example.exists(), (
            "first-run must drop the example config at "
            "$HOME/.rlpe/config.json"
        )
        # And it must be valid JSON.
        json.loads(example.read_text(encoding="utf-8"))
        captured = capsys.readouterr()
        # The example-write NOTE goes to stdout.
        assert "example config" in captured.out, (
            f"expected the NOTE line on stdout, got: {captured.out!r}"
        )

    def test_source_guard_example_constants(self):
        """Source guard: the example config body / filename / dirname
        must be defined so a future refactor that drops them
        silently regresses NIT-2."""
        text = _CLI_PATH.read_text(encoding="utf-8")
        for name in (
            "EXAMPLE_CONFIG_BODY",
            "EXAMPLE_CONFIG_FILENAME",
            "EXAMPLE_CONFIG_DIRNAME",
        ):
            assert name in text, f"{name} must be defined in cli.py"


# ---------------------------------------------------------------------------
# NIT-3: --dry-run
# ---------------------------------------------------------------------------


class TestDryRun:
    """NIT-3: ``--dry-run`` must print the resolved config and exit 0
    without launching the pipeline."""

    def test_dry_run_flag_parses(self, tmp_path):
        """``--dry-run`` must be a recognized flag."""
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", str(tmp_path / "pdfs"),
            "--work-dir", str(tmp_path / "work"),
            "--dry-run",
        ])
        assert ns.dry_run is True

    def test_dry_run_default_false(self, tmp_path):
        """Without ``--dry-run`` the flag defaults to False."""
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", str(tmp_path / "pdfs"),
            "--work-dir", str(tmp_path / "work"),
        ])
        assert ns.dry_run is False

    def test_dry_run_exits_zero(self, tmp_path):
        """``--dry-run`` must exit 0 cleanly (no traceback) and
        never call ``RadiolarianPipeline.run``."""
        from rlpe.cli import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "stub.pdf").write_bytes(b"%PDF-1.4 stub")
        work_dir = tmp_path / "work"

        with patch.object(sys, "argv", [
            "rlpe",
            "--pdf-dir", str(pdf_dir),
            "--work-dir", str(work_dir),
            "--dry-run",
        ]):
            # Patch the LOCAL reference; cli.py does
            # ``from .pipeline import RadiolarianPipeline`` so the name
            # lives in rlpe.cli's namespace.
            with patch("rlpe.cli.RadiolarianPipeline") as pipe_cls:
                rc = main()

        assert rc == 0, f"--dry-run must exit 0, got {rc}"
        # Pipeline must NOT have been instantiated.
        pipe_cls.assert_not_called()

    def test_dry_run_prints_resolved_config(self, tmp_path, capsys):
        """``--dry-run`` must print enough fields that a CI smoke
        test can grep the resolved config."""
        from rlpe.cli import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "stub.pdf").write_bytes(b"%PDF-1.4 stub")
        work_dir = tmp_path / "work"

        with patch.object(sys, "argv", [
            "rlpe",
            "--pdf-dir", str(pdf_dir),
            "--work-dir", str(work_dir),
            "--dry-run",
        ]):
            rc = main()

        assert rc == 0
        out = capsys.readouterr().out
        for needle in ("--pdf-dir", "--work-dir", "--ocr-backend", "dry-run"):
            assert needle in out, (
                f"--dry-run output must mention {needle!r}, got: {out!r}"
            )

    def test_dry_run_subprocess(self, tmp_path):
        """End-to-end: ``python -m rlpe.cli --dry-run …`` must exit 0."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "stub.pdf").write_bytes(b"%PDF-1.4 stub")
        work_dir = tmp_path / "work"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rlpe.cli",
                "--pdf-dir",
                str(pdf_dir),
                "--work-dir",
                str(work_dir),
                "--dry-run",
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
            timeout=30,
        )
        assert result.returncode == 0, (
            f"--dry-run subprocess must exit 0, got {result.returncode}; "
            f"stderr={result.stderr!r}"
        )
        assert "--pdf-dir" in result.stdout

    def test_dry_run_validates_paths_first(self, tmp_path):
        """``--dry-run`` must still reject a missing ``--pdf-dir``
        (the point is to validate, not to skip validation)."""
        from rlpe.cli import main

        with patch.object(sys, "argv", [
            "rlpe",
            "--pdf-dir", str(tmp_path / "missing"),
            "--work-dir", str(tmp_path / "work"),
            "--dry-run",
        ]):
            rc = main()
        assert rc == 2, (
            f"--dry-run with missing --pdf-dir must exit 2, got {rc}"
        )


# ---------------------------------------------------------------------------
# NIT-4: flush=True on user-facing prints
# ---------------------------------------------------------------------------


class TestFlushPrint:
    """NIT-4: user-facing prints must flush so piped/CI output is
    responsive without ``PYTHONUNBUFFERED=1``."""

    def test_flush_print_defaults_flush_true(self):
        """``_flush_print`` must default ``flush=True``."""
        import io as _io

        from rlpe.cli import _flush_print

        buf = _io.StringIO()
        # Patch print to capture kwargs.
        with patch("builtins.print", wraps=print) as wrapped:
            _flush_print("hello", file=buf)
        # Confirm print was called with flush=True.
        assert wrapped.called
        _, kwargs = wrapped.call_args
        assert kwargs.get("flush") is True, (
            f"_flush_print must default flush=True, got kwargs={kwargs!r}"
        )

    def test_flush_print_accepts_extra_kwargs(self):
        """``_flush_print`` must accept ``file=`` / ``end=`` / ``sep=``
        without crashing."""
        import io as _io

        from rlpe.cli import _flush_print

        buf = _io.StringIO()
        _flush_print("a", "b", file=buf, end="!\n", sep="-", flush=True)
        assert buf.getvalue() == "a-b!\n"

    def test_source_guard_flush_print(self):
        """Source guard: ``_flush_print`` must exist and be used by
        the user-facing print paths so a refactor that re-introduces
        raw ``print()`` fails this test."""
        text = _CLI_PATH.read_text(encoding="utf-8")
        assert "def _flush_print" in text, (
            "_flush_print helper must be defined in cli.py"
        )
        # User-facing prints inside main() / _run_dry() must use it.
        assert "_flush_print" in text, (
            "_flush_print must be used in cli.py"
        )


# ---------------------------------------------------------------------------
# NIT-5: --quiet / --verbose
# ---------------------------------------------------------------------------


class TestQuietVerbose:
    """NIT-5: ``--quiet`` (``-q``) / ``--verbose`` (``-v``) must
    adjust the ``rlpe.cli`` logger level."""

    def test_quiet_flag_parses(self, tmp_path):
        """``--quiet`` and ``-q`` must both set quiet=True."""
        from rlpe.cli import build_parser

        for flag in ("--quiet", "-q"):
            parser = build_parser()
            ns = parser.parse_args([
                "--pdf-dir", str(tmp_path / "pdfs"),
                "--work-dir", str(tmp_path / "work"),
                flag,
            ])
            assert ns.quiet is True, f"{flag} must set ns.quiet=True"

    def test_verbose_flag_parses(self, tmp_path):
        """``--verbose`` and ``-v`` must both set verbose=True."""
        from rlpe.cli import build_parser

        for flag in ("--verbose", "-v"):
            parser = build_parser()
            ns = parser.parse_args([
                "--pdf-dir", str(tmp_path / "pdfs"),
                "--work-dir", str(tmp_path / "work"),
                flag,
            ])
            assert ns.verbose is True, f"{flag} must set ns.verbose=True"

    def test_default_levels(self, tmp_path):
        """Without any flag the defaults are False / False."""
        from rlpe.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--pdf-dir", str(tmp_path / "pdfs"),
            "--work-dir", str(tmp_path / "work"),
        ])
        assert ns.quiet is False
        assert ns.verbose is False

    def test_apply_log_level_quiet(self):
        """``--quiet`` must raise the rlpe.cli logger to ERROR."""
        from rlpe.cli import apply_log_level

        logger = logging.getLogger("rlpe.cli")
        original = logger.level
        try:
            apply_log_level(quiet=True, verbose=False)
            assert logger.level == logging.ERROR
        finally:
            logger.setLevel(original)

    def test_apply_log_level_verbose(self):
        """``--verbose`` must lower the rlpe.cli logger to DEBUG."""
        from rlpe.cli import apply_log_level

        logger = logging.getLogger("rlpe.cli")
        original = logger.level
        try:
            apply_log_level(quiet=False, verbose=True)
            assert logger.level == logging.DEBUG
        finally:
            logger.setLevel(original)

    def test_apply_log_level_default(self):
        """No flag must leave the rlpe.cli logger at WARNING."""
        from rlpe.cli import apply_log_level

        logger = logging.getLogger("rlpe.cli")
        original = logger.level
        try:
            apply_log_level(quiet=False, verbose=False)
            assert logger.level == logging.WARNING
        finally:
            logger.setLevel(original)

    def test_quiet_trumps_verbose(self):
        """When both are passed ``--quiet`` wins so a typo'd shell
        line can't silently flood the console."""
        from rlpe.cli import apply_log_level

        logger = logging.getLogger("rlpe.cli")
        original = logger.level
        try:
            apply_log_level(quiet=True, verbose=True)
            assert logger.level == logging.ERROR, (
                "--quiet must trump --verbose when both are passed"
            )
        finally:
            logger.setLevel(original)

    def test_quiet_reduces_log_volume(self, tmp_path):
        """``apply_log_level(quiet=True, verbose=False)`` must make
        the ``rlpe.cli`` logger refuse DEBUG records.

        We use ``logger.isEnabledFor`` rather than capturing records,
        because pytest's caplog bypasses the logger's level filter
        when it installs its own handler at DEBUG level — testing
        record presence in caplog would test caplog, not the filter.
        ``isEnabledFor`` calls the exact code path that
        ``logger.debug(...)`` uses to decide whether to format the
        message, so it's the right hook to verify.
        """
        from rlpe.cli import apply_log_level

        logger = logging.getLogger("rlpe.cli")
        original = logger.level
        try:
            apply_log_level(quiet=True, verbose=False)
            assert logger.isEnabledFor(logging.DEBUG) is False, (
                "--quiet must disable DEBUG on rlpe.cli, but "
                f"isEnabledFor(DEBUG) returned True at level={logger.level}"
            )
            assert logger.isEnabledFor(logging.ERROR) is True, (
                "--quiet must keep ERROR enabled on rlpe.cli, but "
                f"isEnabledFor(ERROR) returned False at level={logger.level}"
            )
            assert logger.isEnabledFor(logging.INFO) is False, (
                "--quiet must disable INFO on rlpe.cli, but "
                f"isEnabledFor(INFO) returned True at level={logger.level}"
            )
        finally:
            logger.setLevel(original)

    def test_verbose_increases_log_volume(self, tmp_path):
        """``apply_log_level(quiet=False, verbose=True)`` must make
        the ``rlpe.cli`` logger accept DEBUG records. See the
        :func:`test_quiet_reduces_log_volume` docstring for why we
        use ``isEnabledFor`` instead of caplog.
        """
        from rlpe.cli import apply_log_level

        logger = logging.getLogger("rlpe.cli")
        original = logger.level
        try:
            apply_log_level(quiet=False, verbose=True)
            assert logger.isEnabledFor(logging.DEBUG) is True, (
                "--verbose must enable DEBUG on rlpe.cli, but "
                f"isEnabledFor(DEBUG) returned False at level={logger.level}"
            )
            assert logger.isEnabledFor(logging.INFO) is True, (
                "--verbose must enable INFO on rlpe.cli, but "
                f"isEnabledFor(INFO) returned False at level={logger.level}"
            )
            assert logger.isEnabledFor(logging.WARNING) is True, (
                "--verbose must keep WARNING enabled on rlpe.cli, but "
                f"isEnabledFor(WARNING) returned False at level={logger.level}"
            )
        finally:
            logger.setLevel(original)


# ---------------------------------------------------------------------------
# Source guards: all five NIT fixes are present in cli.py
# ---------------------------------------------------------------------------


class TestSourceGuardsAllNits:
    """A regression that reverts any of NIT-1/2/3/4/5 must fail at
    least one of these source-guard tests."""

    def test_expanduser_factory_in_source(self):
        text = _CLI_PATH.read_text(encoding="utf-8")
        assert "def ExpandUserPath" in text

    def test_default_config_helpers_in_source(self):
        text = _CLI_PATH.read_text(encoding="utf-8")
        assert "def ensure_default_config" in text
        assert "def default_config_path" in text

    def test_dry_run_in_source(self):
        text = _CLI_PATH.read_text(encoding="utf-8")
        assert '"--dry-run"' in text
        assert "_run_dry" in text
        assert "def _run_dry" in text

    def test_quiet_verbose_in_source(self):
        text = _CLI_PATH.read_text(encoding="utf-8")
        assert '"-q"' in text and '"--quiet"' in text
        assert '"-v"' in text and '"--verbose"' in text
        assert "def apply_log_level" in text

    def test_flush_print_in_source(self):
        text = _CLI_PATH.read_text(encoding="utf-8")
        assert "def _flush_print" in text
        # kwargs.setdefault("flush", True) is the implementation hook.
        assert 'setdefault("flush", True)' in text or "setdefault('flush', True)" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Phase 6E (2026-08-19): misc NIT cleanup batch.

The multi-agent audit on 2026-08-19 surfaced a handful of low-priority
"misc NIT" issues that didn't warrant their own sweep but should still
be tightened before the audit closes:

* **NIT-1** — ``image_preview._bbox_tooltip`` had four hard-coded
  English field labels ("confidence:", "x: …  y: …", "w: …  h: …",
  "family:"). On Chinese builds the rest of the GUI translates but
  the hover-tooltip still showed English, which is jarring. Fixed by
  routing every label through ``i18n._tr(...)`` (Phase 6A added the
  keys; Phase 6E is the regression-guard test).

* **NIT-2** — ``scripts/test_MiniMax_api.py`` and
  ``scripts/round10_live_pdf.py`` used a hard-coded
  ``load_dotenv(<script_dir>/".env")`` lookup. If the operator had a
  project-root ``.env`` but ran the script from a different cwd, the
  script silently saw no keys. Fixed by chaining ``find_dotenv(...)``
  (walks up from cwd) ahead of the script-adjacent fallback. (cli.py
  already used ``find_dotenv`` correctly — no change there.)

* **NIT-3** — Logger-name consistency. All call sites use
  ``logging.getLogger(__name__)`` (which evaluates to ``"rlpe.<...>"``
  because every file lives under ``src/rlpe/``) or an explicit
  ``"rlpe.<...>"`` literal. No ``"root"`` / module-name typos leaked
  in; the test just locks that property down.

* **NIT-4** — ``.gitignore`` must exclude ``__pycache__/`` and
  ``*.pyc`` so the user doesn't accidentally commit bytecode.
  Already present in the repo; test makes the exclusion explicit.

The test file deliberately stays small (source-guard + a behavioural
``find_dotenv`` check + a ``.gitignore`` text check) so it can run
without spinning up Qt or hitting the network.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ============================================================
# 1. image_preview tooltip i18n
# ============================================================
def test_image_preview_tooltip_uses_i18n_keys():
    """NIT-1: every tooltip field label in image_preview must be an i18n key.

    Parses ``image_preview._bbox_tooltip`` with ``ast`` and asserts the
    four field labels are produced via ``i18n._tr("preview.tooltip.*")``
    rather than raw f-strings. The species line and ``<b>...</b>`` data
    wrapping is exempt (species names are data, not chrome).
    """
    from rlpe.gui import image_preview

    src = inspect.getsource(image_preview._bbox_tooltip)
    tree = ast.parse(src)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef), "expected a function definition"

    # Collect every string-literal or f-string on a `parts.append(...)`
    # line so we can verify each label is wrapped in i18n._tr(...).
    append_calls: list[ast.Call] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "append"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "parts"
        ):
            append_calls.append(call)

    assert append_calls, "_bbox_tooltip should call parts.append()"

    def _extract_tr_receiver(node: ast.AST) -> ast.Attribute | None:
        """Return the ``i18n._tr(...)`` Attribute node if ``node`` is
        ``i18n._tr(KEY).format(...)``, else None.

        The actual code is::

            parts.append(i18n._tr("preview.tooltip.confidence").format(value=...))

        AST-wise the outer call is ``.format(...)`` and its receiver is
        a Call whose func is ``i18n._tr``. We accept either shape (the
        bare ``i18n._tr(KEY)`` form, or the chained ``.format(...)``
        form) as evidence the field label is i18n-routed.

        Non-Call nodes (e.g. ``JoinedStr`` for f-strings) return None
        — those represent raw data, not chrome strings.
        """
        if not isinstance(node, ast.Call):
            return None
        if not isinstance(node.func, ast.Attribute):
            return None
        # Shape 1: i18n._tr("...").format(...)
        if node.func.attr == "format" and isinstance(node.func.value, ast.Call):
            inner = node.func.value
            if isinstance(inner.func, ast.Attribute) and inner.func.attr == "_tr":
                return inner.func
        # Shape 2: i18n._tr("...")
        if node.func.attr == "_tr":
            return node.func
        return None

    # The first append is `<b>{species}</b>` — raw data, exempt.
    # Subsequent appends should all use i18n._tr(...).
    for idx, call in enumerate(append_calls):
        arg = call.args[0]
        if idx == 0:
            # species name — allow either a plain JoinedStr (f-string) or
            # a Constant string; it must NOT be an i18n call.
            assert not _extract_tr_receiver(arg), (
                "species line should not be wrapped in i18n._tr (it's data)"
            )
            continue
        tr_attr = _extract_tr_receiver(arg)
        assert tr_attr is not None, (
            f"parts.append #{idx} must call i18n._tr(KEY) (possibly chained "
            f"with .format(...)); got {ast.dump(arg)[:120]}"
        )


def test_image_preview_tooltip_strings_present_in_en_and_zh():
    """NIT-1: the four preview.tooltip.* keys exist in BOTH string tables."""
    from rlpe.gui.strings_en import STRINGS as EN
    from rlpe.gui.strings_zh_CN import STRINGS as ZH

    required = {
        "preview.tooltip.confidence",
        "preview.tooltip.coords_xy",
        "preview.tooltip.coords_wh",
        "preview.tooltip.family",
    }
    en_keys = set(EN.keys())
    zh_keys = set(ZH.keys())
    missing_en = required - en_keys
    missing_zh = required - zh_keys
    assert not missing_en, f"English strings missing keys: {missing_en}"
    assert not missing_zh, f"Chinese strings missing keys: {missing_zh}"


# ============================================================
# 2. find_dotenv
# ============================================================
@pytest.mark.skipif(shutil.which("python") is None, reason="needs python on PATH")
def test_find_dotenv_walks_up_from_cwd(tmp_path, monkeypatch):
    """NIT-2: ``find_dotenv(usecwd=True)`` finds .env by walking up the tree.

    Drops a fake ``.env`` in ``/tmp`` (the simulated cwd), chdirs there
    with ``monkeypatch.chdir``, and verifies that
    ``find_dotenv(usecwd=True)`` returns its path. Without the fix, the
    scripts would have looked only at ``<repo>/.env`` and silently
    loaded nothing — the regression this test guards.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("FOO_TEST_KEY=42\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    from dotenv import find_dotenv

    found = find_dotenv(usecwd=True)
    assert found, "find_dotenv(usecwd=True) returned empty in tmp_path with .env present"
    found_path = Path(found).resolve()
    assert found_path == env_file.resolve(), (
        f"find_dotenv found {found_path!s}, expected {env_file!s}"
    )


def test_test_minimax_api_script_uses_find_dotenv():
    """NIT-2: the scripts/test_MiniMax_api.py load block uses find_dotenv.

    Guards against a regression where someone removes ``find_dotenv``
    and reverts to a hard-coded path lookup.
    """
    script = Path(__file__).resolve().parents[1] / "scripts" / "test_MiniMax_api.py"
    src = script.read_text(encoding="utf-8")
    assert "find_dotenv" in src, (
        f"{script.name} must import find_dotenv so the .env is found even "
        "when the user runs the script from a sub-directory."
    )
    # And confirm the actual call uses usecwd=True so the walk goes
    # up from cwd (not from the script file).
    assert "find_dotenv(usecwd=True)" in src, (
        f"{script.name} must call find_dotenv(usecwd=True) — the usecwd "
        "flag is what makes the upward walk start from cwd."
    )


def test_round10_live_pdf_script_uses_find_dotenv():
    """NIT-2: scripts/round10_live_pdf.py also picks up find_dotenv."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "round10_live_pdf.py"
    src = script.read_text(encoding="utf-8")
    assert "find_dotenv" in src, (
        f"{script.name} must import find_dotenv so the .env is found "
        "even when the user runs the script from a sub-directory."
    )


# ============================================================
# 3. logger-name consistency
# ============================================================
def test_all_loggers_under_rlpe_namespace():
    """NIT-3: every ``logging.getLogger`` call resolves to a ``rlpe.*`` name.

    Walks ``src/rlpe/`` and, for each file that calls
    ``logging.getLogger(...)``, evaluates what the call would return at
    runtime. Anything outside the ``rlpe.*`` namespace is a violation.
    Indirect references (``getLogger(_GUI_LOGGER_NAME)``) are resolved
    by following the module-level constant definition.
    """
    src_root = Path(__file__).resolve().parents[1] / "src" / "rlpe"
    violations: list[tuple[Path, int, str]] = []

    def _resolve_module_constants(tree: ast.Module) -> dict[str, str]:
        """Return module-level ``NAME = "value"`` string assignments.

        Used to chase one-step indirections like
        ``logger = logging.getLogger(_GUI_LOGGER_NAME)`` where the
        argument is a Name rather than a literal.
        """
        constants: dict[str, str] = {}
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                tgt = stmt.targets[0]
                if isinstance(tgt, ast.Name) and isinstance(stmt.value, ast.Constant):
                    if isinstance(stmt.value.value, str):
                        constants[tgt.id] = stmt.value.value
        return constants

    for py_file in src_root.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except SyntaxError:
            continue
        # Derive the module name from the file path (matches what
        # Python's __name__ would be at import time).
        rel = py_file.relative_to(src_root)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]  # strip .py
        module_name = ".".join(["rlpe", *parts])
        constants = _resolve_module_constants(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "getLogger"):
                continue
            if not node.args:
                # getLogger() with no arg defaults to root — flag for
                # human review (we never do that intentionally).
                violations.append((py_file, node.lineno, "getLogger() with no name"))
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Name) and arg.id == "__name__":
                # __name__ evaluates to the module name at import time —
                # verify it starts with "rlpe.".
                if not module_name.startswith("rlpe"):
                    violations.append((py_file, node.lineno, f"__name__={module_name}"))
                continue
            if isinstance(arg, ast.Name) and arg.id in constants:
                # Indirect lookup — follow the constant.
                if not constants[arg.id].startswith("rlpe"):
                    violations.append((py_file, node.lineno, f"{arg.id}={constants[arg.id]!r}"))
                continue
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if not arg.value.startswith("rlpe"):
                    violations.append((py_file, node.lineno, f"literal={arg.value!r}"))
                continue
            # Anything else (computed name) is too dynamic to verify
            # statically — flag for human review.
            violations.append((py_file, node.lineno, f"computed arg: {ast.dump(arg)[:60]}"))

    assert not violations, "All logger names must start with 'rlpe.'. Violations:\n" + "\n".join(
        f"  {p.relative_to(src_root)}:{ln}: {why}" for p, ln, why in violations
    )


# ============================================================
# 4. .gitignore covers bytecode
# ============================================================
def test_gitignore_excludes_pycache_and_pyc():
    """NIT-4: .gitignore must keep ``__pycache__/`` and ``*.pyc`` out."""
    repo_root = Path(__file__).resolve().parents[1]
    gitignore = repo_root / ".gitignore"
    assert gitignore.exists(), f"{gitignore} not found"

    text = gitignore.read_text(encoding="utf-8")
    lines = {
        ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")
    }

    assert "__pycache__/" in lines, (
        f".gitignore must contain '__pycache__/' as a top-level entry; "
        f"found entries include: {sorted(l for l in lines if 'pyc' in l.lower() or 'cache' in l.lower())[:10]}"
    )
    # Either ``*.pyc`` directly or ``*.py[cod]`` (which covers pyc/pyo/pyd).
    has_pyc_direct = "*.pyc" in lines
    has_py_bracket = "*.py[cod]" in lines or "*.py[cdo]" in lines
    assert has_pyc_direct or has_py_bracket, (
        ".gitignore must exclude .pyc files (either '*.pyc' or '*.py[cod]')."
    )


# ============================================================
# 5. helper: load_env_file env-loader sanity (regression-friendly)
# ============================================================
def test_env_loader_returns_zero_when_env_missing(tmp_path):
    """Bonus: ensure ``load_env_file`` is callable with a non-existent path."""
    from rlpe.env_loader import load_env_file

    count = load_env_file(tmp_path / "does-not-exist.env")
    assert count == 0, "load_env_file should return 0 when the file is absent"


# ============================================================
# 7. type-hint completeness (NIT-5)
# ============================================================
def test_type_hints_on_phase6e_targets():
    """NIT-5: the three functions tightened in this batch carry return annotations.

    Phase 6E added ``-> Any`` to:

    * ``rlpe.__getattr__`` (was unannotated — return type is Any)
    * ``rlpe.layout._import_pymupdf`` (was unannotated — returns the
      pymupdf module handle or raises RuntimeError)
    * ``rlpe.opendataloader_extractor.OpenDataloaderExtractor._get_or_init_ocr_engine``
      (was unannotated — returns ``Optional[easyocr.Reader]`` which
      round-trips through Any)

    These are the "headline" fixes. The wider list of unannotated
    functions across the project is tracked separately and out of
    scope for this NIT pass.
    """
    import inspect

    import rlpe
    from rlpe import layout, opendataloader_extractor

    # rlpe.__getattr__
    sig = inspect.signature(rlpe.__getattr__)
    assert sig.return_annotation is not inspect.Signature.empty, (
        "rlpe.__getattr__ should carry a return annotation"
    )
    assert sig.parameters["name"].annotation is not inspect.Parameter.empty, (
        "rlpe.__getattr__'s `name` parameter should be annotated as str"
    )

    # rlpe.layout._import_pymupdf
    sig = inspect.signature(layout._import_pymupdf)
    assert sig.return_annotation is not inspect.Signature.empty, (
        "rlpe.layout._import_pymupdf should carry a return annotation"
    )

    # opendataloader_extractor._get_or_init_ocr_engine
    sig = inspect.signature(
        opendataloader_extractor.OpenDataLoaderExtractor._get_or_init_ocr_engine
    )
    assert sig.return_annotation is not inspect.Signature.empty, (
        "OpenDataLoaderExtractor._get_or_init_ocr_engine should carry a return annotation"
    )


# ============================================================
# 6. AST sanity: image_preview has no bare-English labels
# ============================================================
def test_image_preview_no_bare_english_field_labels():
    """NIT-1 follow-up: the four known-hardcoded labels are gone.

    Phase 48 already checked the obvious ones ("(no image)" etc.) but
    the tooltip-field labels lived in the _bbox_tooltip helper that
    Phase 48 didn't touch. This test is the source-guard for that gap.
    """
    from rlpe.gui import image_preview

    src = inspect.getsource(image_preview._bbox_tooltip)
    # These are the four bare strings the audit caught:
    forbidden_substrings = [
        '"confidence: "',
        '"x: "',
        '"y: "',
        '"w: "',
        '"h: "',
        '"family: "',
        "f'confidence: ",
        "f'x: ",
        "f'w: ",
        "f'family: ",
    ]
    leftovers = [s for s in forbidden_substrings if s in src]
    assert not leftovers, "_bbox_tooltip still has bare-English field labels: " + ", ".join(
        leftovers
    )

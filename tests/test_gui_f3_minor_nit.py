"""Phase F-3 (2026-08-20) — MINOR + NIT regression tests.

After the 4 enumeration agents produced their bug lists, this phase
applied the high-impact fixes in 4 per-area commits. The tests below
guard the most likely regressions (re-introducing magic numbers, lost
i18n keys, deleted dead code) so the next sweep doesn't re-do them.

Each test class maps to one fix area:

* ``TestF3RunTab``         — run_tab.py: i18n tooltips + magic 200
* ``TestF3ResultsTab``     — results_tab.py: caption 280 + badge CSS
* ``TestF3JobsTab``        — jobs_tab.py: status bg colours
* ``TestF3SettingsTab``    — settings_tab.py: invalid border colour
* ``TestF3StringsParity``  — en/zh strings parity (no English leak)
* ``TestF3WebSpaMagicNums`` — web/js/app.js: named timeouts
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

# Mirror the convention used by other GUI tests: force the offscreen
# Qt platform plugin so we don't need a display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


# ============================================================
# run_tab.py: i18n tooltips + magic 200
# ============================================================
class TestF3RunTab:
    """F-3 fixes: hardcoded English tooltips → i18n keys, magic 200 → constant."""

    def test_tooltips_use_i18n(self):
        src = _read("src/rlpe/gui/run_tab.py")
        # The two fixed tooltip lines must go through i18n._tr.
        assert 'self._caption_window.setToolTip(\n            i18n._tr("runtab.label.caption_window.tooltip")\n        )' in src
        assert 'self._od_caption_window.setToolTip(\n            i18n._tr("runtab.label.od_caption_window.tooltip")\n        )' in src
        # And the raw English strings must be gone.
        assert 'GROBID caption→page lookup window' not in src or 'i18n._tr(' in src.split('GROBID caption→page lookup window')[0][-200:]

    def test_status_label_constant_exists(self):
        src = _read("src/rlpe/gui/run_tab.py")
        assert "_STATUS_LABEL_MAX_LEN" in src
        assert "_STATUS_LABEL_MAX_LEN = 200" in src
        # The magic 200 in the log_to_statusbar method must reference
        # the constant, not the literal.
        body = re.search(r"def _log_to_statusbar.*?(?=\n    def )", src, re.DOTALL)
        assert body is not None
        assert "[-200:]" not in body.group(0), (
            "magic -200 must be replaced with [-_STATUS_LABEL_MAX_LEN:]"
        )
        assert "[-_STATUS_LABEL_MAX_LEN:]" in body.group(0)


# ============================================================
# results_tab.py: caption snippet 280 + badge CSS extract
# ============================================================
class TestF3ResultsTab:
    """F-3 fixes: caption magic 280 + badge CSS extract + heading colour."""

    def test_caption_snippet_constant_exists(self):
        src = _read("src/rlpe/gui/results_tab.py")
        assert "_CAPTION_SNIPPET_MAX" in src
        assert "_CAPTION_SNIPPET_MAX = 280" in src

    def test_caption_snippet_uses_constant(self):
        src = _read("src/rlpe/gui/results_tab.py")
        body = re.search(r"cap_snippet\[", src)
        assert body is not None, "cap_snippet slice must exist"
        # The literal 280 must no longer appear in the truncation line.
        assert "cap_snippet[:280]" not in src
        assert "cap_snippet[:_CAPTION_SNIPPET_MAX]" in src

    def test_badge_css_extracted(self):
        src = _read("src/rlpe/gui/results_tab.py")
        # The CSS block must live in a module-level constant.
        assert "_DETAIL_BADGE_CSS" in src
        assert "_DETAIL_HEADING_COLOR" in src
        # The inline ``background:#d6e4ff`` literal in the render
        # body must NOT exist anymore — it's now in the constant.
        body = re.search(r"def _render_detail.*?(?=\n    def |\nclass )", src, re.DOTALL)
        assert body is not None
        body_text = body.group(0)
        # The literal must not appear inside the function body (it
        # can still appear in the module-level constant definition).
        # We check: the render_detail method itself doesn't define
        # the CSS inline.
        assert ".badge-info{padding" not in body_text, (
            "_render_detail must use _DETAIL_BADGE_CSS, not inline CSS"
        )

    def test_format_bbox_kept_with_docstring(self):
        # Dead-code helper kept (with explicit docstring) per the
        # agent recommendation; flag if it disappears entirely.
        src = _read("src/rlpe/gui/results_tab.py")
        assert "def _format_bbox(bbox) -> str:" in src


# ============================================================
# jobs_tab.py: status background colours
# ============================================================
class TestF3JobsTab:
    """F-3 fixes: hardcoded QColor dict → module-level constant."""

    def test_status_bg_colors_constant(self):
        src = _read("src/rlpe/gui/jobs_tab.py")
        assert "_STATUS_BG_COLORS" in src
        # All five statuses must be in the map.
        for status in ("STATUS_RUNNING", "STATUS_DONE", "STATUS_FAILED",
                       "STATUS_CANCELLED", "STATUS_QUEUED"):
            assert status in src.split("_STATUS_BG_COLORS")[1].split("\n\n")[0], (
                f"{status} missing from _STATUS_BG_COLORS"
            )

    def test_qcolor_import_at_module_level(self):
        src = _read("src/rlpe/gui/jobs_tab.py")
        # QColor must be imported at module scope (not inside a method).
        # Read the first 60 lines (the import block + a small buffer)
        # — the test doesn't assume exact paragraph boundaries.
        head = "\n".join(src.splitlines()[:60])
        assert "from PySide6.QtGui import" in head
        import_line = [l for l in head.splitlines() if "from PySide6.QtGui import" in l]
        assert import_line, "QtGui import must be at module scope"
        assert "QColor" in import_line[0]


# ============================================================
# settings_tab.py: invalid border colour
# ============================================================
class TestF3SettingsTab:
    """F-3 fixes: invalid-border hex + log path."""

    def test_invalid_border_color_constant(self):
        src = _read("src/rlpe/gui/settings_tab.py")
        assert "_INVALID_BORDER_COLOR" in src
        # The QSS template must interpolate the constant (no raw
        # hex inside the template literal). The template is multi-line
        # so we look for the f-string in the assignment block.
        qss_block = re.search(
            r"_INVALID_BORDER_QSS\s*=\s*\(([\s\S]*?)\)", src
        )
        assert qss_block is not None, "_INVALID_BORDER_QSS multi-line def must exist"
        block = qss_block.group(1)
        assert "#dc3545" not in block, (
            "_INVALID_BORDER_QSS must reference the constant, not hex"
        )
        assert "_INVALID_BORDER_COLOR" in block


# ============================================================
# strings en/zh parity
# ============================================================
class TestF3StringsParity:
    """F-3 fixes: every new en key must have a zh_CN counterpart."""

    @pytest.mark.parametrize(
        "key",
        [
            "runtab.label.caption_window.tooltip",
            "runtab.label.od_caption_window.tooltip",
        ],
    )
    def test_new_tooltip_key_in_both_languages(self, key):
        from rlpe.gui import strings_en, strings_zh_CN

        assert key in strings_en.STRINGS, f"{key} missing from strings_en"
        assert key in strings_zh_CN.STRINGS, f"{key} missing from strings_zh_CN"


# ============================================================
# Web SPA: magic timeouts (sample)
# ============================================================
class TestF3WebSpaMagicNums:
    """F-3 spot-checks: timeouts that should be named constants."""

    def test_notification_timeout_named(self):
        # The 3000ms notification hide should be a named constant.
        # We don't require a strict name; the source must NOT have
        # a bare ``3000`` literal where it used to.
        js = _read("web/js/app.js")
        # Magic 3000 in notification context must be eliminated.
        # We allow 3000 to exist elsewhere (e.g. fetchWithTimeout).
        # Look at the ``showNotification`` function body.
        m = re.search(r"function\s+showNotification\s*\([^)]*\)\s*\{([\s\S]*?)\n\}", js)
        assert m is not None, "showNotification function must exist"
        body = m.group(1)
        assert "3000" not in body, (
            "showNotification should use _NOTIFICATION_HIDE_MS, not 3000"
        )
        # And the named constant must be defined.
        assert "_NOTIFICATION_HIDE_MS = 3000" in js

    def test_auto_switch_and_revoke_constants(self):
        js = _read("web/js/app.js")
        assert "_AUTO_SWITCH_GRACE_MS = 1200" in js
        assert "_BLOB_REVOKE_DELAY_MS = 1000" in js
        # Both call sites must reference the constants, not literals.
        m = re.search(r"_autoSwitchTimer\s*=\s*setTimeout\([^,]+,\s*(\d+)\)", js)
        assert m is None, (
            f"_autoSwitchTimer must use _AUTO_SWITCH_GRACE_MS, not literal {m.group(1) if m else '?'}"
        )
        m2 = re.search(r"revokeObjectURL\([^)]+\)\s*,\s*\d+\)", js)
        assert m2 is None, (
            "revokeObjectURL timeout must use _BLOB_REVOKE_DELAY_MS, not literal"
        )

    def test_no_console_info_in_production(self):
        # Phase F-3 NIT: two console.info calls inside showMiniMaxFallbackModal
        # were removed; a stray debug leak should fail this test.
        js = _read("web/js/app.js")
        # The showMiniMaxFallbackModal function must not invoke
        # console.info(...). Comments referencing it are fine.
        m = re.search(r"function\s+showMiniMaxFallbackModal[\s\S]*?\n\}", js)
        assert m is not None
        body = m.group(0)
        assert "console.info(" not in body, (
            "showMiniMaxFallbackModal should not call console.info(...) "
            "in production"
        )
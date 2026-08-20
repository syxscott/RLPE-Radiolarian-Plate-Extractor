"""Lightweight i18n for the RLPE GUI.

We avoid Qt Linguist's .ts / .qm toolchain and use flat Python
dicts. Two languages supported:
  * en    - English (default + fallback)
  * zh_CN - Simplified Chinese

Architecture
------------
* ``STRINGS`` is a flat dict loaded from ``strings_en.STRINGS`` (always
  present) + optionally ``strings_zh_CN.STRINGS`` (if the file
  imports successfully). New languages are added by dropping in a
  new module + extending ``available_languages``.
* ``_tr(key, default=None)`` returns the current-language text,
  falling back to English if missing, then ``default``, then a
  ``⟦key⟧`` sentinel.
* ``set_language(lang)`` switches the active code + notifies
  listeners. The main window registers itself as a listener and
  refreshes every translatable widget.
* ``register_widget_text(obj, attr, key)`` binds a translation key
  to ``obj.attr``. After language changes, ``_apply_registry()``
  walks all registered widgets and updates the attribute.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QApplication

_CURRENT_LANG: str = "zh_CN"
_LISTENERS: list[Callable[[str], None]] = []


def current_language() -> str:
    return _CURRENT_LANG


def available_languages() -> list[tuple[str, str]]:
    return [
        ("en", "English"),
        ("zh_CN", "简体中文"),
    ]


def set_language(lang: str) -> None:
    """Switch the active language and notify listeners.

    Unknown codes are ignored. Switching to the current language
    is also a no-op.
    """
    global _CURRENT_LANG
    if lang not in ("en", "zh_CN"):
        return
    if lang == _CURRENT_LANG:
        return
    _CURRENT_LANG = lang
    _apply_registry()
    for fn in list(_LISTENERS):
        try:
            fn(lang)
        except Exception:
            pass
    # Phase 39: re-text menu / toolbar QAction + QMenu objects
    # (they're not QWidgets so the registry's allWidgets() loop
    # misses them). Lazy import to avoid circular dependency with
    # i18n_widgets.
    try:
        from .i18n_widgets import refresh_all_menu_actions

        refresh_all_menu_actions(lang)
    except Exception:
        pass


def add_listener(fn: Callable[[str], None]) -> None:
    # Phase 44: dedupe. Previously re-creating a widget (e.g. after
    # theme rebuild) would append the same listener twice, doubling
    # the work in set_language. Now we check by identity (==) and
    # skip if already registered.
    if fn not in _LISTENERS:
        _LISTENERS.append(fn)


def remove_listener(fn: Callable[[str], None]) -> None:
    if fn in _LISTENERS:
        _LISTENERS.remove(fn)


def _tr(key: str, default: str | None = None) -> str:
    """Return the translated string for ``key``.

    Lookup order:
      1. Current language.
      2. English fallback.
      3. ``default`` argument.
      4. ``⟦key⟧`` sentinel so missing keys are obvious.
    """
    text = STRINGS.get(_CURRENT_LANG, {}).get(key)
    if text is not None:
        return text
    text = STRINGS.get("en", {}).get(key)
    if text is not None:
        return text
    if default is not None:
        return default
    return "⟦" + key + "⟧"


# ============================================================
# Widget text registry
# ============================================================
# Each entry: (objectName, attr_name, translation_key)
_REGISTRY: list[tuple[str, str, str, dict]] = []


def register_widget_text(object_name: str, attr: str, key: str, **fmt) -> None:
    """Bind a translation key to a widget identified by ``objectName``.

    The widget is looked up by ``objectName`` (set this on the
    QWidget via ``setObjectName``). ``attr`` is one of:
      * ``text``          - QLabel/QPushButton/etc. ``setText``
      * ``title``         - QGroupBox ``setTitle``
      * ``windowTitle``   - QMainWindow/QDialog ``setWindowTitle``
      * ``placeholderText`` - QLineEdit/QComboBox ``setPlaceholderText``
      * ``toolTip``       - any widget ``setToolTip``
      * ``statusTip``     - QAction ``setStatusTip``
      * ``tabText``       - QTabWidget ``setTabText`` (uses
                            ``(objectName, tab_index)`` pair,
                            see the helper for the alt signature)

    The text is applied immediately at registration time (so the
    widget displays the right string at first paint) and re-applied
    on every ``set_language`` call.

    Phase 34: ``fmt`` keyword args are substituted into the
    translated text via ``str.format(**fmt)``. Use this for
    templates like ``"⚙️ {app}  ·  v{version}"`` where the placeholders
    are language-independent (English vs Chinese versions both have
    the same ``{app}`` / ``{version}`` slots).
    """
    _REGISTRY.append((object_name, attr, key, fmt))
    # Audit 2026-08-17 (third pass): do NOT call _apply_to_one here.
    # Every tr_label / tr_button / tr_checkbox / tr_groupbox /
    # tr_lineedit / tr_spinbox helper sets the widget's text
    # directly via ``widget.setText(i18n._tr(text_key))`` right
    # after ``register_widget_text`` returns, so the immediate-
    # apply lookup via ``app.allWidgets()`` was redundant for
    # first-paint correctness. The lookup is the source of the
    # CI pytest 3.11 segfault (exit code 139) at ``_apply_to_one``
    # because PySide6 6.11.x with Python 3.11 segfaults INSIDE the
    # Qt parent-tree walk when a stale widget is present, and
    # SIGSEGV cannot be caught by Python ``try/except``.
    #
    # On language switches, ``_apply_registry`` re-applies the text
    # to every registered widget via its own ``allWidgets()`` walk
    # — but that path is wrapped in ``try/except`` AND skips on
    # failure, so a single stale widget can no longer crash pytest.
    # Skipping the lookup here removes the most common crash site
    # (widget construction triggers dozens of register_widget_text
    # calls, each of which used to walk the entire Qt tree).


def _apply_to_one(object_name: str, attr: str, key: str, fmt: dict | None = None) -> None:
    """Apply a single (objectName, attr, key) immediately.

    Skipped if QApplication is not yet constructed (early imports).

    Audit 2026-08-17: wrap the ``allWidgets()`` iteration in a
    try/except so a stale/deleted widget doesn't segfault the whole
    interpreter. PySide6 raises ``RuntimeError`` ("Internal C++
    object already deleted") when the underlying C++ object has
    been destroyed but the Python wrapper is still alive (common
    during widget construction / destruction in tests). Without
    this guard a single stale widget crashes pytest with exit
    code 139 — observed in CI pytest 3.11 at ``RunTab.__init__``
    when ``tr_checkbox`` was called and a previously-deleted
    widget was still in the Qt parent tree.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    # Phase 37: for ``comboItem`` attr the registered objectName is
    # ``<combo_objectName>:item:<index>`` but the actual widget's
    # objectName is just ``<combo_objectName>``. Strip the suffix
    # so we can find the right QComboBox in the allWidgets() walk.
    lookup_name = object_name
    if attr == "comboItem" and ":item:" in object_name:
        lookup_name = object_name.rsplit(":item:", 1)[0]
    # Iterate defensively: ``app.allWidgets()`` can return widgets
    # whose underlying C++ object has been deleted (e.g. a test
    # constructed a widget, then deleted it without the Python
    # wrapper being GC'd). Calling ``.objectName()`` on a deleted
    # wrapper raises RuntimeError on most PySide6 builds, but on
    # some builds (notably 6.11.x with Python 3.11) it segfaults
    # before the RuntimeError can be raised. Filter explicitly via
    # ``shiboken6.isValid`` (the canonical "is the C++ object still
    # alive" check) and catch RuntimeError as a fallback.
    #
    # Audit 2026-08-17 (second pass): wrap the ``app.allWidgets()``
    # CALL itself in try/except. PySide6 6.11.x with Python 3.11
    # can segfault DURING the internal Qt parent-tree walk when a
    # stale widget is present (CI pytest 3.11 second crash, in
    # ``test_phase42_remaining_fixes::test_main_window_batch_stop_on_error_pauses_batch``
    # -> ``MainWindow._build_ui`` -> ``tr_label`` -> ``register_widget_text``).
    # The previous per-widget ``isValid`` guard did not help because
    # the crash happens inside the Qt parent-tree enumeration
    # before we ever see a widget to validate. Catching at the
    # outer level means: if the tree walk itself crashes, we skip
    # this ``_apply_to_one`` call. The widget was JUST created
    # (tr_label sets objectName THEN calls register_widget_text),
    # so the caller's subsequent ``widget.setText(_tr(text_key))``
    # sets the right text directly — the lookup here is redundant
    # for first-paint correctness.
    try:
        import shiboken6 as _shiboken

        _is_valid = _shiboken.isValid
    except ImportError:
        # Audit 2026-08-17: ruff E731 forbids reassigning a lambda to
        # a name, so we use a def. The shiboken path is the
        # production one; this fallback exists only for headless
        # test environments without PySide6 fully wired in.
        def _is_valid(obj: object) -> bool:  # type: ignore[misc]
            return True

    try:
        all_widgets_iter = app.allWidgets()
    except (RuntimeError, TypeError):
        # Qt parent-tree walk itself crashed; the lookup is
        # non-essential for first-paint (caller will setText right
        # after we return) and for re-apply (will retry on next
        # set_language call when the tree has stabilised).
        return

    for w in list(all_widgets_iter):
        try:
            if not _is_valid(w):
                continue
            if w.objectName() != lookup_name:
                continue
        except (RuntimeError, TypeError):
            # Stale wrapper — skip without breaking the whole pass.
            continue
        text = _tr(key)
        # Phase 34: apply .format() so translated templates
        # like "⚙️ {app}  ·  v{version}" can carry language-
        # independent placeholders.
        if fmt and "{" in text:
            try:
                text = text.format(**fmt)
            except (KeyError, IndexError):
                pass  # leave unformatted if caller didn't supply all keys
        try:
            if attr == "text":
                w.setText(text)
            elif attr == "title":
                w.setTitle(text)
            elif attr == "windowTitle":
                w.setWindowTitle(text)
            elif attr == "placeholderText":
                w.setPlaceholderText(text)
            elif attr == "toolTip":
                w.setToolTip(text)
            elif attr == "statusTip":
                w.setStatusTip(text)
            elif attr == "tabText":
                w.setTabText(int(object_name.split(":")[1]), text)
            elif attr == "comboItem":
                # Phase 37: tr_combobox item translation. The
                # objectName looks like "<combo>:item:<index>".
                # We split on ":" and update the item at index N.
                try:
                    parts = object_name.split(":")
                    idx = int(parts[-1])
                    combo = w
                    # Preserve userData when re-setting text
                    user_data = combo.itemData(idx)
                    combo.setItemText(idx, text)
                    # setItemText may clear userData in some Qt versions
                    if user_data is not None and combo.itemData(idx) is None:
                        combo.setItemData(idx, user_data)
                except (ValueError, IndexError, AttributeError):
                    pass
            else:
                setattr(w, attr, text)
        except Exception as exc:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                "i18n _apply_to_one failed for %s.%s (%s): %s", object_name, attr, key, exc
            )


def _apply_registry() -> None:
    """Re-apply every registered widget for the current language.

    Phase 56 audit: prune dead entries periodically to prevent unbounded growth.
    """
    global _REGISTRY
    app = QApplication.instance()
    if app is None:
        return
    # audit 2026-07-31: the prune used ``app.findChild(QWidget, name)``
    # — findChild searches the QObject tree rooted at the application,
    # but PARENTLESS top-level widgets are NOT in that tree, so every
    # registry entry was pruned and language switches silently stopped
    # re-texting the UI (tests construct widgets without parents).
    # allWidgets() returns every widget including top-levels; index by
    # objectName and prune only entries whose widget is truly gone.
    #
    # Audit 2026-08-17: same defensive iteration guard as
    # ``_apply_to_one`` — wrap ``objectName()`` calls in try/except
    # so a stale/deleted widget doesn't segfault the whole interpreter
    # when ``set_language`` is called from a fixture after a prior
    # test left a deleted C++ object in the allWidgets() walk. Also
    # wrap the ``app.allWidgets()`` call itself — PySide6 6.11.x
    # with Python 3.11 can segfault INSIDE the parent-tree walk
    # when a stale widget is present (CI pytest 3.11 second crash,
    # in MainWindow._build_ui -> tr_label -> register_widget_text).
    try:
        import shiboken6 as _shiboken

        _is_valid = _shiboken.isValid
    except ImportError:
        # Audit 2026-08-17: ruff E731 forbids reassigning a lambda to
        # a name, so we use a def.
        def _is_valid(obj: object) -> bool:  # type: ignore[misc]
            return True

    try:
        all_widgets_iter = app.allWidgets()
    except (RuntimeError, TypeError):
        # Qt parent-tree walk itself crashed — skip this pass and
        # try again on the next set_language() call when the tree
        # has stabilised.
        return

    all_widgets: dict[str, object] = {}
    for w in list(all_widgets_iter):
        try:
            if not _is_valid(w):
                continue
            obj_name = w.objectName()
        except (RuntimeError, TypeError):
            continue
        if obj_name:
            all_widgets[obj_name] = w
    # Rebuild registry keeping only entries whose widgets are still alive
    kept, removed = [], 0
    for entry in _REGISTRY:
        object_name = entry[0]
        lookup_name = object_name
        # comboItem entries register as "<combo>:item:<index>" while
        # the widget's objectName is just "<combo>" — strip the suffix
        # before the lookup (mirrors _apply_to_one).
        if object_name and ":item:" in object_name:
            lookup_name = object_name.rsplit(":item:", 1)[0]
        w = all_widgets.get(lookup_name) if lookup_name else None
        if w is not None:
            kept.append(entry)
        else:
            removed += 1
    if removed > 0:
        _REGISTRY = kept
    for object_name, attr, key, fmt in _REGISTRY:
        _apply_to_one(object_name, attr, key, fmt)


# ============================================================
# Build STRINGS dict at import time. English is always present;
# Chinese loads if the file exists.
# ============================================================
try:
    from .strings_en import STRINGS as _EN_STRINGS
except ImportError:
    _EN_STRINGS = {}

try:
    from .strings_zh_CN import STRINGS as _ZH_STRINGS
except ImportError:
    _ZH_STRINGS = {}

STRINGS: dict[str, dict[str, str]] = {
    "en": dict(_EN_STRINGS),
    "zh_CN": dict(_ZH_STRINGS),
}

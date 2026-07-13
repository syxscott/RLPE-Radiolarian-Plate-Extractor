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

from typing import Callable, Optional


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


def _tr(key: str, default: Optional[str] = None) -> str:
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
    _apply_to_one(object_name, attr, key, fmt)


def _apply_to_one(object_name: str, attr: str, key: str, fmt: dict | None = None) -> None:
    """Apply a single (objectName, attr, key) immediately.

    Skipped if QApplication is not yet constructed (early imports).
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
    for w in app.allWidgets():
        if w.objectName() != lookup_name:
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
        except Exception:
            pass


def _apply_registry() -> None:
    """Re-apply every registered widget for the current language."""
    for object_name, attr, key, fmt in list(_REGISTRY):
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
    "en":    dict(_EN_STRINGS),
    "zh_CN": dict(_ZH_STRINGS),
}

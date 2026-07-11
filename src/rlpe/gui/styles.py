"""QSS styling for the RLPE GUI.

Two themes — Light (default, paper-like) and Dark (research night mode).
The styles target PySide6 / Qt 6 and aim for the look of a polished
scientific desktop app (think: ImageJ, IGV, ChimeraX, Zotero, IgorPro)
— dense info, restrained colour, monospace numerics, no skeuomorphism.
"""
from __future__ import annotations

from PySide6.QtCore import Qt


# ============================================================
# Shared design tokens
# ============================================================
FONT_FAMILY_SANS = (
    "Inter, 'Segoe UI', 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif"
)
FONT_FAMILY_MONO = (
    "'JetBrains Mono', 'Cascadia Code', 'Source Code Pro', 'Menlo', 'Consolas', monospace"
)

# Spacing scale (multiples of 4 px, Material-style)
SPACE_XS = 4
SPACE_S = 8
SPACE_M = 12
SPACE_L = 16
SPACE_XL = 24
SPACE_XXL = 32

# Border radii
RADIUS_S = 3
RADIUS_M = 6
RADIUS_L = 10

# ============================================================
# Light theme (default)
# ============================================================
LIGHT_QSS = f"""
/* --- Global --- */
QWidget {{
    font-family: {FONT_FAMILY_SANS};
    font-size: 13px;
    color: #1f2933;
    background-color: #f7f9fc;
}}
QMainWindow, QDialog {{
    background-color: #eef1f6;
}}
QToolTip {{
    background-color: #1f2933;
    color: #f7f9fc;
    border: 1px solid #4a5568;
    padding: 4px 6px;
    border-radius: 3px;
}}

/* --- Menu / menubar --- */
QMenuBar {{
    background-color: #ffffff;
    border-bottom: 1px solid #cdd5e0;
    padding: 2px 4px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 6px 12px;
    border-radius: {RADIUS_S}px;
}}
QMenuBar::item:selected, QMenuBar::item:hover {{
    background-color: #d6e4ff;
}}
QMenu {{
    background-color: #ffffff;
    border: 1px solid #cdd5e0;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: {RADIUS_S}px;
}}
QMenu::item:selected {{
    background-color: #1f77b4;
    color: #ffffff;
}}
QMenu::separator {{
    height: 1px;
    background: #cdd5e0;
    margin: 4px 8px;
}}

/* --- Toolbar --- */
QToolBar {{
    background-color: #ffffff;
    border-bottom: 1px solid #cdd5e0;
    spacing: 6px;
    padding: 4px 8px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    padding: 4px 10px;
    border-radius: {RADIUS_S}px;
    color: #1f2933;
}}
QToolButton:hover {{
    background-color: #d6e4ff;
    border: 1px solid #1f77b4;
}}
QToolButton:pressed {{
    background-color: #1f77b4;
    color: #ffffff;
}}

/* --- Statusbar --- */
QStatusBar {{
    background-color: #1f2933;
    color: #cdd5e0;
    padding: 2px 8px;
    font-family: {FONT_FAMILY_MONO};
    font-size: 11px;
}}
QStatusBar::item {{
    border: none;
}}

/* --- Groupboxes --- */
QGroupBox {{
    background-color: #ffffff;
    border: 1px solid #cdd5e0;
    border-radius: {RADIUS_M}px;
    margin-top: 16px;
    padding: 18px 10px 10px 10px;
    font-weight: 600;
    color: #1f2933;
}}
/* Phase 36: drop the blue "title badge" subcontrol — it was
   clipping the title text to its first 4 characters when the
   subcontrol's geometry was computed before QGroupBox had its
   final width. Instead use a plain title above the border. */
QGroupBox::title {{
    subcontrol-origin: padding;
    subcontrol-position: top left;
    padding: 0 4px;
    left: 8px;
    color: #1f2933;
    background-color: transparent;
}}

/* --- Inputs --- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: #ffffff;
    color: #1f2933;
    border: 1px solid #cdd5e0;
    border-radius: {RADIUS_S}px;
    /* Phase 36: vertical padding 2px (was 4) so a 30-px tall row
       leaves 26 px for the inner lineedit / arrows. The previous
       4px padding stole 8 px and pushed the spinbox arrows OUTSIDE
       the widget rectangle. */
    padding: 2px 8px;
    selection-background-color: #1f77b4;
    selection-color: #ffffff;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid #1f77b4;
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background-color: #eef1f6;
    color: #7f7f7f;
}}
QPlainTextEdit, QTextEdit {{
    font-family: {FONT_FAMILY_MONO};
    font-size: 12px;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #cdd5e0;
}}
/* Phase 36: keep spinbox up/down buttons inside the widget rect. */
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 16px;
    subcontrol-origin: border;
}}
QSpinBox::up-button {{ subcontrol-position: top right; }}
QSpinBox::down-button {{ subcontrol-position: bottom right; }}
QDoubleSpinBox::up-button {{ subcontrol-position: top right; }}
QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; }}

/* --- Buttons --- */
QPushButton {{
    background-color: #ffffff;
    color: #1f2933;
    border: 1px solid #cdd5e0;
    border-radius: {RADIUS_S}px;
    padding: 6px 16px;
    /* Phase 35: bumped from 22 → 30 so buttons inside QFormLayout rows
       are the same height as QSpinBox / QComboBox / QLineEdit, and
       the text never clips at 150% DPI or under the QSS dark theme. */
    min-height: 30px;
}}
QPushButton:hover {{
    background-color: #d6e4ff;
    border: 1px solid #1f77b4;
}}
QPushButton:pressed {{
    background-color: #1f77b4;
    color: #ffffff;
}}
QPushButton:disabled {{
    background-color: #eef1f6;
    color: #7f7f7f;
    border: 1px solid #eef1f6;
}}
QPushButton#primary {{
    background-color: #1f77b4;
    color: #ffffff;
    border: 1px solid #1f77b4;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background-color: #2e8bc0;
    border: 1px solid #2e8bc0;
}}
QPushButton#danger {{
    background-color: #d62728;
    color: #ffffff;
    border: 1px solid #d62728;
}}
QPushButton#danger:hover {{
    background-color: #e84545;
    border: 1px solid #e84545;
}}
QPushButton#flat {{
    background: transparent;
    border: 1px solid transparent;
}}
QPushButton#flat:hover {{
    background-color: #eef1f6;
}}

/* --- Checkboxes / radios --- */
QCheckBox, QRadioButton {{
    spacing: 6px;
    color: #1f2933;
    /* Phase 35: ensure the checkbox row in QFormLayout is the same
       height as the spinbox/combobox/lineedit rows so long labels
       like "Enable PBDB enrichment (taxonomy + occurrences)" wrap
       cleanly instead of clipping to ~16 px. */
    min-height: 30px;
    padding: 2px 0;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid #4a5568;
    border-radius: 3px;
    background: #ffffff;
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QCheckBox::indicator:checked {{
    background-color: #1f77b4;
    border: 1px solid #1f77b4;
    image: url(none);
}}
QRadioButton::indicator:checked {{
    background-color: #1f77b4;
    border: 4px solid #ffffff;
}}

/* --- Tabs --- */
QTabWidget::pane {{
    border: 1px solid #cdd5e0;
    background: #ffffff;
    border-radius: 0 0 {RADIUS_M}px {RADIUS_M}px;
}}
QTabBar::tab {{
    background: #eef1f6;
    color: #1f2933;
    padding: 8px 18px;
    margin-right: 1px;
    border: 1px solid #cdd5e0;
    border-bottom: none;
    border-top-left-radius: {RADIUS_M}px;
    border-top-right-radius: {RADIUS_M}px;
}}
QTabBar::tab:selected {{
    background: #ffffff;
    color: #1f77b4;
    font-weight: 600;
    border-bottom: 1px solid #ffffff;
}}
QTabBar::tab:hover:!selected {{
    background: #d6e4ff;
}}

/* --- Tables --- */
QTableWidget, QTableView, QTreeView, QTreeWidget, QListWidget {{
    background-color: #ffffff;
    alternate-background-color: #f7f9fc;
    gridline-color: #eef1f6;
    border: 1px solid #cdd5e0;
    border-radius: {RADIUS_M}px;
    selection-background-color: #1f77b4;
    selection-color: #ffffff;
}}
QHeaderView::section {{
    background-color: #eef1f6;
    color: #1f2933;
    padding: 6px 10px;
    border: none;
    border-right: 1px solid #cdd5e0;
    border-bottom: 1px solid #cdd5e0;
    font-weight: 600;
}}
QHeaderView::section:last {{
    border-right: none;
}}
QTableCornerButton::section {{
    background-color: #eef1f6;
    border: none;
}}

/* --- Progress bars --- */
QProgressBar {{
    background-color: #eef1f6;
    color: #1f2933;
    border: 1px solid #cdd5e0;
    border-radius: {RADIUS_S}px;
    text-align: center;
    min-height: {PROGRESS_BAR_MIN_HEIGHT_PX if False else 18}px;
}}
QProgressBar::chunk {{
    background-color: #1f77b4;
    border-radius: {RADIUS_S}px;
}}
QProgressBar::chunk[status="failed"] {{
    background-color: #d62728;
}}
QProgressBar::chunk[status="done"] {{
    background-color: #2ca02c;
}}

/* --- Scrollbars --- */
QScrollBar:vertical {{
    background: #f7f9fc;
    width: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #cdd5e0;
    min-height: 24px;
    border-radius: 6px;
}}
QScrollBar::handle:vertical:hover {{
    background: #4a5568;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
    background: none;
}}
QScrollBar:horizontal {{
    background: #f7f9fc;
    height: 12px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: #cdd5e0;
    min-width: 24px;
    border-radius: 6px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
    background: none;
}}

/* --- Splitter --- */
QSplitter::handle {{
    background-color: #cdd5e0;
}}
QSplitter::handle:horizontal {{
    height: 2px;
}}
QSplitter::handle:vertical {{
    width: 2px;
}}

/* --- Frames (used for box layouts) --- */
QFrame[frameShape="4"], QFrame[frameShape="5"], QFrame[frameShape="6"] {{
    color: #cdd5e0;
}}

/* --- Status badges (small colour-coded labels) --- */
QLabel#statusQueued   {{ color: #7f7f7f; font-weight: 600; }}
QLabel#statusRunning  {{ color: #1f77b4; font-weight: 600; }}
QLabel#statusDone     {{ color: #2ca02c; font-weight: 600; }}
QLabel#statusFailed   {{ color: #d62728; font-weight: 600; }}
QLabel#statusCancelled{{ color: #ff7f0e; font-weight: 600; }}

QLabel#caption {{
    font-family: {FONT_FAMILY_MONO};
    font-size: 12px;
    color: #1f2933;
    background-color: #f7f9fc;
    padding: 4px 6px;
    border-radius: 3px;
}}
QLabel#metric {{
    font-family: {FONT_FAMILY_MONO};
    font-size: 14px;
    font-weight: 600;
    color: #1f77b4;
}}
QLabel#metricLabel {{
    font-size: 11px;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QLabel#sectionTitle {{
    font-size: 16px;
    font-weight: 700;
    color: #1f2933;
    padding-bottom: 4px;
    border-bottom: 2px solid #1f77b4;
}}
QLabel#warning {{
    color: #ff7f0e;
}}
QLabel#error {{
    color: #d62728;
    font-weight: 600;
}}
QLabel#success {{
    color: #2ca02c;
    font-weight: 600;
}}
"""


# ============================================================
# Dark theme (research night mode)
# ============================================================
DARK_QSS = f"""
QWidget {{
    font-family: {FONT_FAMILY_SANS};
    font-size: 13px;
    color: #e2e8f0;
    background-color: #1a202c;
}}
QMainWindow, QDialog {{
    background-color: #171923;
}}
QToolTip {{
    background-color: #e2e8f0;
    color: #1a202c;
    border: 1px solid #4a5568;
    padding: 4px 6px;
    border-radius: 3px;
}}
QMenuBar {{
    background-color: #1f2937;
    border-bottom: 1px solid #2d3748;
    padding: 2px 4px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 6px 12px;
    border-radius: {RADIUS_S}px;
}}
QMenuBar::item:selected, QMenuBar::item:hover {{
    background-color: #2c5282;
}}
QMenu {{
    background-color: #1f2937;
    border: 1px solid #2d3748;
    padding: 4px;
    color: #e2e8f0;
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: {RADIUS_S}px;
}}
QMenu::item:selected {{
    background-color: #3182ce;
    color: #ffffff;
}}
QToolBar {{
    background-color: #1f2937;
    border-bottom: 1px solid #2d3748;
    spacing: 6px;
    padding: 4px 8px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    padding: 4px 10px;
    border-radius: {RADIUS_S}px;
    color: #e2e8f0;
}}
QToolButton:hover {{
    background-color: #2c5282;
    border: 1px solid #3182ce;
}}
QToolButton:pressed {{
    background-color: #3182ce;
    color: #ffffff;
}}
QStatusBar {{
    background-color: #0f1419;
    color: #a0aec0;
    padding: 2px 8px;
    font-family: {FONT_FAMILY_MONO};
    font-size: 11px;
}}
QGroupBox {{
    background-color: #1f2937;
    border: 1px solid #2d3748;
    border-radius: {RADIUS_M}px;
    margin-top: 16px;
    padding: 18px 10px 10px 10px;
    font-weight: 600;
    color: #e2e8f0;
}}
/* Phase 36: drop the blue "title badge" subcontrol (same as
   light theme) — it was clipping the title text. Plain title
   above the border. */
QGroupBox::title {{
    subcontrol-origin: padding;
    subcontrol-position: top left;
    padding: 0 4px;
    left: 8px;
    color: #e2e8f0;
    background-color: transparent;
}}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: #0f1419;
    color: #e2e8f0;
    border: 1px solid #2d3748;
    border-radius: {RADIUS_S}px;
    /* Phase 36: same fix as light theme — vertical padding 2px so
       the spinbox up/down arrows fit inside the 30-px row. */
    padding: 2px 8px;
    selection-background-color: #3182ce;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid #3182ce;
}}
QPlainTextEdit, QTextEdit {{
    font-family: {FONT_FAMILY_MONO};
    font-size: 12px;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 16px;
    subcontrol-origin: border;
}}
QSpinBox::up-button {{ subcontrol-position: top right; }}
QSpinBox::down-button {{ subcontrol-position: bottom right; }}
QDoubleSpinBox::up-button {{ subcontrol-position: top right; }}
QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; }}
QPushButton {{
    background-color: #1f2937;
    color: #e2e8f0;
    border: 1px solid #2d3748;
    border-radius: {RADIUS_S}px;
    padding: 6px 16px;
    /* Phase 35: bumped from 22 → 30 for parity with form-input rows. */
    min-height: 30px;
}}
QPushButton:hover {{
    background-color: #2c5282;
    border: 1px solid #3182ce;
}}
QPushButton:pressed {{
    background-color: #3182ce;
    color: #ffffff;
}}
QPushButton:disabled {{
    background-color: #1a202c;
    color: #4a5568;
    border: 1px solid #1f2937;
}}
QPushButton#primary {{
    background-color: #3182ce;
    color: #ffffff;
    border: 1px solid #3182ce;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background-color: #4299e1;
    border: 1px solid #4299e1;
}}
QPushButton#danger {{
    background-color: #e53e3e;
    color: #ffffff;
    border: 1px solid #e53e3e;
}}
QPushButton#danger:hover {{
    background-color: #f56565;
    border: 1px solid #f56565;
}}
QPushButton#flat {{
    background: transparent;
    border: 1px solid transparent;
}}
QPushButton#flat:hover {{
    background-color: #2d3748;
}}
QCheckBox, QRadioButton {{
    spacing: 6px;
    color: #e2e8f0;
    /* Phase 35: same as the light theme — min-height 30px + 2px
       vertical padding keeps long checkbox labels readable. */
    min-height: 30px;
    padding: 2px 0;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid #4a5568;
    border-radius: 3px;
    background: #0f1419;
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QCheckBox::indicator:checked {{
    background-color: #3182ce;
    border: 1px solid #3182ce;
}}
QRadioButton::indicator:checked {{
    background-color: #3182ce;
    border: 4px solid #0f1419;
}}
QTabWidget::pane {{
    border: 1px solid #2d3748;
    background: #1f2937;
    border-radius: 0 0 {RADIUS_M}px {RADIUS_M}px;
}}
QTabBar::tab {{
    background: #1a202c;
    color: #e2e8f0;
    padding: 8px 18px;
    margin-right: 1px;
    border: 1px solid #2d3748;
    border-bottom: none;
    border-top-left-radius: {RADIUS_M}px;
    border-top-right-radius: {RADIUS_M}px;
}}
QTabBar::tab:selected {{
    background: #1f2937;
    color: #63b3ed;
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    background: #2d3748;
}}
QTableWidget, QTableView, QTreeView, QTreeWidget, QListWidget {{
    background-color: #1f2937;
    alternate-background-color: #1a202c;
    gridline-color: #2d3748;
    border: 1px solid #2d3748;
    border-radius: {RADIUS_M}px;
    selection-background-color: #3182ce;
    selection-color: #ffffff;
}}
QHeaderView::section {{
    background-color: #1a202c;
    color: #e2e8f0;
    padding: 6px 10px;
    border: none;
    border-right: 1px solid #2d3748;
    border-bottom: 1px solid #2d3748;
    font-weight: 600;
}}
QProgressBar {{
    background-color: #0f1419;
    color: #e2e8f0;
    border: 1px solid #2d3748;
    border-radius: {RADIUS_S}px;
    text-align: center;
    min-height: 18px;
}}
QProgressBar::chunk {{
    background-color: #3182ce;
    border-radius: {RADIUS_S}px;
}}
QScrollBar:vertical {{
    background: #1a202c;
    width: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #4a5568;
    min-height: 24px;
    border-radius: 6px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
    background: none;
}}
QScrollBar:horizontal {{
    background: #1a202c;
    height: 12px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: #4a5568;
    min-width: 24px;
    border-radius: 6px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
    background: none;
}}
QSplitter::handle {{
    background-color: #2d3748;
}}
QLabel#statusQueued   {{ color: #a0aec0; font-weight: 600; }}
QLabel#statusRunning  {{ color: #63b3ed; font-weight: 600; }}
QLabel#statusDone     {{ color: #68d391; font-weight: 600; }}
QLabel#statusFailed   {{ color: #fc8181; font-weight: 600; }}
QLabel#statusCancelled{{ color: #f6ad55; font-weight: 600; }}
QLabel#caption {{
    font-family: {FONT_FAMILY_MONO};
    font-size: 12px;
    color: #e2e8f0;
    background-color: #0f1419;
    padding: 4px 6px;
    border-radius: 3px;
}}
QLabel#metric {{
    font-family: {FONT_FAMILY_MONO};
    font-size: 14px;
    font-weight: 600;
    color: #63b3ed;
}}
QLabel#metricLabel {{
    font-size: 11px;
    color: #a0aec0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QLabel#sectionTitle {{
    font-size: 16px;
    font-weight: 700;
    color: #e2e8f0;
    padding-bottom: 4px;
    border-bottom: 2px solid #3182ce;
}}
QLabel#warning {{
    color: #f6ad55;
}}
QLabel#error {{
    color: #fc8181;
    font-weight: 600;
}}
QLabel#success {{
    color: #68d391;
    font-weight: 600;
}}
"""


def apply_theme(app, theme: str) -> None:
    """Apply the requested QSS theme to a ``QApplication``.

    Parameters
    ----------
    app : QApplication
        The running application instance.
    theme : str
        One of ``"light"``, ``"dark"``, or ``"system"``.
    """
    from PySide6.QtWidgets import QApplication

    if theme == "dark":
        app.setStyleSheet(DARK_QSS)
    elif theme == "light":
        app.setStyleSheet(LIGHT_QSS)
    else:
        # ``system`` = pick based on Qt's palette hint
        app.setStyleSheet(DARK_QSS if app.palette().window().color().lightness() < 128 else LIGHT_QSS)
"""Image preview widget — QGraphicsView with zoom/pan + bbox overlay.

Designed for radiolarian plate images:
* Loads PNG / JPG via cv2 + numpy → QPixmap
* Renders detected panel bounding boxes as QGraphicsRectItem overlays
* Mouse wheel = zoom (anchored at cursor)
* Drag = pan
* Double-click = fit to window
* Crosshair cursor when "measure" tool is active (Phase 33+)

This is the single most complex widget in the GUI. It is also
the one most likely to need iteration — keep the interface small
(zoom/pan/fit only for now) and add tools (measure, annotate,
delete bbox) in Phase 33+.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .constants import BUTTON_MIN_WIDTH, IMAGE_PREVIEW_MIN_SIZE
from .styles import SPACE_M, SPACE_S
from . import i18n
from .i18n_widgets import tr_label
from .utils import get_gui_logger


# ============================================================
# Image preview widget
# ============================================================
class ImagePreviewWidget(QWidget):
    """Single-image preview with bbox overlay + zoom/pan controls."""

    bbox_clicked = Signal(dict)  # the bbox dict that was clicked

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log = get_gui_logger()
        self._current_path: Path | None = None
        self._bboxes: list[dict[str, Any]] = []
        # Phase 41: track QGraphicsRectItem references for bbox
        # overlays so we can remove them on set_image() and avoid
        # the leak where old bboxes pile up across image swaps.
        self._bbox_items: list[QGraphicsRectItem] = []
        # Phase 44: also track QGraphicsTextItem labels (species
        # name + confidence). Without this, text labels pile up
        # across set_image() / set_bboxes() calls because the
        # text items aren't QGraphicsRectItem and were never
        # removed in _overlay_bboxes().
        self._text_items: list[QGraphicsTextItem] = []
        self._zoom: float = 1.0
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Toolbar (small)
        bar = QHBoxLayout()
        bar.setContentsMargins(SPACE_S, SPACE_S, SPACE_S, 0)
        bar.setSpacing(SPACE_S)

        self._path_label = tr_label("preview.no_image", object_name="preview.path_label")
        self._path_label.setObjectName("metricLabel")
        bar.addWidget(self._path_label, 1)

        # Phase 37 audit fix: use i18n keys for the toolbar buttons so
        # the labels translate on language switch. (Emoji characters
        # have inconsistent width on offscreen renderers, so we set
        # an explicit max width to keep the toolbar from reflowing.)
        from .i18n_widgets import tr_button
        for key, slot in (
            ("preview.zoom_in", self._zoom_in),
            ("preview.zoom_out", self._zoom_out),
            ("preview.fit", self._fit_window),
            ("preview.actual", self._zoom_actual),
        ):
            btn = tr_button(key)
            btn.setMaximumWidth(BUTTON_MIN_WIDTH + 30)  # emoji + cn label
            btn.clicked.connect(slot)
            bar.addWidget(btn)

        outer.addLayout(bar)

        # Graphics view
        self._scene = QGraphicsScene(self)
        self._view = _PreviewGraphicsView(self._scene, self)
        self._view.setRenderHint(QPainter.Antialiasing)
        self._view.setRenderHint(QPainter.SmoothPixmapTransform)
        self._view.setBackgroundBrush(QBrush(QColor("#1f2937")))
        self._view.setMinimumSize(*IMAGE_PREVIEW_MIN_SIZE)
        self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self._view, 1)

        # Hint bar
        self._hint = QLabel("Wheel = zoom · Drag = pan · Double-click = fit · "
                            "Click a bbox to select")
        self._hint.setObjectName("metricLabel")
        self._hint.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._hint)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_image(self, image_path: Path | str | None) -> None:
        """Load an image from disk and display it."""
        if image_path is None:
            self._current_path = None
            self._scene.clear()
            self._bbox_items.clear()  # Phase 41: clear bbox items too
            self._path_label.setText(i18n._tr("preview.no_image"))
            return
        path = Path(image_path)
        self._current_path = path
        if not path.exists():
            self._scene.clear()
            self._bbox_items.clear()  # Phase 41: clear bbox items too
            self._path_label.setText(i18n._tr("preview.missing").format(name=path.name))
            return
        pix = self._load_pixmap(path)
        if pix is None:
            self._scene.clear()
            self._bbox_items.clear()  # Phase 41: clear bbox items too
            self._path_label.setText(i18n._tr("preview.failed").format(name=path.name))
            return
        self._scene.clear()
        # Phase 41: also clear our explicit bbox item list so old
        # overlays don't survive a set_image() call.
        self._bbox_items.clear()
        item = QGraphicsPixmapItem(pix)
        self._scene.addItem(item)
        self._scene.setSceneRect(QRectF(pix.rect()))
        # Reset zoom and fit
        self._zoom = 1.0
        self._path_label.setText(str(path))
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        # Re-overlay bboxes
        self._overlay_bboxes()

    def set_bboxes(self, bboxes: list[dict[str, Any]]) -> None:
        """Replace the list of bbox overlays.

        Each bbox is a dict with keys:
          - bbox: (x, y, w, h) in image pixel coords
          - species: optional str label
          - confidence: optional float
        """
        self._bboxes = list(bboxes)
        self._overlay_bboxes()

    def clear(self) -> None:
        self._scene.clear()
        self._bboxes = []
        self._path_label.setText(i18n._tr("preview.no_image"))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _load_pixmap(self, path: Path) -> QPixmap | None:
        """Load image via cv2 (BGR → RGB) or PIL fallback."""
        try:
            import cv2

            arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if arr is None:
                # Fallback: try PIL
                from PIL import Image as PILImage
                with PILImage.open(path) as im:
                    arr = np.array(im.convert("RGB"))
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            h, w = arr.shape[:2]
            qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)
        except Exception as exc:
            self._log.warning("Failed to load image %s: %s", path, exc)
            return None

    def _overlay_bboxes(self) -> None:
        """Draw bounding boxes as QGraphicsRectItem overlays."""
        # Phase 41: remove any existing bbox items first so successive
        # set_bboxes / set_image calls don't pile up overlays.
        for item in self._bbox_items:
            try:
                self._scene.removeItem(item)
            except RuntimeError:
                # Item may have been deleted by Qt already
                pass
        self._bbox_items.clear()
        # Phase 44: also remove QGraphicsTextItem labels. Previously
        # these leaked across set_image() / set_bboxes() calls
        # because the text items weren't tracked in _bbox_items
        # (they're a different Qt class). Result: species name
        # labels piled up at the top-left of the preview.
        for item in self._text_items:
            try:
                self._scene.removeItem(item)
            except RuntimeError:
                pass
        self._text_items.clear()

        if not self._current_path or not self._bboxes:
            return
        # Ensure scene still has the pixmap item
        items = self._scene.items()
        if not items:
            return
        # Colours cycle through a small palette
        palette = [
            QColor("#1f77b4"),  # blue
            QColor("#2ca02c"),  # green
            QColor("#d62728"),  # red
            QColor("#9467bd"),  # purple
            QColor("#ff7f0e"),  # orange
        ]
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        for i, bbox in enumerate(self._bboxes):
            coords = bbox.get("bbox") or bbox.get("bounding_box")
            if not coords or len(coords) != 4:
                continue
            x, y, w, h = coords
            colour = palette[i % len(palette)]
            pen = QPen(colour, 3)
            rect_item = QGraphicsRectItem(x, y, w, h)
            rect_item.setPen(pen)
            rect_item.setBrush(QBrush(QColor(colour.red(), colour.green(), colour.blue(), 32)))
            rect_item.setData(0, bbox)
            rect_item.setToolTip(_bbox_tooltip(bbox))
            self._scene.addItem(rect_item)
            self._bbox_items.append(rect_item)
            # Label with species name (if any)
            species = bbox.get("species") or bbox.get("label_text") or ""
            confidence = bbox.get("confidence")
            label_text = species[:30] if species else f"#{i+1}"
            if isinstance(confidence, (int, float)):
                label_text = f"{label_text}  ({confidence:.2f})"
            text_item = self._scene.addText(label_text, font)
            text_item.setDefaultTextColor(colour)
            text_item.setPos(x + 4, y + 4)
            text_item.setData(0, bbox)
            text_item.setToolTip(_bbox_tooltip(bbox))
            # Phase 44: track the text item so _overlay_bboxes can
            # remove it on the next set_bboxes / set_image call.
            self._text_items.append(text_item)

    # ------------------------------------------------------------------
    # Toolbar slots
    # ------------------------------------------------------------------
    def _zoom_in(self) -> None:
        self._view.zoom_by(1.25, anchor_view_center=True)

    def _zoom_out(self) -> None:
        self._view.zoom_by(1 / 1.25, anchor_view_center=True)

    def _fit_window(self) -> None:
        if self._scene.sceneRect().isEmpty():
            return
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom = 1.0

    def _zoom_actual(self) -> None:
        if self._scene.sceneRect().isEmpty():
            return
        self._view.resetTransform()
        self._zoom = 1.0


# ============================================================
# Internal graphics view with wheel-zoom + drag-pan
# ============================================================
class _PreviewGraphicsView(QGraphicsView):
    """QGraphicsView subclass with the preview UX conventions."""

    ZOOM_MIN = 0.05
    ZOOM_MAX = 25.0

    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self._drag_mode = False
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setRenderHints(self.renderHints())

    # ------------------------------------------------------------------
    # Zoom + pan
    # ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """Wheel = zoom (1.15× per notch, anchored at cursor)."""
        # AngleDelta is the standard wheel event payload
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        self.zoom_by(factor)
        # Phase 44: accept the event so it doesn't propagate to
        # the parent QScrollArea (which would scroll the panel
        # while the user is trying to zoom the image).
        event.accept()

    def zoom_by(self, factor: float, anchor_view_center: bool = False) -> None:
        if anchor_view_center:
            # Save current centre
            centre_scene = self.mapToScene(self.viewport().rect().center())
            self.scale(factor, factor)
            # Re-centre
            self.centerOn(centre_scene)
        else:
            self.scale(factor, factor)
        # Phase 42: clamp zoom WITHOUT resetting the transform.
        # The old code did ``self.resetTransform()`` when the
        # accumulated zoom exceeded ZOOM_MAX / ZOOM_MIN, which threw
        # away the pan offset and reset the centre to the origin.
        # Fixed: scale incrementally UP to the limit, not from
        # the original baseline. After this, the accumulated scale
        # is at most ZOOM_MAX / ZOOM_MIN but the pan offset is
        # preserved.
        current = self.transform().m11()
        if current > self.ZOOM_MAX:
            # We overshot ZOOM_MAX. Compute the inverse-scale factor
            # needed to bring us back to ZOOM_MAX, and apply it
            # relative to the current scale (not via resetTransform).
            correction = self.ZOOM_MAX / current
            self.scale(correction, correction)
        elif current < self.ZOOM_MIN:
            correction = self.ZOOM_MIN / current
            self.scale(correction, correction)

    # ------------------------------------------------------------------
    # Drag-pan with middle / right mouse button
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._drag_mode = True
            self._drag_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        # Bbox click on left-button → emit signal
        if event.button() == Qt.LeftButton and self.scene():
            scene_pos = self.mapToScene(event.pos())
            for item in self.scene().items(scene_pos):
                # Phase 44: also accept clicks on QGraphicsTextItem
                # labels (species name + confidence). Previously the
                # check skipped text items, so clicking a label did
                # nothing (the cursor changed but no signal fired).
                if isinstance(item, (QGraphicsRectItem, QGraphicsTextItem)):
                    bbox_data = item.data(0)
                    if bbox_data:
                        # Bubble up via parent widget
                        w = self.parentWidget()
                        while w is not None and not isinstance(w, ImagePreviewWidget):
                            w = w.parentWidget()
                        if w is not None:
                            w.bbox_clicked.emit(bbox_data)
                    break
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_mode:
            delta = event.pos() - self._drag_start
            self._drag_start = event.pos()
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_mode and event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._drag_mode = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        """Double-click anywhere → fit image to window."""
        if self.scene() and not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


# ============================================================
# Helpers
# ============================================================
def _bbox_tooltip(bbox: dict[str, Any]) -> str:
    """Render a bbox dict as a multi-line tooltip."""
    parts = []
    species = bbox.get("species")
    if species:
        parts.append(f"<b>{species}</b>")
    confidence = bbox.get("confidence")
    if isinstance(confidence, (int, float)):
        parts.append(f"confidence: {confidence:.2f}")
    coords = bbox.get("bbox") or bbox.get("bounding_box")
    if coords and len(coords) == 4:
        x, y, w, h = coords
        parts.append(f"x: {x:.0f}  y: {y:.0f}")
        parts.append(f"w: {w:.0f}  h: {h:.0f}")
    family = (bbox.get("metadata") or {}).get("paleodb", {}).get("taxonomy", {}).get("family") if bbox.get("metadata") else None
    if family:
        parts.append(f"family: {family}")
    return "<br>".join(parts) if parts else ""
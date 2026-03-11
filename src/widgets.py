from typing import Any, Callable, Dict, List, Optional
import math
import copy

from PySide6.QtCore import Qt, QPointF, QRectF, QLineF, QEvent
from PySide6.QtGui import (
    QMouseEvent,
    QPainter,
    QPen,
    QBrush,
    QPainterPath,
    QWheelEvent,
    QColor,
    QPixmap,
)
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QFrame,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsPathItem,
    QGraphicsLineItem,
    QGraphicsEllipseItem,
)


class DraggablePanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_handle: Optional[QWidget] = None
        self._drag_offset = QPointF()
        self.user_moved = False

    def set_drag_handle(self, widget: QWidget):
        if self._drag_handle is not None:
            self._drag_handle.removeEventFilter(self)
        self._drag_handle = widget
        self._drag_handle.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self._drag_handle:
            if (
                event.type() == QEvent.MouseButtonPress
                and event.button() == Qt.LeftButton
            ):
                self._drag_offset = (
                    event.globalPosition() - self.frameGeometry().topLeft()
                )
                event.accept()
                return True
            if event.type() == QEvent.MouseMove and event.buttons() & Qt.LeftButton:
                parent = self.parentWidget()
                if parent is None:
                    return False
                new_top_left = event.globalPosition() - self._drag_offset
                p = parent.mapFromGlobal(new_top_left.toPoint())
                max_x = max(0, parent.width() - self.width())
                max_y = max(0, parent.height() - self.height())
                self.move(max(0, min(p.x(), max_x)), max(0, min(p.y(), max_y)))
                self.user_moved = True
                event.accept()
                return True
            if (
                event.type() == QEvent.MouseButtonRelease
                and event.button() == Qt.LeftButton
            ):
                event.accept()
                return True
        return super().eventFilter(obj, event)


class ROIListWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("ROI List")
        self.resize(300, 420)
        self.on_closed: Optional[Callable[[], None]] = None
        self._closed_by_user = True

        self.setStyleSheet(
            "QWidget { background: rgba(28,28,28,235); color: #ddd; }"
            "QListWidget { background: rgba(20,20,20,220); border: 1px solid #555; }"
            "QPushButton { background: #3a3a3a; border: 1px solid #666; padding: 4px 8px; }"
            "QPushButton:hover { background: #4a4a4a; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(QLabel("ROI List"))

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget, 1)

        row = QHBoxLayout()
        self.btn_save = QPushButton("Save")
        self.btn_load = QPushButton("Load")
        self.btn_close = QPushButton("Close")
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_close)
        layout.addLayout(row)

        self.btn_close.clicked.connect(self.close)

    def hide_programmatically(self):
        self._closed_by_user = False
        self.hide()
        self._closed_by_user = True

    def closeEvent(self, event):
        super().closeEvent(event)
        if self._closed_by_user and self.on_closed:
            self.on_closed()


class ImageViewer(QGraphicsView):
    ROI_NONE = "none"
    ROI_POLYGON = "polygon"
    ROI_RECT = "rect"
    ROI_ELLIPSE = "ellipse"

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)

        self.setRenderHints(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self._has_image = False
        self._min_zoom = 0.05
        self._max_zoom = 40.0
        self._pan_ema = QPointF(0.0, 0.0)
        self._pan_residual = QPointF(0.0, 0.0)

        # UI Overlay
        self._setup_ui_overlay()

        # ROI state
        self._roi_mode = False
        self._roi_type = self.ROI_NONE
        self._rois: List[Dict[str, Any]] = []
        self._selected_idx: Optional[int] = None
        self._drawing_points: List[QPointF] = []
        self._drawing_hover: Optional[QPointF] = None
        self._shape_start: Optional[QPointF] = None
        self._shape_end: Optional[QPointF] = None
        self._shape_drawing = False

        self._snap_threshold_px = 14.0
        self._vertex_radius = 4.0
        self._vertex_radius_hover = 6.0
        self._edge_hit_threshold_px = 8.0

        self._roi_paths: List[QGraphicsPathItem] = []
        self._roi_preview = QGraphicsLineItem()
        self._roi_preview.setZValue(11)
        self._roi_preview.setPen(QPen(QColor(0, 220, 255), 1.2, Qt.DashLine))
        self._scene.addItem(self._roi_preview)
        self._roi_preview.hide()

        self._drawing_path = QGraphicsPathItem()
        self._drawing_path.setZValue(12)
        self._drawing_path.setPen(QPen(QColor(0, 220, 255), 1.2, Qt.DashLine))
        self._drawing_path.setBrush(QBrush(Qt.NoBrush))
        self._scene.addItem(self._drawing_path)

        self._shape_preview_path = QGraphicsPathItem()
        self._shape_preview_path.setZValue(12)
        self._shape_preview_path.setPen(QPen(QColor(0, 220, 255), 1.4, Qt.DashLine))
        self._shape_preview_path.setBrush(QBrush(QColor(120, 220, 255, 25)))
        self._scene.addItem(self._shape_preview_path)

        self._roi_vertex_items: List[QGraphicsEllipseItem] = []
        self._hover_vertex_idx: Optional[int] = None
        self._drag_vertex_idx: Optional[int] = None
        self._drag_rect_handle: Optional[str] = None
        self._drag_move_roi = False
        self._drag_last_scene: Optional[QPointF] = None
        self._newly_inserted_vertex = False

        self.on_rois_changed: Optional[
            Callable[[List[Dict[str, Any]], Optional[int]], None]
        ] = None
        self._suppress_notify = False

    def _setup_ui_overlay(self):
        # Create a container for the buttons
        self.overlay_panel = QFrame(self)
        self.overlay_panel.setObjectName("overlayPanel")
        self.overlay_panel.setStyleSheet(
            "#overlayPanel { background: rgba(40,40,40,150); border: 1px solid #555; border-radius: 4px; }"
            "QPushButton { background: rgba(60,60,60,200); border: 1px solid #777; border-radius: 2px; color: white; min-width: 32px; min-height: 32px; font-weight: bold; font-size: 16px; }"
            "QPushButton:hover { background: rgba(80,80,80,220); border-color: #999; }"
            "QPushButton:pressed { background: rgba(45,123,216,200); }"
            "QPushButton:checked { background: rgba(45,123,216,220); border-color: #64a7ff; }"
        )

        v_layout = QVBoxLayout(self.overlay_panel)
        v_layout.setContentsMargins(4, 4, 4, 4)
        v_layout.setSpacing(4)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setToolTip("Zoom In")
        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setToolTip("Zoom Out")
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setToolTip("Fit View")
        self.btn_fit.setStyleSheet("font-size: 11px;")
        self.btn_roi = QPushButton("ROI")
        self.btn_roi.setCheckable(True)
        self.btn_roi.setToolTip("Toggle ROI Mode")
        self.btn_roi.setStyleSheet("font-size: 11px;")

        v_layout.addWidget(self.btn_zoom_in)
        v_layout.addWidget(self.btn_zoom_out)
        v_layout.addWidget(self.btn_fit)
        v_layout.addWidget(self.btn_roi)

        self.overlay_panel.adjustSize()
        self.overlay_panel.hide()
        self._update_overlay_pos()

    def _update_overlay_pos(self):
        # Position at top right with some margin
        margin = 10
        x = self.width() - self.overlay_panel.width() - margin
        y = margin
        self.overlay_panel.move(max(0, x), y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_overlay_pos()

    def set_image(self, pixmap: QPixmap, fit: bool = True):
        self._pix_item.setPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self._has_image = not pixmap.isNull()

        if self._has_image:
            self.overlay_panel.show()
        else:
            self.overlay_panel.hide()

        if fit:
            self.fit_in_view()

    def fit_in_view(self):
        if not self._has_image:
            return
        self.resetTransform()
        self.fitInView(self._pix_item, Qt.KeepAspectRatio)
        self._clamp_zoom_to_limits()

    def _current_zoom(self) -> float:
        return float(self.transform().m11())

    def _clamp_zoom_to_limits(self):
        cur = self._current_zoom()
        if cur <= 0:
            return
        if cur < self._min_zoom:
            self.scale(self._min_zoom / cur, self._min_zoom / cur)
        elif cur > self._max_zoom:
            self.scale(self._max_zoom / cur, self._max_zoom / cur)

    def _apply_zoom_factor(self, factor: float):
        if not self._has_image:
            return
        if factor <= 0:
            return
        cur = self._current_zoom()
        target = max(self._min_zoom, min(self._max_zoom, cur * factor))
        actual = target / max(cur, 1e-12)
        self.scale(actual, actual)

    def _reset_pan_smoothing(self):
        self._pan_ema = QPointF(0.0, 0.0)
        self._pan_residual = QPointF(0.0, 0.0)

    def _apply_smooth_pan(self, pd):
        dx = float(pd.x())
        dy = float(pd.y())
        speed = math.hypot(dx, dy)

        # Nonlinear gain: precise at low speed, accelerated at high speed.
        if speed < 1.0:
            gain = 0.55
        elif speed < 6.0:
            gain = 0.75
        elif speed < 18.0:
            gain = 1.00
        else:
            gain = 1.25

        # EMA smoothing. At higher speed we reduce smoothing to keep responsiveness.
        alpha = 0.42 if speed < 8.0 else 0.62
        target_x = dx * gain
        target_y = dy * gain
        fx = alpha * target_x + (1.0 - alpha) * float(self._pan_ema.x())
        fy = alpha * target_y + (1.0 - alpha) * float(self._pan_ema.y())
        self._pan_ema = QPointF(fx, fy)

        # Accumulate sub-pixel remainder to avoid jitter/stair-stepping.
        tx = fx + float(self._pan_residual.x())
        ty = fy + float(self._pan_residual.y())
        sx = int(tx)
        sy = int(ty)
        self._pan_residual = QPointF(tx - sx, ty - sy)

        if sx != 0:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - sx)
        if sy != 0:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - sy)

    def zoom_in(self):
        if self._has_image:
            self._apply_zoom_factor(1.15)

    def zoom_out(self):
        if self._has_image:
            self._apply_zoom_factor(1 / 1.15)

    def set_roi_mode(self, enabled: bool):
        self._roi_mode = enabled
        self._shape_drawing = False
        self._drawing_points = []
        self._drawing_hover = None
        self._hover_vertex_idx = None
        self._drag_vertex_idx = None
        self._drag_rect_handle = None
        self._drag_move_roi = False
        self._drag_last_scene = None
        if enabled:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.unsetCursor()
        self._update_roi_graphics()

    def set_roi_type(self, roi_type: str):
        if roi_type not in (
            self.ROI_NONE,
            self.ROI_POLYGON,
            self.ROI_RECT,
            self.ROI_ELLIPSE,
        ):
            return
        self._roi_type = roi_type
        self.cancel_current_roi()
        self._update_roi_graphics()

    def roi_type(self) -> str:
        return self._roi_type

    def clear_roi(self, notify: bool = True):
        self._rois = []
        self._selected_idx = None
        self._drawing_points = []
        self._drawing_hover = None
        self._hover_vertex_idx = None
        self._shape_start = None
        self._shape_end = None
        self._shape_drawing = False
        self._drag_vertex_idx = None
        self._drag_rect_handle = None
        self._drag_move_roi = False
        self._drag_last_scene = None
        self._update_roi_graphics()
        if notify:
            self._notify_rois_changed()

    def cancel_current_roi(self):
        had_drawing = False
        if self._drawing_points:
            self._drawing_points = []
            self._drawing_hover = None
            self._hover_vertex_idx = None
            had_drawing = True

        if self._roi_type in (self.ROI_RECT, self.ROI_ELLIPSE) and self._shape_drawing:
            self._shape_drawing = False
            self._shape_start = None
            self._shape_end = None
            had_drawing = True

        self._drag_vertex_idx = None
        self._drag_rect_handle = None
        self._drag_move_roi = False
        self._drag_last_scene = None

        if had_drawing:
            self._update_roi_graphics()
            return

        # If no active drawing, delete the selected ROI
        if self._selected_idx is not None and 0 <= self._selected_idx < len(self._rois):
            del self._rois[self._selected_idx]
            if not self._rois:
                self._selected_idx = None
            else:
                self._selected_idx = min(self._selected_idx, len(self._rois) - 1)
            self._update_roi_graphics()
            self._notify_rois_changed()

    def clear_all_rois(self):
        self.clear_roi(notify=True)

    def get_rois(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._rois)

    def selected_roi(self) -> Optional[Dict[str, Any]]:
        if self._selected_idx is None:
            return None
        if 0 <= self._selected_idx < len(self._rois):
            return copy.deepcopy(self._rois[self._selected_idx])
        return None

    def selected_roi_index(self) -> Optional[int]:
        return self._selected_idx

    def set_rois(self, rois: List[Dict[str, Any]], selected_idx: Optional[int] = None):
        self._suppress_notify = True
        self.clear_roi(notify=False)
        self._rois = copy.deepcopy(rois)
        if selected_idx is not None and 0 <= selected_idx < len(self._rois):
            self._selected_idx = selected_idx
        elif self._rois:
            self._selected_idx = len(self._rois) - 1
        else:
            self._selected_idx = None
        self._update_roi_graphics()
        self._suppress_notify = False
        self._notify_rois_changed()

    def _notify_rois_changed(self):
        if self._suppress_notify:
            return
        if self.on_rois_changed:
            self.on_rois_changed(self.get_rois(), self._selected_idx)

    def _clamp_to_image(self, p: QPointF) -> QPointF:
        if self._pix_item.pixmap().isNull():
            return p
        rect = self._pix_item.boundingRect()
        x = min(max(p.x(), rect.left()), rect.right())
        y = min(max(p.y(), rect.top()), rect.bottom())
        return QPointF(x, y)

    def _find_snap_idx(self, points: List[QPointF], mouse_pos_view) -> Optional[int]:
        if not points:
            return None
        best_idx = None
        best_dist = None
        for i, p in enumerate(points):
            pv = self.mapFromScene(p)
            dx = pv.x() - mouse_pos_view.x()
            dy = pv.y() - mouse_pos_view.y()
            d = (dx * dx + dy * dy) ** 0.5
            if d <= self._snap_threshold_px and (best_dist is None or d < best_dist):
                best_dist = d
                best_idx = i
        return best_idx

    def _roi_path(self, roi: Dict[str, Any]) -> QPainterPath:
        path = QPainterPath()
        typ = roi.get("type")
        if typ == self.ROI_POLYGON:
            pts = roi.get("points", [])
            if len(pts) >= 3:
                first = QPointF(float(pts[0][0]), float(pts[0][1]))
                path.moveTo(first)
                for x, y in pts[1:]:
                    path.lineTo(QPointF(float(x), float(y)))
                path.closeSubpath()
        elif typ in (self.ROI_RECT, self.ROI_ELLIPSE):
            x = float(roi.get("x", 0.0))
            y = float(roi.get("y", 0.0))
            w = float(roi.get("w", 0.0))
            h = float(roi.get("h", 0.0))
            rect = QRectF(x, y, w, h).normalized()
            if typ == self.ROI_RECT:
                path.addRect(rect)
            else:
                path.addEllipse(rect)
        return path

    def _current_selected_points(self) -> List[QPointF]:
        if self._selected_idx is None or self._selected_idx >= len(self._rois):
            return []
        roi = self._rois[self._selected_idx]
        typ = roi.get("type")
        if typ == self.ROI_POLYGON:
            return [QPointF(float(x), float(y)) for x, y in roi.get("points", [])]
        if typ in (self.ROI_RECT, self.ROI_ELLIPSE):
            rect = QRectF(
                float(roi.get("x", 0.0)),
                float(roi.get("y", 0.0)),
                float(roi.get("w", 0.0)),
                float(roi.get("h", 0.0)),
            ).normalized()
            return [
                rect.topLeft(),
                rect.topRight(),
                rect.bottomRight(),
                rect.bottomLeft(),
                QPointF(rect.center().x(), rect.top()),
                QPointF(rect.right(), rect.center().y()),
                QPointF(rect.center().x(), rect.bottom()),
                QPointF(rect.left(), rect.center().y()),
            ]
        return []

    def _shape_rect(self) -> Optional[QRectF]:
        if self._shape_start is None or self._shape_end is None:
            return None
        return QRectF(self._shape_start, self._shape_end).normalized()

    def _set_rect(self, rect: QRectF):
        self._shape_start = rect.topLeft()
        self._shape_end = rect.bottomRight()

    def _point_segment_distance_scene(
        self, p: QPointF, a: QPointF, b: QPointF
    ) -> float:
        apx = p.x() - a.x()
        apy = p.y() - a.y()
        abx = b.x() - a.x()
        aby = b.y() - a.y()
        ab2 = abx * abx + aby * aby
        if ab2 <= 1e-9:
            return QLineF(p, a).length()
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
        proj = QPointF(a.x() + t * abx, a.y() + t * aby)
        return QLineF(p, proj).length()

    def _find_polygon_edge_idx(self, mouse_pos_view) -> Optional[int]:
        if self._selected_idx is None or self._selected_idx >= len(self._rois):
            return None
        roi = self._rois[self._selected_idx]
        if roi.get("type") != self.ROI_POLYGON:
            return None
        pts = [QPointF(float(x), float(y)) for x, y in roi.get("points", [])]
        if len(pts) < 3:
            return None
        scene_p = self.mapToScene(mouse_pos_view)
        best_idx = None
        best_d = None
        n = len(pts)
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            d_scene = self._point_segment_distance_scene(scene_p, a, b)
            d_view = d_scene * self.transform().m11()
            if d_view <= self._edge_hit_threshold_px and (
                best_d is None or d_view < best_d
            ):
                best_d = d_view
                best_idx = i
        return best_idx

    def _update_vertex_items(self):
        points: List[QPointF] = []
        style_type = None
        if (
            self._roi_mode
            and self._roi_type == self.ROI_POLYGON
            and self._drawing_points
        ):
            points = self._drawing_points
            style_type = self.ROI_POLYGON
        else:
            points = self._current_selected_points()
            if self._selected_idx is not None and self._selected_idx < len(self._rois):
                style_type = self._rois[self._selected_idx].get("type")

        while len(self._roi_vertex_items) < len(points):
            item = QGraphicsEllipseItem()
            item.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
            item.setZValue(12)
            self._scene.addItem(item)
            self._roi_vertex_items.append(item)
        while len(self._roi_vertex_items) > len(points):
            item = self._roi_vertex_items.pop()
            self._scene.removeItem(item)

        for i, p in enumerate(points):
            item = self._roi_vertex_items[i]
            is_hover = style_type == self.ROI_POLYGON and self._hover_vertex_idx == i
            radius = self._vertex_radius_hover if is_hover else self._vertex_radius
            if style_type == self.ROI_POLYGON:
                # Keep start-point visually distinct for easier closure.
                color = (
                    QColor(0, 220, 255)
                    if is_hover
                    else QColor(255, 120, 0 if i == 0 else 255)
                )
            else:
                color = QColor(90, 210, 255)
            item.setRect(-radius, -radius, 2 * radius, 2 * radius)
            item.setPos(p)
            item.setPen(QPen(Qt.black, 1))
            item.setBrush(QBrush(color))
            item.setVisible(True)

    def _update_roi_graphics(self):
        while len(self._roi_paths) < len(self._rois):
            item = QGraphicsPathItem()
            item.setZValue(10)
            self._scene.addItem(item)
            self._roi_paths.append(item)
        while len(self._roi_paths) > len(self._rois):
            item = self._roi_paths.pop()
            self._scene.removeItem(item)

        for i, roi in enumerate(self._rois):
            path = self._roi_path(roi)
            item = self._roi_paths[i]
            selected = self._selected_idx == i
            if selected:
                item.setPen(QPen(QColor(255, 210, 60), 2.2))
                item.setBrush(QBrush(QColor(255, 190, 0, 60)))
            else:
                item.setPen(QPen(QColor(140, 220, 255), 1.4))
                item.setBrush(QBrush(QColor(80, 170, 230, 30)))
            item.setPath(path)

        if (
            self._roi_mode
            and self._roi_type == self.ROI_POLYGON
            and self._drawing_points
            and self._drawing_hover is not None
        ):
            self._roi_preview.setLine(
                QLineF(self._drawing_points[-1], self._drawing_hover)
            )
            self._roi_preview.show()
        else:
            self._roi_preview.hide()

        dpath = QPainterPath()
        if self._drawing_points:
            dpath.moveTo(self._drawing_points[0])
            for p in self._drawing_points[1:]:
                dpath.lineTo(p)
        self._drawing_path.setPath(dpath)

        spath = QPainterPath()
        if (
            self._roi_mode
            and self._shape_drawing
            and self._shape_start is not None
            and self._shape_end is not None
        ):
            rect = QRectF(self._shape_start, self._shape_end).normalized()
            if self._roi_type == self.ROI_RECT:
                spath.addRect(rect)
            elif self._roi_type == self.ROI_ELLIPSE:
                spath.addEllipse(rect)
        self._shape_preview_path.setPath(spath)

        self._update_vertex_items()

    def mousePressEvent(self, event: QMouseEvent):
        if self._roi_mode and self._has_image:
            mouse = event.position().toPoint()
            scene_p = self._clamp_to_image(self.mapToScene(mouse))

            if event.button() == Qt.RightButton:
                hit = self._hit_test_roi(scene_p)
                if hit is not None:
                    if self._selected_idx == hit:
                        del self._rois[hit]
                        if self._selected_idx is not None:
                            self._selected_idx = (
                                min(hit, len(self._rois) - 1) if self._rois else None
                            )
                        self._update_roi_graphics()
                        self._notify_rois_changed()
                    else:
                        self._selected_idx = hit
                        self._update_roi_graphics()
                        self._notify_rois_changed()
                    event.accept()
                    return

            if event.button() == Qt.LeftButton:
                if self._selected_idx is not None and self._selected_idx < len(
                    self._rois
                ):
                    roi = self._rois[self._selected_idx]
                    if roi.get("type") == self.ROI_POLYGON:
                        pts = [
                            QPointF(float(x), float(y))
                            for x, y in roi.get("points", [])
                        ]
                        idx = self._find_snap_idx(pts, mouse)
                        if idx is not None:
                            self._drag_vertex_idx = idx
                            self._drag_last_scene = scene_p
                            event.accept()
                            return
                        edge_idx = self._find_polygon_edge_idx(mouse)
                        if edge_idx is not None:
                            pts.insert(edge_idx + 1, scene_p)
                            roi["points"] = [(p.x(), p.y()) for p in pts]
                            self._drag_vertex_idx = edge_idx + 1
                            self._newly_inserted_vertex = True
                            self._drag_last_scene = scene_p
                            self._update_roi_graphics()
                            event.accept()
                            return
                    if roi.get("type") in (self.ROI_RECT, self.ROI_ELLIPSE):
                        handle = self._hit_rect_handle(roi, mouse)
                        if handle is not None:
                            self._drag_rect_handle = handle
                            self._shape_drawing = True
                            self._drag_last_scene = scene_p
                            event.accept()
                            return
                    if self._roi_path(roi).contains(scene_p):
                        self._drag_move_roi = True
                        self._drag_last_scene = scene_p
                        event.accept()
                        return

                hit = self._hit_test_roi(scene_p)
                if hit is not None:
                    self._selected_idx = hit
                    self._update_roi_graphics()
                    self._notify_rois_changed()
                    event.accept()
                    return

                if self._roi_type == self.ROI_POLYGON:
                    snap_idx = self._find_snap_idx(self._drawing_points, mouse)
                    if snap_idx is not None:
                        scene_p = self._drawing_points[snap_idx]
                    if not self._drawing_points:
                        self._drawing_points.append(scene_p)
                    elif snap_idx == 0 and len(self._drawing_points) >= 3:
                        roi = {
                            "type": self.ROI_POLYGON,
                            "points": [(p.x(), p.y()) for p in self._drawing_points],
                        }
                        self._rois.append(roi)
                        self._selected_idx = len(self._rois) - 1
                        self._drawing_points = []
                        self._drawing_hover = None
                        self._hover_vertex_idx = None
                        self._notify_rois_changed()
                    else:
                        if QLineF(self._drawing_points[-1], scene_p).length() > 1e-6:
                            self._drawing_points.append(scene_p)
                    self._update_roi_graphics()
                    event.accept()
                    return

                if self._roi_type in (self.ROI_RECT, self.ROI_ELLIPSE):
                    self._shape_start = scene_p
                    self._shape_end = scene_p
                    self._shape_drawing = True
                    self._drag_rect_handle = None
                    self._update_roi_graphics()
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._roi_mode and self._has_image:
            mouse = event.position().toPoint()
            scene_p = self._clamp_to_image(self.mapToScene(mouse))
            if (
                self._drag_move_roi
                and self._selected_idx is not None
                and self._drag_last_scene is not None
            ):
                dx = scene_p.x() - self._drag_last_scene.x()
                dy = scene_p.y() - self._drag_last_scene.y()
                self._move_roi_by(self._selected_idx, dx, dy)
                self._drag_last_scene = scene_p
                self._update_roi_graphics()
                return
            if self._drag_vertex_idx is not None and self._selected_idx is not None:
                roi = self._rois[self._selected_idx]
                pts = [QPointF(float(x), float(y)) for x, y in roi.get("points", [])]
                if 0 <= self._drag_vertex_idx < len(pts):
                    pts[self._drag_vertex_idx] = scene_p
                    roi["points"] = [(p.x(), p.y()) for p in pts]
                    self._update_roi_graphics()
                return
            if self._drag_rect_handle is not None and self._selected_idx is not None:
                roi = self._rois[self._selected_idx]
                self._resize_roi_handle(roi, self._drag_rect_handle, scene_p)
                self._update_roi_graphics()
                return
            if self._shape_drawing and self._shape_start is not None:
                self._shape_end = scene_p
                self._update_roi_graphics()
                return
            if self._drawing_points:
                snap_idx = self._find_snap_idx(self._drawing_points, mouse)
                self._hover_vertex_idx = snap_idx
                self._drawing_hover = (
                    self._drawing_points[snap_idx] if snap_idx is not None else scene_p
                )
                self._update_roi_graphics()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._roi_mode and event.button() == Qt.LeftButton:
            if self._drag_move_roi:
                self._drag_move_roi = False
                self._drag_last_scene = None
                self._notify_rois_changed()
                event.accept()
                return
            if self._drag_vertex_idx is not None:
                self._drag_vertex_idx = None
                self._newly_inserted_vertex = False
                self._notify_rois_changed()
                event.accept()
                return
            if self._drag_rect_handle is not None:
                self._drag_rect_handle = None
                self._shape_drawing = False
                self._notify_rois_changed()
                event.accept()
                return
            if (
                self._shape_drawing
                and self._shape_start is not None
                and self._shape_end is not None
            ):
                rect = QRectF(self._shape_start, self._shape_end).normalized()
                self._shape_drawing = False
                if rect.width() >= 1.0 and rect.height() >= 1.0:
                    roi = {
                        "type": self._roi_type,
                        "x": rect.x(),
                        "y": rect.y(),
                        "w": rect.width(),
                        "h": rect.height(),
                    }
                    self._rois.append(roi)
                    self._selected_idx = len(self._rois) - 1
                    self._notify_rois_changed()
                self._shape_start = None
                self._shape_end = None
                self._update_roi_graphics()
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if self._roi_mode and self._drawing_points:
            self._hover_vertex_idx = None
            self._drawing_hover = None
            self._update_roi_graphics()
        super().leaveEvent(event)

    def _hit_test_roi(self, scene_p: QPointF) -> Optional[int]:
        for i in range(len(self._rois) - 1, -1, -1):
            if self._roi_path(self._rois[i]).contains(scene_p):
                return i
        return None

    def _rect_from_roi(self, roi: Dict[str, Any]) -> QRectF:
        return QRectF(
            float(roi.get("x", 0.0)),
            float(roi.get("y", 0.0)),
            float(roi.get("w", 0.0)),
            float(roi.get("h", 0.0)),
        ).normalized()

    def _hit_rect_handle(self, roi: Dict[str, Any], mouse_pos_view) -> Optional[str]:
        rect = self._rect_from_roi(roi)
        if rect.width() < 1.0 or rect.height() < 1.0:
            return None
        handles = {
            "tl": rect.topLeft(),
            "tr": rect.topRight(),
            "br": rect.bottomRight(),
            "bl": rect.bottomLeft(),
            "t": QPointF(rect.center().x(), rect.top()),
            "r": QPointF(rect.right(), rect.center().y()),
            "b": QPointF(rect.center().x(), rect.bottom()),
            "l": QPointF(rect.left(), rect.center().y()),
        }
        for name, hp in handles.items():
            vp = self.mapFromScene(hp)
            d = (
                (vp.x() - mouse_pos_view.x()) ** 2 + (vp.y() - mouse_pos_view.y()) ** 2
            ) ** 0.5
            if d <= self._snap_threshold_px:
                return name
        return None

    def _resize_roi_handle(self, roi: Dict[str, Any], handle: str, scene_p: QPointF):
        rect = self._rect_from_roi(roi)
        left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
        if handle in ("tl", "bl", "l"):
            left = scene_p.x()
        if handle in ("tr", "br", "r"):
            right = scene_p.x()
        if handle in ("tl", "tr", "t"):
            top = scene_p.y()
        if handle in ("bl", "br", "b"):
            bottom = scene_p.y()
        nr = QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()
        roi["x"], roi["y"], roi["w"], roi["h"] = nr.x(), nr.y(), nr.width(), nr.height()

    def _move_roi_by(self, idx: int, dx: float, dy: float):
        if not (0 <= idx < len(self._rois)):
            return
        roi = self._rois[idx]
        if roi.get("type") == self.ROI_POLYGON:
            roi["points"] = [
                (float(x) + dx, float(y) + dy) for x, y in roi.get("points", [])
            ]
        else:
            roi["x"] = float(roi.get("x", 0.0)) + dx
            roi["y"] = float(roi.get("y", 0.0)) + dy

    def nudge_selected_roi(self, dx: float, dy: float) -> bool:
        if self._selected_idx is None:
            return False
        self._move_roi_by(self._selected_idx, dx, dy)
        self._update_roi_graphics()
        self._notify_rois_changed()
        return True

    def wheelEvent(self, event: QWheelEvent):
        # Ctrl / Cmd + wheel/trackpad scroll => zoom.
        if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            self._reset_pan_smoothing()
            delta = event.angleDelta().y()
            if delta == 0:
                delta = event.pixelDelta().y()
            if delta > 0:
                # Smooth wheel scaling.
                self._apply_zoom_factor(1.0015 ** float(delta))
            elif delta < 0:
                self._apply_zoom_factor(1.0015 ** float(delta))
            event.accept()
            return

        # Ignore synthesized pinch-as-wheel events (no modifier), so gesture
        # zoom is effectively disabled and won't cause jitter.
        source = event.source()
        src_system = getattr(Qt, "MouseEventSynthesizedBySystem", None)
        if src_system is None and hasattr(Qt, "MouseEventSource"):
            src_system = Qt.MouseEventSource.MouseEventSynthesizedBySystem
        if (
            src_system is not None
            and source == src_system
            and event.modifiers() == Qt.NoModifier
            and event.pixelDelta().isNull()
            and event.angleDelta().y() != 0
        ):
            self._reset_pan_smoothing()
            event.accept()
            return

        # Trackpad two-finger pan in both browse and ROI modes.
        pd = event.pixelDelta()
        if not pd.isNull():
            self._apply_smooth_pan(pd)
            event.accept()
            return
        self._reset_pan_smoothing()
        super().wheelEvent(event)

    def event(self, e):
        return super().event(e)

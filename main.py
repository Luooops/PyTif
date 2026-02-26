import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import tifffile

from PySide6.QtCore import Qt, QSize, QTimer, QEvent, QSettings
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QKeyEvent,
    QColor,
    QWheelEvent,
    QPainter,
    QAction,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QToolButton,
    QMenu,
)

SUPPORTED_EXTS = (".tif", ".tiff")


# -------------------------
# Utilities
# -------------------------
def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def to_8bit_grayscale(img2d: np.ndarray) -> np.ndarray:
    x = img2d.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.uint8)

    vmin = np.percentile(x[finite], 1)
    vmax = np.percentile(x[finite], 99)
    if vmax <= vmin:
        vmax = vmin + 1

    y = (x - vmin) / (vmax - vmin)
    y = np.clip(y, 0, 1)
    return (y * 255).astype(np.uint8)


def numpy_to_qimage(img: np.ndarray) -> QImage:
    if img.ndim == 2:
        u8 = to_8bit_grayscale(img)
        h, w = u8.shape
        return QImage(u8.data, w, h, w, QImage.Format_Grayscale8).copy()
    raise ValueError(f"Unsupported image shape: {img.shape}")


def flatten_to_slices(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Return:
      flat: (H,W) if single slice OR (S,H,W) if multi-slice
      slices: number of slices
    """
    if arr.ndim == 2:
        return arr, 1

    h, w = arr.shape[-2:]
    s = int(np.prod(arr.shape[:-2]))
    return arr.reshape(s, h, w), s


# -------------------------
# Image Viewer (Zoom/Pan)
# -------------------------
class ImageViewer(QGraphicsView):
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
        self._pinch_last_scale = 1.0
        self.grabGesture(Qt.PinchGesture)

    def set_image(self, pixmap: QPixmap, fit: bool = True):
        self._pix_item.setPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self._has_image = not pixmap.isNull()
        if fit:
            self.fit_in_view()

    def fit_in_view(self):
        if not self._has_image:
            return
        self.resetTransform()
        self.fitInView(self._pix_item, Qt.KeepAspectRatio)

    def zoom_in(self):
        if self._has_image:
            self.scale(1.15, 1.15)

    def zoom_out(self):
        if self._has_image:
            self.scale(1 / 1.15, 1 / 1.15)

    def wheelEvent(self, event: QWheelEvent):
        # Ctrl / Cmd + wheel/trackpad scroll to zoom
        if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            delta = event.angleDelta().y()
            if delta == 0:
                delta = event.pixelDelta().y()
            if delta > 0:
                self.zoom_in()
            elif delta < 0:
                self.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)

    def event(self, e):
        # Pinch gesture (mac trackpad)
        if e.type() == QEvent.Gesture:
            pinch = e.gesture(Qt.PinchGesture)
            if pinch:
                scale_factor = pinch.scaleFactor()
                inc = scale_factor / max(self._pinch_last_scale, 1e-6)
                self.scale(inc, inc)
                self._pinch_last_scale = scale_factor
                return True
        return super().event(e)


# -------------------------
# Main Window
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyTIF Viewer")
        self.resize(1400, 850)
        self.setAcceptDrops(True)

        self.settings = QSettings("PyTIF", "Viewer")

        # Navigation state
        self.root_folder: Optional[str] = None     # root of current browsing context
        self.current_folder: Optional[str] = None  # currently displayed folder
        self.entries: List[Tuple[str, str]] = []   # ("up"/"dir"/"tif", path)

        # Image state
        self.loaded: Optional[np.ndarray] = None   # (H,W) or (S,H,W)
        self.total_slices: int = 1
        self.current_slice: int = 0

        # thumbs not included in this lean version (you already had it earlier)
        self._build_ui()
        self._build_menu()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        layout.addLayout(top)

        # File dropdown button
        self.btn_file = QToolButton()
        self.btn_file.setText("File")
        self.btn_file.setPopupMode(QToolButton.InstantPopup)
        self.file_dropdown = QMenu(self.btn_file)
        self.file_dropdown.addAction("Open File", self.open_file_dialog)
        self.file_dropdown.addAction("Open Folder", self.open_folder_dialog)
        self.btn_file.setMenu(self.file_dropdown)
        top.addWidget(self.btn_file)

        self.btn_sidebar = QPushButton("Hide Sidebar")
        self.btn_sidebar.clicked.connect(self.toggle_sidebar)
        top.addWidget(self.btn_sidebar)

        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_in = QPushButton("+")
        self.btn_fit = QPushButton("Fit")
        top.addWidget(self.btn_zoom_out)
        top.addWidget(self.btn_zoom_in)
        top.addWidget(self.btn_fit)

        self.status = QLabel("")
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.status, 1)

        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self.btn_fit.clicked.connect(self._fit)

        self.splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.splitter, 1)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_entry_selected)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.splitter.addWidget(self.list_widget)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(1, 1)

        self.viewer = ImageViewer()
        right_layout.addWidget(self.viewer, 1)

        # Slice controls (only for multi-slice)
        self.slice_controls = QWidget()
        sl = QHBoxLayout(self.slice_controls)
        sl.setContentsMargins(0, 0, 0, 0)

        self.slice_info = QLabel("")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_spin = QSpinBox()

        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_spin.setMinimum(1)
        self.slice_spin.setMaximum(1)

        sl.addWidget(self.slice_info, 1)
        sl.addWidget(self.slice_slider, 4)
        sl.addWidget(QLabel("Slice"))
        sl.addWidget(self.slice_spin)

        right_layout.addWidget(self.slice_controls)
        self.slice_controls.hide()

        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        self.slice_spin.valueChanged.connect(self.on_spin_changed)

    def _build_menu(self):
        # macOS standard menu
        file_menu = self.menuBar().addMenu("File")

        act_open_file = QAction("Open File…", self)
        act_open_file.setShortcut("Meta+O")  # ⌘O
        act_open_file.triggered.connect(self.open_file_dialog)
        file_menu.addAction(act_open_file)

        act_open_folder = QAction("Open Folder…", self)
        act_open_folder.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(act_open_folder)

        file_menu.addSeparator()

        act_close = QAction("Close Window", self)
        act_close.setShortcut("Meta+W")  # ⌘W
        act_close.triggered.connect(self.close)
        file_menu.addAction(act_close)

    # ---------------- Open ----------------
    def open_file_dialog(self):
        start = self.current_folder or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TIFF File",
            start,
            "TIFF files (*.tif *.tiff)",
        )
        if path:
            self.open_path(path)

    def open_folder_dialog(self):
        start = self.current_folder or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(
            self,
            "Open Folder",
            start,
        )
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        path = os.path.abspath(path)
        if os.path.isdir(path):
            # Folder chosen => root is this folder
            self.root_folder = path
            self.open_folder(path, select_first_tif=True)
            return

        # File chosen
        if os.path.isfile(path):
            self.open_single_file(path)
            return

        self.status.setText(f"Path does not exist: {path}")

    # ---------------- Folder browsing ----------------
    def open_single_file(self, path: str):
        path = os.path.abspath(path)
        folder = os.path.dirname(path)
        self.root_folder = folder
        self.current_folder = folder
        self.entries = [("tif", path)]

        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        item = QListWidgetItem(os.path.basename(path))
        item.setToolTip(path)
        self.list_widget.addItem(item)
        self.list_widget.setCurrentRow(0)
        self.list_widget.blockSignals(False)

        self.load_tiff(path)

    def open_folder(self, folder: str, select_first_tif: bool = False, select_path: Optional[str] = None):
        folder = os.path.abspath(folder)
        self.current_folder = folder

        try:
            names = os.listdir(folder)
        except Exception as e:
            self.status.setText(f"Failed to list folder: {folder} ({e})")
            return

        subdirs = sorted(
            [n for n in names if os.path.isdir(os.path.join(folder, n))],
            key=natural_key,
        )
        files = sorted(
            [n for n in names if os.path.isfile(os.path.join(folder, n)) and n.lower().endswith(SUPPORTED_EXTS)],
            key=natural_key,
        )

        self.entries = []
        # Add ".." (go up) if not at root
        if self.root_folder and os.path.abspath(folder) != os.path.abspath(self.root_folder):
            self.entries.append(("up", os.path.dirname(folder)))

        for d in subdirs:
            self.entries.append(("dir", os.path.join(folder, d)))
        for f in files:
            self.entries.append(("tif", os.path.join(folder, f)))

        # Refresh list
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        for typ, p in self.entries:
            if typ == "up":
                text = "⬅︎  .. (Back)"
            elif typ == "dir":
                text = f"📁  {os.path.basename(p)}"
            else:
                text = os.path.basename(p)

            item = QListWidgetItem(text)
            item.setToolTip(p)
            self.list_widget.addItem(item)

        self.list_widget.blockSignals(False)

        self.status.setText(f"Folder: {folder}")

        # Select item
        if select_path:
            target = os.path.abspath(select_path)
            for i, (typ, p) in enumerate(self.entries):
                if typ == "tif" and os.path.abspath(p) == target:
                    self.list_widget.setCurrentRow(i)
                    return
            # If not found, fallback
            select_first_tif = True

        if select_first_tif:
            for i, (typ, _) in enumerate(self.entries):
                if typ == "tif":
                    self.list_widget.setCurrentRow(i)
                    return
            # No tif in folder: keep selection at 0 if exists
            if self.entries:
                self.list_widget.setCurrentRow(0)

    def on_item_double_clicked(self, item: QListWidgetItem):
        row = self.list_widget.row(item)
        if row < 0 or row >= len(self.entries):
            return

        typ, path = self.entries[row]
        if typ in ("dir", "up"):
            self.open_folder(path, select_first_tif=True)
        elif typ == "tif":
            self.load_tiff(path)

    def on_entry_selected(self, row: int):
        if row < 0 or row >= len(self.entries):
            return
        typ, path = self.entries[row]
        if typ == "tif":
            self.load_tiff(path)

    # ---------------- TIFF load/render ----------------
    def load_tiff(self, path: str):
        try:
            arr = tifffile.imread(path)
            flat, slices = flatten_to_slices(arr)
        except Exception as e:
            self.status.setText(f"Failed to load {os.path.basename(path)}: {e}")
            return

        self.loaded = flat
        self.total_slices = slices
        self.current_slice = 0

        # Slice UI only if slices > 1
        if slices > 1:
            self.slice_controls.show()

            self.slice_slider.blockSignals(True)
            self.slice_spin.blockSignals(True)

            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(slices - 1)
            self.slice_slider.setValue(0)

            self.slice_spin.setMinimum(1)
            self.slice_spin.setMaximum(slices)
            self.slice_spin.setValue(1)

            self.slice_slider.blockSignals(False)
            self.slice_spin.blockSignals(False)
        else:
            self.slice_controls.hide()

        self._render(fit=True)
        self._update_slice_info(path)

    def _update_slice_info(self, path: str):
        name = os.path.basename(path)
        if self.total_slices > 1:
            self.slice_info.setText(f"{name} — slice {self.current_slice + 1}/{self.total_slices}")
        else:
            self.slice_info.setText(f"{name} — 2D")

        if self.current_folder:
            self.status.setText(f"Folder: {self.current_folder}  |  {name}")

    def _render(self, fit: bool = False):
        if self.loaded is None:
            return

        if self.loaded.ndim == 2:
            img = self.loaded
        else:
            img = self.loaded[self.current_slice]

        qimg = numpy_to_qimage(img)
        pix = QPixmap.fromImage(qimg)

        # Keep zoom when switching slices; fit only on new file / explicit fit
        self.viewer.set_image(pix, fit=fit)

    # ---------------- Slice ----------------
    def on_slice_changed(self, v: int):
        if self.loaded is None or self.total_slices <= 1:
            return

        self.current_slice = int(v)

        self.slice_spin.blockSignals(True)
        self.slice_spin.setValue(self.current_slice + 1)
        self.slice_spin.blockSignals(False)

        self._render(fit=False)
        self.slice_info.setText(f"{os.path.basename(self._current_tif_name())} — slice {self.current_slice + 1}/{self.total_slices}")

    def on_spin_changed(self, v: int):
        if self.loaded is None or self.total_slices <= 1:
            return
        self.slice_slider.setValue(int(v) - 1)

    def _current_tif_name(self) -> str:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.entries) and self.entries[row][0] == "tif":
            return self.entries[row][1]
        return ""

    # ---------------- Sidebar ----------------
    def toggle_sidebar(self):
        if self.list_widget.isVisible():
            self.list_widget.hide()
            self.btn_sidebar.setText("Show Sidebar")
        else:
            self.list_widget.show()
            self.btn_sidebar.setText("Hide Sidebar")

    # ---------------- Zoom helpers ----------------
    def _zoom_in(self):
        self.viewer.zoom_in()

    def _zoom_out(self):
        self.viewer.zoom_out()

    def _fit(self):
        self.viewer.fit_in_view()

    # ---------------- Drag & Drop ----------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if not path:
            return
        self.open_path(path)

    # ---------------- Keyboard ----------------
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()

        # Up/Down => file only (skip dirs/up)
        if key in (Qt.Key_Up, Qt.Key_Down):
            delta = 1 if key == Qt.Key_Down else -1
            self._move_to_prev_next_tif(delta)
            event.accept()
            return

        # Left/Right => slices only
        if key in (Qt.Key_Left, Qt.Key_Right):
            if self.loaded is not None and self.total_slices > 1:
                delta = 1 if key == Qt.Key_Right else -1
                new_slice = max(0, min(self.total_slices - 1, self.current_slice + delta))
                if new_slice != self.current_slice:
                    self.slice_slider.setValue(new_slice)
            event.accept()
            return

        super().keyPressEvent(event)

    def _move_to_prev_next_tif(self, delta: int):
        if not self.entries:
            return
        cur = self.list_widget.currentRow()
        if cur < 0:
            return

        idx = cur
        while True:
            idx += delta
            if idx < 0 or idx >= len(self.entries):
                break
            if self.entries[idx][0] == "tif":
                self.list_widget.setCurrentRow(idx)
                break


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


'''
1. 框选section，鼠标单击确认起始点，多边形区域选择
2. 图片contrast
3. multi-channel support，支持多channel同时显示
4. 支持框选区域的参数计算显示，面积，周长
5. nd2 support等（待商榷）
6. 找出多边形区域内最长的线段作为长方形的边（后续）
'''
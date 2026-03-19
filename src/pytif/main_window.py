import copy
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Set

import numpy as np
import qdarktheme
from PySide6.QtCore import (
    QDir,
    QModelIndex,
    QPointF,
    QSettings,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QIcon,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFileSystemModel,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from .constants import (
    APP_TITLE,
    BORDER_COLOR,
    CUSTOM_APP_STYLESHEET,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    ORG_DOMAIN,
    ORG_NAME,
    PRIMARY_COLOR,
    ROI_JSON_EXTENSION,
    SUPPORTED_EXTS,
)
from .file_filter_model import FileFilterModel
from .io_handler import (
    RenderWorker,
    TiffLoaderWorker,
    load_rois_from_json,
    save_rois_to_json,
    save_tiff_with_metadata,
)
from .roi import roi_geometry, roi_mask
from .utils import natural_key, numpy_to_qimage
from .widgets import DraggablePanel, ImageViewer, ROIListWindow


# -------------------------
# Main Window
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.setAcceptDrops(True)

        self.settings = QSettings(ORG_NAME, ORG_DOMAIN)

        # Navigation state
        self.current_folder: Optional[str] = None
        self.tree_root_path: Optional[str] = None
        self.opened_files: Set[str] = set()
        self.opened_folders: Set[str] = set()
        self.excluded_paths: Set[str] = set()
        self.active_file_path: Optional[str] = None

        # Image state
        self.auto_contrast = False
        self.loaded: Optional[np.ndarray] = None  # (H,W) or (S,H,W)
        self.total_slices: int = 1
        self.current_slice: int = 0
        self.rois_by_file: Dict[str, List[Dict[str, Any]]] = {}
        self.selected_roi_by_file: Dict[str, int] = {}
        self.calibrations_by_file: Dict[str, Dict[str, Any]] = {}
        self.undo_stack_by_file: Dict[str, List[Dict[str, Any]]] = {}
        self.redo_stack_by_file: Dict[str, List[Dict[str, Any]]] = {}
        self.changed_files: Set[str] = set()
        self._updating_rois_from_file = False
        self._updating_roi_list_ui = False
        self.loading_pool = QThreadPool.globalInstance()

        self.loading_timer = QTimer(self)
        self.loading_timer.setSingleShot(True)
        self.loading_timer.setInterval(1000)
        self.loading_timer.timeout.connect(self._show_loading_overlay)

        self._build_ui()
        self._build_menu()

    def _show_loading_overlay(self):
        self.viewer.loading_overlay.show()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        top = QHBoxLayout()
        layout.addLayout(top)

        # File dropdown button
        self.btn_file = QToolButton()
        self.btn_file.setFixedHeight(30)
        self.btn_file.setText("Open")
        if platform.system() == "Darwin":  # macOS
            self.btn_file.setPopupMode(QToolButton.InstantPopup)
        else:
            self.btn_file.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_file.clicked.connect(self.open_file_dialog)
        self.file_dropdown = QMenu(self.btn_file)
        self.file_dropdown.addAction("Open File(s)", self.open_file_dialog)
        self.file_dropdown.addAction("Open Folder(s)", self.open_folder_dialog)
        self.btn_file.setMenu(self.file_dropdown)
        top.addWidget(self.btn_file)

        # Close dropdown button
        self.btn_close = QToolButton()
        self.btn_close.setFixedHeight(30)
        self.btn_close.setText("Close")
        if platform.system() == "Darwin":  # macOS
            self.btn_close.setPopupMode(QToolButton.InstantPopup)
        else:
            self.btn_close.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_close.clicked.connect(self.close_current_entry)
        self.close_dropdown = QMenu(self.btn_close)
        self.close_dropdown.addAction("Close Current File", self.close_current_entry)
        self.close_dropdown.addAction("Close All", self.close_all_entries)
        self.btn_close.setMenu(self.close_dropdown)
        top.addWidget(self.btn_close)

        # Save dropdown button
        self.btn_save = QToolButton()
        self.btn_save.setFixedHeight(30)
        self.btn_save.setText("Save")
        if platform.system() == "Darwin":  # macOS
            self.btn_save.setPopupMode(QToolButton.InstantPopup)
        else:
            self.btn_save.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_save.clicked.connect(self.save_selected)
        self.save_dropdown = QMenu(self.btn_save)
        self.save_dropdown.addAction("Save Selected", self.save_selected)
        self.save_dropdown.addAction("Save All", self.save_all)
        self.save_dropdown.addAction("Save As ...", self.save_as)
        self.btn_save.setMenu(self.save_dropdown)
        top.addWidget(self.btn_save)

        self.btn_sidebar = QPushButton("Hide Sidebar")
        self.btn_sidebar.setFixedHeight(30)
        self.btn_sidebar.clicked.connect(self.toggle_sidebar)
        top.addWidget(self.btn_sidebar)

        self.status = QLabel("")
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.status, 1)

        self._build_roi_panel()

        self.splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.splitter, 1)

        self.file_model = QFileSystemModel(self)
        self.file_model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
        self.file_model.setRootPath("")

        self.file_proxy = FileFilterModel(self)
        self.file_proxy.setSourceModel(self.file_model)

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_proxy)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.sortByColumn(0, Qt.AscendingOrder)
        self.tree_view.setSelectionMode(QTreeView.ExtendedSelection)
        self.tree_view.setSelectionBehavior(QTreeView.SelectRows)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.on_tree_context_menu)
        self.tree_view.clicked.connect(self.on_tree_clicked)
        self.tree_view.doubleClicked.connect(self.on_tree_double_clicked)
        self.tree_view.selectionModel().currentChanged.connect(
            self.on_tree_current_changed
        )
        self.splitter.addWidget(self.tree_view)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(1, 1)

        self.viewer = ImageViewer()
        self.viewer.on_rois_changed = self._on_viewer_rois_changed
        self.viewer.on_scale_set = self._on_viewer_scale_set
        self.viewer.btn_zoom_in.clicked.connect(self._zoom_in)
        self.viewer.btn_zoom_out.clicked.connect(self._zoom_out)
        self.viewer.btn_fit.clicked.connect(self._fit)
        self.viewer.btn_roi.toggled.connect(self._toggle_roi_mode)
        self.viewer.btn_scale.toggled.connect(self._toggle_scale_mode)
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
        self._build_roi_list_window()
        self._build_roi_stats_panel(right_layout)

        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        self.slice_spin.valueChanged.connect(self.on_spin_changed)

    def _build_menu(self):
        # Platform-specific shortcuts
        is_mac = platform.system() == "Darwin"
        mod = "Meta" if is_mac else "Ctrl"

        file_menu = self.menuBar().addMenu("File")

        act_open_file = QAction("Open File(s)", self)
        act_open_file.setShortcut(f"{mod}+O")
        act_open_file.triggered.connect(self.open_file_dialog)
        file_menu.addAction(act_open_file)

        act_open_folder = QAction("Open Folder(s)", self)
        act_open_folder.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(act_open_folder)

        file_menu.addSeparator()

        act_save_selected = QAction("Save Selected", self)
        act_save_selected.setShortcut(f"{mod}+S")
        act_save_selected.triggered.connect(self.save_selected)
        file_menu.addAction(act_save_selected)

        act_save_all = QAction("Save All", self)
        act_save_all.triggered.connect(self.save_all)
        file_menu.addAction(act_save_all)

        act_save_as = QAction("Save As...", self)
        # Shift modifier is standard across platforms for "Save As"
        act_save_as.setShortcut(f"{mod}+Shift+S")
        act_save_as.triggered.connect(self.save_as)
        file_menu.addAction(act_save_as)

        file_menu.addSeparator()

        act_close_all = QAction("Close All", self)
        act_close_all.triggered.connect(self.close_all_entries)
        file_menu.addAction(act_close_all)

        file_menu.addSeparator()

        act_close = QAction("Close Window", self)
        act_close.setShortcut(f"{mod}+W")
        act_close.triggered.connect(self.close)
        file_menu.addAction(act_close)

        # Edit menu for Undo/Redo
        edit_menu = self.menuBar().addMenu("Edit")
        act_undo = QAction("Undo", self)
        act_undo.setShortcut(f"{mod}+Z")
        act_undo.triggered.connect(self._undo)
        edit_menu.addAction(act_undo)

        act_redo = QAction("Redo", self)
        act_redo.setShortcut(f"{mod}+Y")
        act_redo.triggered.connect(self._redo)
        edit_menu.addAction(act_redo)

        # Contrast menu
        contrast_menu = self.menuBar().addMenu("Contrast")
        self.act_auto_contrast = QAction("Auto Contrast", self)
        self.act_auto_contrast.setCheckable(True)
        self.act_auto_contrast.setChecked(self.auto_contrast)
        self.act_auto_contrast.triggered.connect(self._toggle_auto_contrast)
        contrast_menu.addAction(self.act_auto_contrast)

        # Preferences menu
        pref_menu = self.menuBar().addMenu("Preferences")
        theme_menu = pref_menu.addMenu("Theme")

        self.act_theme_auto = QAction("Follow System", self)
        self.act_theme_auto.setCheckable(True)
        self.act_theme_auto.triggered.connect(lambda: self._set_theme("auto"))

        self.act_theme_light = QAction("Light", self)
        self.act_theme_light.setCheckable(True)
        self.act_theme_light.triggered.connect(lambda: self._set_theme("light"))

        self.act_theme_dark = QAction("Dark", self)
        self.act_theme_dark.setCheckable(True)
        self.act_theme_dark.triggered.connect(lambda: self._set_theme("dark"))

        theme_menu.addAction(self.act_theme_auto)
        theme_menu.addAction(self.act_theme_light)
        theme_menu.addAction(self.act_theme_dark)

        # Set default state of menu actions
        current_theme = self.settings.value("theme", "auto")
        self.act_theme_auto.setChecked(current_theme == "auto")
        self.act_theme_light.setChecked(current_theme == "light")
        self.act_theme_dark.setChecked(current_theme == "dark")

    def _build_roi_panel(self):
        self.roi_panel = DraggablePanel(self)
        self.roi_panel.setFrameShape(QFrame.StyledPanel)
        self.roi_panel.setObjectName("roiPanel")
        panel_layout = QVBoxLayout(self.roi_panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(6)

        title = QLabel("ROI Tools")
        title.setCursor(Qt.SizeAllCursor)
        self.roi_panel.set_drag_handle(title)
        panel_layout.addWidget(title)

        self.lbl_keep_mode = QLabel("Keep ROI in Current File: On")
        panel_layout.addWidget(self.lbl_keep_mode)

        row = QHBoxLayout()

        self.btn_roi_rect = QToolButton()
        self.btn_roi_rect.setCheckable(True)
        self.btn_roi_rect.setIcon(self._make_roi_icon(ImageViewer.ROI_RECT))
        self.btn_roi_rect.setIconSize(QSize(20, 20))
        self.btn_roi_rect.setFixedSize(34, 34)
        self.btn_roi_rect.setToolTip("Rectangle ROI")

        self.btn_roi_ellipse = QToolButton()
        self.btn_roi_ellipse.setCheckable(True)
        self.btn_roi_ellipse.setIcon(self._make_roi_icon(ImageViewer.ROI_ELLIPSE))
        self.btn_roi_ellipse.setIconSize(QSize(20, 20))
        self.btn_roi_ellipse.setFixedSize(34, 34)
        self.btn_roi_ellipse.setToolTip("Ellipse ROI")

        self.btn_roi_poly = QToolButton()
        self.btn_roi_poly.setCheckable(True)
        self.btn_roi_poly.setIcon(self._make_roi_icon(ImageViewer.ROI_POLYGON))
        self.btn_roi_poly.setIconSize(QSize(20, 20))
        self.btn_roi_poly.setFixedSize(34, 34)
        self.btn_roi_poly.setToolTip("Polygon ROI")

        self.roi_type_group = QButtonGroup(self)
        self.roi_type_group.setExclusive(True)
        self.roi_type_group.addButton(self.btn_roi_rect)
        self.roi_type_group.addButton(self.btn_roi_ellipse)
        self.roi_type_group.addButton(self.btn_roi_poly)

        row.addWidget(self.btn_roi_rect)
        row.addWidget(self.btn_roi_ellipse)
        row.addWidget(self.btn_roi_poly)
        panel_layout.addLayout(row)

        self.btn_roi_rect.clicked.connect(
            lambda: self._select_roi_type(ImageViewer.ROI_RECT)
        )
        self.btn_roi_ellipse.clicked.connect(
            lambda: self._select_roi_type(ImageViewer.ROI_ELLIPSE)
        )
        self.btn_roi_poly.clicked.connect(
            lambda: self._select_roi_type(ImageViewer.ROI_POLYGON)
        )

        self.btn_clear_rois = QPushButton("Clear All ROIs")
        self.btn_clear_rois.clicked.connect(self._clear_current_file_rois)
        panel_layout.addWidget(self.btn_clear_rois)

        self.roi_panel.hide()

    def _make_roi_icon(self, roi_type: str) -> QIcon:
        pm = QPixmap(24, 24)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(235, 235, 235), 2))
        if roi_type == ImageViewer.ROI_RECT:
            p.drawRect(4, 5, 16, 14)
        elif roi_type == ImageViewer.ROI_ELLIPSE:
            p.drawEllipse(4, 5, 16, 14)
        else:
            poly = QPolygonF(
                [QPointF(5, 18), QPointF(8, 6), QPointF(18, 8), QPointF(19, 18)]
            )
            p.drawPolygon(poly)
        p.end()
        return QIcon(pm)

    def _build_roi_stats_panel(self, parent_layout: QVBoxLayout):
        self.roi_stats_panel = QFrame()
        self.roi_stats_panel.setObjectName("roiStatsPanel")
        stats_layout = QFormLayout(self.roi_stats_panel)
        stats_layout.setContentsMargins(8, 6, 8, 6)
        stats_layout.setLabelAlignment(Qt.AlignRight)
        stats_layout.setHorizontalSpacing(12)
        stats_layout.setVerticalSpacing(4)

        self.lbl_roi_type = QLabel("—")
        self.lbl_roi_count = QLabel("0")
        self.lbl_roi_coords = QLabel("—")
        self.lbl_roi_area = QLabel("—")
        self.lbl_roi_perimeter = QLabel("—")
        self.lbl_roi_pixels = QLabel("—")
        self.lbl_roi_mean = QLabel("—")
        self.lbl_roi_minmax = QLabel("—")
        self.lbl_roi_std = QLabel("—")

        self.lbl_area_head = QLabel("Area (px²)")
        self.lbl_perim_head = QLabel("Perimeter (px)")

        stats_layout.addRow("ROI Type", self.lbl_roi_type)
        stats_layout.addRow("ROIs in File", self.lbl_roi_count)
        stats_layout.addRow("Coordinates", self.lbl_roi_coords)
        stats_layout.addRow(self.lbl_area_head, self.lbl_roi_area)
        stats_layout.addRow(self.lbl_perim_head, self.lbl_roi_perimeter)
        stats_layout.addRow("Pixel Count", self.lbl_roi_pixels)
        stats_layout.addRow("Mean", self.lbl_roi_mean)
        stats_layout.addRow("Min / Max", self.lbl_roi_minmax)
        stats_layout.addRow("Std Dev", self.lbl_roi_std)

        parent_layout.addWidget(self.roi_stats_panel)

    def _build_roi_list_window(self):
        self.roi_window = ROIListWindow(self)
        self.roi_list_widget = self.roi_window.list_widget
        self.roi_list_widget.currentRowChanged.connect(
            self._on_roi_list_selection_changed
        )
        self.roi_list_widget.itemChanged.connect(self._on_roi_list_item_changed)
        self.roi_window.btn_save.clicked.connect(self._save_current_file_rois)
        self.roi_window.btn_load.clicked.connect(self._load_rois_into_current_file)
        self.roi_window.on_closed = self._on_roi_window_closed

    # ---------------- Open ----------------
    def open_file_dialog(self):
        start = self.current_folder or os.path.expanduser("~")
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open TIFF Files",
            start,
            "TIFF files (*.tif *.tiff)",
        )
        if paths:
            self.add_files(paths)

    def add_files(self, paths: List[str], auto_load: bool = True):
        new_paths: List[str] = []
        for p in paths:
            ap = os.path.abspath(p)
            if not os.path.isfile(ap):
                continue
            if not ap.lower().endswith(SUPPORTED_EXTS):
                continue
            if ap in self.opened_files:
                continue
            if any(self._is_under(ap, folder) for folder in self.opened_folders):
                continue
            self.opened_files.add(ap)
            new_paths.append(ap)

        if not new_paths:
            self.status.setText(
                "The selected file(s) are already opened or not valid TIFF files."
            )
            return

        self.current_folder = os.path.dirname(new_paths[0])
        self._rebuild_file_tree(
            preserve_selection=self.active_file_path or new_paths[0]
        )

        if auto_load and self.loaded is None:
            self._select_tree_path(new_paths[0])
            self.load_tiff(new_paths[0])
        else:
            self.status.setText(f"Added {len(new_paths)} file(s).")

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
            self.add_folder(path)
            return

        if os.path.isfile(path):
            self.add_files([path])
            return

        self.status.setText(f"Path does not exist: {path}")

    # ---------------- Close Logic ----------------
    def close_current_entry(self):
        path = self._path_from_proxy_index(self.tree_view.currentIndex())
        if not path:
            path = self.active_file_path
        if not path:
            return
        self.close_entries_by_paths([path])

    def close_selected_entries(self):
        paths = self._selected_tree_paths()
        if not paths:
            return
        self.close_entries_by_paths(paths)

    def close_entries_by_paths(self, paths: List[str]):
        # Ensure latest state is reflected in sidebar before closing
        self._refresh_roi_list()
        self._update_roi_stats()

        normalized_paths = {os.path.abspath(path) for path in paths}
        current_path = self.active_file_path

        # Explicitly remove from opened sets if they exist there
        explicit_files_to_remove = {
            path
            for path in self.opened_files
            if any(self._is_under_or_equal(path, p) for p in normalized_paths)
        }
        folder_roots_to_remove = {
            folder
            for folder in self.opened_folders
            if any(self._is_under_or_equal(folder, p) for p in normalized_paths)
        }

        # Any path that isn't a direct root but is being closed gets added to exclusions
        # if it's currently visible.
        new_exclusions = set()
        for path in normalized_paths:
            # If it's a sub-path of something already opened, exclude it
            if any(
                self._is_under(path, folder) for folder in self.opened_folders
            ) or any(self._is_under(path, f) for f in self.opened_files):
                new_exclusions.add(path)

        self.opened_files.difference_update(explicit_files_to_remove)
        self.opened_folders.difference_update(folder_roots_to_remove)
        self.excluded_paths.update(new_exclusions)

        # Optimization: Cleanup exclusions that are no longer relevant
        # (i.e. if their parent folder/file was also removed from opened sets)
        self.excluded_paths = {
            ex
            for ex in self.excluded_paths
            if any(self._is_under(ex, opened) for opened in self.opened_folders)
            or any(self._is_under(ex, opened) for opened in self.opened_files)
        }

        affected_files = {
            path
            for path in list(self.rois_by_file.keys())
            if any(self._is_under_or_equal(path, p) for p in normalized_paths)
        }
        for path in affected_files:
            self.rois_by_file.pop(path, None)
            self.selected_roi_by_file.pop(path, None)
            self.calibrations_by_file.pop(path, None)
            self.changed_files.discard(path)

        if not self.opened_files and not self.opened_folders:
            self._clear_loaded_image("All files closed.")
            self.excluded_paths.clear()
            self._rebuild_file_tree()
            return

        self._rebuild_file_tree(
            preserve_selection=(
                current_path if self._path_is_visible(current_path) else None
            )
        )

        if current_path and not self._path_is_visible(current_path):
            self._clear_loaded_image("")
            self.tree_view.clearSelection()
            return

        if current_path:
            self._select_tree_path(current_path)

    def close_all_entries(self):
        # Ensure latest state is reflected in sidebar before closing
        self._refresh_roi_list()
        self._update_roi_stats()

        self.status.setText("Closing all files...")
        QApplication.processEvents()

        self.opened_files.clear()
        self.opened_folders.clear()
        self.excluded_paths.clear()
        self.rois_by_file.clear()
        self.selected_roi_by_file.clear()
        self.calibrations_by_file.clear()
        self.changed_files.clear()
        self._clear_loaded_image("")
        self._rebuild_file_tree()
        self._refresh_roi_list()
        self._update_roi_stats()
        self.status.setText("All files closed.")

    # ---------------- Folder browsing ----------------
    def add_folder(self, folder: str):
        folder = os.path.abspath(folder)
        self.current_folder = folder

        self.status.setText(f"Scanning {os.path.basename(folder)} for TIFF files...")
        QApplication.processEvents()

        first_tif = self._find_first_tif_path(folder)
        if not first_tif:
            self.status.setText(
                f"No TIFF files found in {folder} or its subdirectories."
            )
            return

        self.opened_folders.add(folder)
        self._rebuild_file_tree(preserve_selection=self.active_file_path)
        self.status.setText(f"Opened folder: {os.path.basename(folder)}")

    def on_tree_context_menu(self, pos):
        index = self.tree_view.indexAt(pos)
        selected_paths = self._selected_tree_paths()
        if not index.isValid() or not selected_paths:
            return

        menu = QMenu()

        # Filter selected paths to find valid image files
        image_paths = [
            p
            for p in selected_paths
            if os.path.isfile(p) and p.lower().endswith(SUPPORTED_EXTS)
        ]

        if image_paths and self.active_file_path:
            current_calibration = self.calibrations_by_file.get(self.active_file_path)
            if current_calibration:
                apply_scale_action = menu.addAction(
                    f"Apply Current Scale to ({len(image_paths)})"
                )
                apply_scale_action.triggered.connect(
                    lambda: self.apply_current_scale_to_files(image_paths)
                )

        close_action = menu.addAction(f"Close Selected ({len(selected_paths)})")
        close_action.triggered.connect(self.close_selected_entries)
        menu.exec(self.tree_view.viewport().mapToGlobal(pos))

    def apply_current_scale_to_files(self, target_paths: List[str]):
        if not self.active_file_path:
            return

        current_calibration = self.calibrations_by_file.get(self.active_file_path)
        if not current_calibration:
            self.status.setText("No scale set for the current image.")
            return

        count = 0
        for path in target_paths:
            # Skip if it's the current file (already has the scale)
            if path == self.active_file_path:
                continue

            # Apply the calibration
            self.calibrations_by_file[path] = copy.deepcopy(current_calibration)
            self.changed_files.add(path)
            count += 1

        if count > 0:
            self.status.setText(f"Applied current scale to {count} file(s).")
        else:
            self.status.setText("No changes made.")

    def on_tree_current_changed(self, current: QModelIndex, previous: QModelIndex):
        pass

    def on_tree_clicked(self, index: QModelIndex):
        path = self._path_from_proxy_index(index)
        if path and os.path.isfile(path) and path.lower().endswith(SUPPORTED_EXTS):
            self.load_tiff(path)

    def on_tree_double_clicked(self, index: QModelIndex):
        path = self._path_from_proxy_index(index)
        if path and os.path.isfile(path) and path.lower().endswith(SUPPORTED_EXTS):
            self.load_tiff(path)

    # ---------------- TIFF load/render ----------------
    def load_tiff(self, path: str):
        path = os.path.abspath(path)
        if self.active_file_path == path:
            return

        self.status.setText(f"Loading {os.path.basename(path)}...")
        self.loading_timer.start()

        worker = TiffLoaderWorker(path, auto_contrast=self.auto_contrast)
        worker.signals.finished.connect(self._on_tiff_loaded)
        worker.signals.error.connect(self._on_tiff_load_error)
        self.loading_pool.start(worker)

    def _on_tiff_loaded(
        self,
        path: str,
        flat: np.ndarray,
        slices: int,
        calibration: Any,
        initial_pix: QPixmap,
    ):
        self.loading_timer.stop()
        self.viewer.loading_overlay.hide()
        self.loaded = flat
        self.total_slices = slices
        self.current_slice = 0
        self.active_file_path = path

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

        # Prefer manually set calibration from current session if it exists
        cached_calib = self.calibrations_by_file.get(path) if path else None
        final_calib = cached_calib if cached_calib is not None else calibration

        self.viewer.set_scale_calibration(final_calib)
        if path:
            self.calibrations_by_file[path] = (
                copy.deepcopy(final_calib) if final_calib else {}
            )
        self.viewer.set_image(initial_pix, fit=True)
        self._apply_rois_for_current_file()
        self._update_roi_stats()

        self.status.setText("")
        self._select_tree_path(path)
        self._update_slice_info(path)

    def _on_tiff_load_error(self, path: str, error_msg: str):
        self.loading_timer.stop()
        self.viewer.loading_overlay.hide()
        self.status.setText(f"Failed to load {os.path.basename(path)}: {error_msg}")

    def _update_slice_info(self, path: str):
        name = path
        if self.total_slices > 1:
            self.slice_info.setText(
                f"{name} — slice {self.current_slice + 1}/{self.total_slices}"
            )
        else:
            self.slice_info.setText(f"{name} — 2D")

        display_root = self.tree_root_path or self.current_folder
        if display_root:
            self.status.setText(f"File: {name}")

    def _render(self, fit: bool = False):
        if self.loaded is None:
            return

        if self.loaded.ndim == 2:
            img = self.loaded
        else:
            img = self.loaded[self.current_slice]

        self.loading_timer.start()
        worker = RenderWorker(img, self.auto_contrast, fit)
        worker.signals.finished.connect(self._on_render_finished)
        self.loading_pool.start(worker)

    def _on_render_finished(self, pix: QPixmap, fit: bool):
        self.loading_timer.stop()
        self.viewer.set_image(pix, fit=fit)
        self.viewer.loading_overlay.hide()

    # ---------------- Slice ----------------
    def on_slice_changed(self, v: int):
        if self.loaded is None or self.total_slices <= 1:
            return
        self.current_slice = int(v)
        self.slice_spin.blockSignals(True)
        self.slice_spin.setValue(self.current_slice + 1)
        self.slice_spin.blockSignals(False)
        self._render(fit=False)
        self._update_roi_stats()
        self.slice_info.setText(
            f"{os.path.basename(self._current_tif_name())} — slice {self.current_slice + 1}/{self.total_slices}"
        )

    def on_spin_changed(self, v: int):
        if self.loaded is None or self.total_slices <= 1:
            return
        self.slice_slider.setValue(int(v) - 1)

    def _current_tif_name(self) -> str:
        return self.active_file_path or ""

    # ---------------- Save Logic ----------------
    def save_selected(self):
        path = self.active_file_path
        if not path:
            self.status.setText("No active file to save.")
            return
        if self._save_file_internal(path):
            self.status.setText(f"Saved {os.path.basename(path)}")
            self.changed_files.discard(path)

    def save_all(self):
        if not self.changed_files:
            self.status.setText("No unsaved changes.")
            return

        count = 0
        to_save = list(self.changed_files)
        for path in to_save:
            if self._save_file_internal(path):
                count += 1
                self.changed_files.discard(path)
        self.status.setText(f"Saved {count} file(s).")

    def save_as(self):
        source_path = self.active_file_path
        if not source_path:
            self.status.setText("No active file to save.")
            return

        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save As",
            source_path,
            "TIFF files (*.tif *.tiff)",
        )
        if not target_path:
            return

        if self._save_file_internal(source_path, target_path):
            self.status.setText(f"Saved as {os.path.basename(target_path)}")

    def _save_file_internal(
        self, source_path: str, target_path: Optional[str] = None
    ) -> bool:
        # We need the original data. If it's the current active file, we have 'self.loaded'.
        # However, if it's 'save_all', we might need to reload or keep data in memory.
        # For this requirement, we'll assume we are saving what's in 'self.loaded' if it matches source_path,
        # otherwise we'd need a more complex state management.

        if source_path != self.active_file_path or self.loaded is None:
            # For simplicity in this implementation, we only support saving the active file's calibration.
            # A full implementation would likely store calibration in a dict and reload data if needed.
            # But here we'll try to use what we have.
            self.status.setText(
                f"Can only save active file for now: {os.path.basename(source_path)}"
            )
            return False

        path_to_write = target_path or source_path
        calib = self.calibrations_by_file.get(source_path, {})
        units_per_pixel = calib.get("x_units_per_pixel", 1.0)
        unit_label = calib.get("unit_label", "px")

        try:
            save_tiff_with_metadata(
                path_to_write, self.loaded, units_per_pixel, unit_label
            )
            return True
        except Exception as e:
            self.status.setText(
                f"Failed to save {os.path.basename(path_to_write)}: {e}"
            )
            return False

    # ---------------- Sidebar ----------------
    def toggle_sidebar(self):
        if self.tree_view.isVisible():
            self.tree_view.hide()
            self.btn_sidebar.setText("Show Sidebar")
        else:
            self.tree_view.show()
            self.btn_sidebar.setText("Hide Sidebar")

    # ---------------- Zoom helpers ----------------
    def _zoom_in(self):
        self.viewer.zoom_in()

    def _zoom_out(self):
        self.viewer.zoom_out()

    def _toggle_auto_contrast(self, checked: bool):
        self.auto_contrast = checked
        if self.loaded is not None:
            self._render(fit=False)

    def _fit(self):
        self.viewer.fit_in_view()

    def _toggle_roi_mode(self, enabled: bool):
        self.viewer.set_roi_mode(enabled)
        # Ensure buttons stay in sync
        if self.viewer.btn_roi.isChecked() != enabled:
            self.viewer.btn_roi.blockSignals(True)
            self.viewer.btn_roi.setChecked(enabled)
            self.viewer.btn_roi.blockSignals(False)

        if enabled:
            if self.viewer.btn_scale.isChecked():
                self.viewer.btn_scale.setChecked(False)
            self._sync_roi_type_buttons()
            self._show_roi_panel()
            self._show_roi_window()
            self.status.setText(
                "ROI mode: choose rectangle, ellipse, or polygon from the floating panel. Press Esc to delete selected ROI."
            )
        else:
            self._hide_roi_panel()
            self._hide_roi_window()
            self.btn_roi_rect.setChecked(False)
            self.btn_roi_ellipse.setChecked(False)
            self.btn_roi_poly.setChecked(False)
            # Restore status text to show current image/folder info
            path = self._current_file_path()
            if path:
                self._update_slice_info(path)
            else:
                self.status.setText("")

    def _show_roi_panel(self):
        self.roi_panel.adjustSize()
        if not self.roi_panel.user_moved or not self.roi_panel.isVisible():
            # map the viewer overlay button's coordinate to the main window's coordinate space
            global_pos = self.viewer.overlay_panel.mapToGlobal(
                self.viewer.btn_roi.pos()
            )
            local_pos = self.mapFromGlobal(global_pos)

            x = local_pos.x() - self.roi_panel.width() - 10
            y = local_pos.y()
            self.roi_panel.move(x, y)
        self._clamp_roi_panel_pos()
        self.roi_panel.show()
        self.roi_panel.raise_()

    def _hide_roi_panel(self):
        self.roi_panel.hide()

    def _show_roi_window(self):
        if not self.roi_window.isVisible():
            p = self.roi_window.pos()
            if p.x() == 0 and p.y() == 0:
                g = self.geometry()
                self.roi_window.move(g.right() + 12, g.top() + 60)
        self.roi_window.show()
        self.roi_window.raise_()
        self.roi_window.activateWindow()

    def _hide_roi_window(self):
        self.roi_window.hide_programmatically()

    def _on_roi_window_closed(self):
        if self.viewer.btn_roi.isChecked():
            self.viewer.btn_roi.setChecked(False)

    def _clamp_roi_panel_pos(self):
        max_x = max(0, self.width() - self.roi_panel.width() - 8)
        max_y = max(0, self.height() - self.roi_panel.height() - 8)
        x = max(0, min(self.roi_panel.x(), max_x))
        y = max(0, min(self.roi_panel.y(), max_y))
        self.roi_panel.move(x, y)

    def _sync_roi_type_buttons(self):
        t = self.viewer.roi_type()
        self.btn_roi_rect.setChecked(t == ImageViewer.ROI_RECT)
        self.btn_roi_ellipse.setChecked(t == ImageViewer.ROI_ELLIPSE)
        self.btn_roi_poly.setChecked(t == ImageViewer.ROI_POLYGON)

    def _select_roi_type(self, roi_type: str):
        if not self.viewer.btn_roi.isChecked():
            self.viewer.btn_roi.setChecked(True)
        self.viewer.set_roi_type(roi_type)
        labels = {
            ImageViewer.ROI_RECT: "Rectangle",
            ImageViewer.ROI_ELLIPSE: "Ellipse",
            ImageViewer.ROI_POLYGON: "Polygon",
        }
        self.status.setText(
            f"ROI mode: {labels.get(roi_type, '')}. Press Esc to delete selected ROI."
        )

    def _push_undo_state(self, action_name: str, clear_redo: bool = True):
        path = self._current_file_path()
        if not path:
            return

        state = {
            "action": action_name,
            "rois": copy.deepcopy(self.rois_by_file.get(path, [])),
            "selected_roi": self.selected_roi_by_file.get(path),
            "calibration": copy.deepcopy(self.calibrations_by_file.get(path, {})),
        }

        if path not in self.undo_stack_by_file:
            self.undo_stack_by_file[path] = []

        self.undo_stack_by_file[path].append(state)
        # Limit undo stack size
        if len(self.undo_stack_by_file[path]) > 50:
            self.undo_stack_by_file[path].pop(0)

        if clear_redo and path in self.redo_stack_by_file:
            self.redo_stack_by_file[path].clear()

    def _undo(self):
        path = self._current_file_path()
        if not path or not self.undo_stack_by_file.get(path):
            self.status.setText("Nothing to undo.")
            return

        # Prepare redo state from current before applying undo
        redo_state = {
            "action": self.undo_stack_by_file[path][-1]["action"],
            "rois": copy.deepcopy(self.rois_by_file.get(path, [])),
            "selected_roi": self.selected_roi_by_file.get(path),
            "calibration": copy.deepcopy(self.calibrations_by_file.get(path, {})),
        }
        if path not in self.redo_stack_by_file:
            self.redo_stack_by_file[path] = []
        self.redo_stack_by_file[path].append(redo_state)

        state = self.undo_stack_by_file[path].pop()

        self.rois_by_file[path] = state["rois"]
        self.selected_roi_by_file[path] = state["selected_roi"]
        self.calibrations_by_file[path] = state["calibration"]

        self.viewer.set_scale_calibration(self.calibrations_by_file[path])
        self._apply_rois_for_current_file()
        self._update_roi_stats()
        self.status.setText(f"Undo successful: {state['action']}")

    def _redo(self):
        path = self._current_file_path()
        if not path or not self.redo_stack_by_file.get(path):
            self.status.setText("Nothing to redo.")
            return

        state = self.redo_stack_by_file[path].pop()

        # Push current state to undo stack before redoing, but DON'T clear redo stack
        self._push_undo_state(state["action"], clear_redo=False)

        self.rois_by_file[path] = state["rois"]
        self.selected_roi_by_file[path] = state["selected_roi"]
        self.calibrations_by_file[path] = state["calibration"]

        self.viewer.set_scale_calibration(self.calibrations_by_file[path])
        self._apply_rois_for_current_file()
        self._update_roi_stats()
        self.status.setText(f"Redo successful: {state['action']}")

    def _current_file_path(self) -> Optional[str]:
        return self.active_file_path

    def _on_viewer_rois_changed(
        self, rois: List[Dict[str, Any]], selected_idx: Optional[int]
    ):
        if self._updating_rois_from_file:
            return
        path = self._current_file_path()
        if not path:
            return

        # Determine if we should push to undo stack
        # We push if ROI count changed or if a ROI was moved (rois content changed)
        # Note: selected_idx changes usually don't need undo, but they are part of the state.
        old_rois = self.rois_by_file.get(path, [])
        if rois != old_rois:
            action = "Modify ROI"
            if len(rois) > len(old_rois):
                action = "Add ROI"
            elif len(rois) < len(old_rois):
                action = "Remove ROI"

            self._push_undo_state(action)

        self._ensure_roi_ids_and_names(path, rois)
        self.rois_by_file[path] = copy.deepcopy(rois)
        if selected_idx is None:
            self.selected_roi_by_file.pop(path, None)
        else:
            self.selected_roi_by_file[path] = int(selected_idx)
        self._refresh_roi_list()
        self._update_roi_stats()

    def _toggle_scale_mode(self, enabled: bool):
        self.viewer.set_scale_line_mode(enabled)
        if enabled:
            # Disable ROI mode if scale mode is enabled
            if self.viewer.btn_roi.isChecked():
                self.viewer.btn_roi.setChecked(False)
            self.status.setText(
                "Scale mode: Click and drag a line over an object of known size."
            )
        else:
            path = self._current_file_path()
            if path:
                self._update_slice_info(path)
            else:
                self.status.setText("")

    def _on_viewer_scale_set(self, pixel_dist: float, known_dist: float, unit: str):
        path = self._current_file_path()
        if not path:
            return

        self._push_undo_state("Set Scale")

        units_per_pixel = known_dist / pixel_dist
        calibration = {
            "x_units_per_pixel": units_per_pixel,
            "y_units_per_pixel": units_per_pixel,
            "unit_label": unit,
        }
        self.viewer.set_scale_calibration(calibration)
        self.calibrations_by_file[path] = calibration
        self.changed_files.add(path)
        self._update_roi_stats()
        self.status.setText(f"Scale set: 1 pixel = {units_per_pixel:.4g} {unit}")

    def _apply_rois_for_current_file(self):
        path = self._current_file_path()
        rois = copy.deepcopy(self.rois_by_file.get(path, [])) if path else []
        if path:
            self._ensure_roi_ids_and_names(path, rois)
            self.rois_by_file[path] = copy.deepcopy(rois)
        sel = self.selected_roi_by_file.get(path) if path else None
        self._updating_rois_from_file = True
        self.viewer.set_rois(rois, sel)
        self._updating_rois_from_file = False
        self._refresh_roi_list()
        self._sync_roi_type_buttons()

    def _clear_current_file_rois(self):
        path = self._current_file_path()
        if not path:
            return

        if self.rois_by_file.get(path):
            self._push_undo_state("Clear ROIs")

        self.rois_by_file[path] = []
        self.selected_roi_by_file.pop(path, None)
        self._apply_rois_for_current_file()
        self._update_roi_stats()

    def _ensure_roi_ids_and_names(self, path: str, rois: List[Dict[str, Any]]):
        next_id = 1
        for roi in rois:
            rid = roi.get("_id")
            if isinstance(rid, int) and rid > 0:
                next_id = max(next_id, rid + 1)

        for roi in rois:
            rid = roi.get("_id")
            if not (isinstance(rid, int) and rid > 0):
                roi["_id"] = next_id
                rid = next_id
                next_id += 1
            name = str(roi.get("name", "")).strip()
            if not name:
                roi["name"] = str(rid)

    def _refresh_roi_list(self):
        path = self._current_file_path()
        rois = self.rois_by_file.get(path, []) if path else []
        sel = self.selected_roi_by_file.get(path) if path else None

        self._updating_roi_list_ui = True
        self.roi_list_widget.clear()
        for i, roi in enumerate(rois):
            rid = roi.get("_id")
            default_name = str(rid) if isinstance(rid, int) and rid > 0 else str(i + 1)
            name = str(roi.get("name", default_name))
            typ = str(roi.get("type", ""))
            rid_text = str(rid) if isinstance(rid, int) and rid > 0 else "?"
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setData(Qt.UserRole, i)
            item.setToolTip(f"ID {rid_text} · {name} ({typ})")
            self.roi_list_widget.addItem(item)
        if sel is not None and 0 <= sel < self.roi_list_widget.count():
            self.roi_list_widget.setCurrentRow(sel)
        self._updating_roi_list_ui = False

    def _on_roi_list_selection_changed(self, row: int):
        if self._updating_roi_list_ui:
            return
        path = self._current_file_path()
        if not path:
            return
        rois = copy.deepcopy(self.rois_by_file.get(path, []))
        sel = row if 0 <= row < len(rois) else None
        if sel is None:
            self.selected_roi_by_file.pop(path, None)
        else:
            self.selected_roi_by_file[path] = sel
        self._updating_rois_from_file = True
        self.viewer.set_rois(rois, sel)
        self._updating_rois_from_file = False
        self._update_roi_stats()

    def _on_roi_list_item_changed(self, item: QListWidgetItem):
        if self._updating_roi_list_ui:
            return
        path = self._current_file_path()
        if not path:
            return
        idx = self.roi_list_widget.row(item)
        rois = self.rois_by_file.get(path, [])
        if not (0 <= idx < len(rois)):
            return
        rid = rois[idx].get("_id", idx + 1)
        name = item.text().strip() or str(rid)
        rois[idx]["name"] = name
        item.setText(name)
        item.setToolTip(f"ID {rid} · {name} ({rois[idx].get('type', '')})")
        sel = self.selected_roi_by_file.get(path)
        self._updating_rois_from_file = True
        self.viewer.set_rois(copy.deepcopy(rois), sel)
        self._updating_rois_from_file = False

    def _save_current_file_rois(self):
        path = self._current_file_path()
        if not path:
            self.status.setText("No active TIFF file for ROI save.")
            return
        rois = self.rois_by_file.get(path, [])

        default_name = os.path.splitext(os.path.basename(path))[0] + ROI_JSON_EXTENSION
        default_path = os.path.join(os.path.dirname(path), default_name)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROI File",
            default_path,
            "ROI JSON (*.json)",
        )
        if not save_path:
            return

        try:
            count = save_rois_to_json(save_path, path, rois)
        except Exception as e:
            self.status.setText(f"Failed to save ROI file: {e}")
            return
        self.status.setText(f"Saved {count} ROI(s) to {os.path.basename(save_path)}")

    def _load_rois_into_current_file(self):
        path = self._current_file_path()
        if not path:
            self.status.setText("No active TIFF file for ROI load.")
            return
        load_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load ROI File",
            os.path.dirname(path),
            "ROI JSON (*.json)",
        )
        if not load_path:
            return
        try:
            rois = load_rois_from_json(load_path)
        except Exception as e:
            self.status.setText(f"Failed to load ROI file: {e}")
            return

        self.rois_by_file[path] = rois
        self.selected_roi_by_file[path] = 0 if rois else None
        if not rois:
            self.selected_roi_by_file.pop(path, None)
        self._apply_rois_for_current_file()
        self._update_roi_stats()
        self.status.setText(
            f"Loaded {len(rois)} ROI(s) from {os.path.basename(load_path)}"
        )

    def _current_slice_image(self) -> Optional[np.ndarray]:
        if self.loaded is None:
            return None
        if self.loaded.ndim == 2:
            return self.loaded
        return self.loaded[self.current_slice]

    def _update_roi_stats(self):
        path = self._current_file_path()
        rois = self.rois_by_file.get(path, []) if path else []
        self.lbl_roi_count.setText(str(len(rois)))
        state = self.viewer.selected_roi()

        calib = self.viewer._scale_calibration
        units_per_pixel = calib.get("x_units_per_pixel", 1.0) if calib else 1.0
        unit_label = calib.get("unit_label", "px") if calib else "px"

        if unit_label == "px":
            self.lbl_area_head.setText("Area (px²)")
            self.lbl_perim_head.setText("Perimeter (px)")
        else:
            self.lbl_area_head.setText(f"Area ({unit_label}²)")
            self.lbl_perim_head.setText(f"Perimeter ({unit_label})")

        if state is None:
            for lbl in [
                self.lbl_roi_type,
                self.lbl_roi_coords,
                self.lbl_roi_area,
                self.lbl_roi_perimeter,
                self.lbl_roi_pixels,
                self.lbl_roi_mean,
                self.lbl_roi_minmax,
                self.lbl_roi_std,
            ]:
                lbl.setText("—")
            return

        typ_map = {
            "polygon": "Polygon",
            "rect": "Rectangle",
            "ellipse": "Ellipse",
        }
        self.lbl_roi_type.setText(typ_map.get(state.get("type", ""), "Unknown"))
        typ = state.get("type")
        if typ in ("rect", "ellipse"):
            x = float(state.get("x", 0.0))
            y = float(state.get("y", 0.0))
            w = float(state.get("w", 0.0))
            h = float(state.get("h", 0.0))
            self.lbl_roi_coords.setText(f"x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
        elif typ == "polygon":
            pts = state.get("points", [])
            self.lbl_roi_coords.setText(f"{len(pts)} vertices")
        else:
            self.lbl_roi_coords.setText("—")

        area_px, perimeter_px = roi_geometry(state)
        area_scaled = area_px * (units_per_pixel**2)
        perimeter_scaled = perimeter_px * units_per_pixel

        self.lbl_roi_area.setText(f"{area_scaled:.2f}")
        self.lbl_roi_perimeter.setText(f"{perimeter_scaled:.2f}")

        img = self._current_slice_image()
        if img is None:
            for lbl in [
                self.lbl_roi_pixels,
                self.lbl_roi_mean,
                self.lbl_roi_minmax,
                self.lbl_roi_std,
            ]:
                lbl.setText("—")
            return

        mask = roi_mask(state, img.shape)
        vals = img[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            self.lbl_roi_pixels.setText("0")
            for lbl in [self.lbl_roi_mean, self.lbl_roi_minmax, self.lbl_roi_std]:
                lbl.setText("—")
            return

        self.lbl_roi_pixels.setText(str(int(vals.size)))
        self.lbl_roi_mean.setText(f"{float(np.mean(vals)):.4g}")
        self.lbl_roi_minmax.setText(
            f"{float(np.min(vals)):.4g} / {float(np.max(vals)):.4g}"
        )
        self.lbl_roi_std.setText(f"{float(np.std(vals)):.4g}")

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
        mod = event.modifiers()

        # Handle Undo/Redo shortcuts
        is_mac = platform.system() == "Darwin"
        accel_mod = Qt.MetaModifier if is_mac else Qt.ControlModifier

        if mod & accel_mod:
            if key == Qt.Key_Z:
                if mod & Qt.ShiftModifier:
                    self._redo()
                else:
                    self._undo()
                event.accept()
                return
            if not is_mac and key == Qt.Key_Y:
                self._redo()
                event.accept()
                return

        if self.viewer.btn_roi.isChecked():
            if key == Qt.Key_Left and self.viewer.nudge_selected_roi(-1, 0):
                event.accept()
                return
            if key == Qt.Key_Right and self.viewer.nudge_selected_roi(1, 0):
                event.accept()
                return
            if key == Qt.Key_Up and self.viewer.nudge_selected_roi(0, -1):
                event.accept()
                return
            if key == Qt.Key_Down and self.viewer.nudge_selected_roi(0, 1):
                event.accept()
                return

        if key in (Qt.Key_Up, Qt.Key_Down):
            delta = 1 if key == Qt.Key_Down else -1
            self._move_to_prev_next_tif(delta)
            event.accept()
            return

        if key in (Qt.Key_Left, Qt.Key_Right):
            if self.loaded is not None and self.total_slices > 1:
                delta = 1 if key == Qt.Key_Right else -1
                new_slice = max(
                    0, min(self.total_slices - 1, self.current_slice + delta)
                )
                if new_slice != self.current_slice:
                    self.slice_slider.setValue(new_slice)
            event.accept()
            return

        if key == Qt.Key_Escape and self.viewer.btn_roi.isChecked():
            # Force focus back to main window or viewer to ensure key events are captured correctly
            self.setFocus()
            self.viewer.cancel_current_roi()
            event.accept()
            return

        if key == Qt.Key_Plus or key == Qt.Key_Equal:
            self._zoom_in()
            event.accept()
            return

        if key == Qt.Key_Minus:
            self._zoom_out()
            event.accept()
            return

        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "roi_panel") and self.roi_panel.isVisible():
            self._show_roi_panel()

    def _move_to_prev_next_tif(self, delta: int):
        paths = self._visible_tif_paths()
        if not paths or not self.active_file_path:
            return
        try:
            current_index = paths.index(self.active_file_path)
        except ValueError:
            return
        new_index = max(0, min(len(paths) - 1, current_index + delta))
        if new_index == current_index:
            return
        path = paths[new_index]
        self._select_tree_path(path)
        self.load_tiff(path)

    def _rebuild_file_tree(self, preserve_selection: Optional[str] = None):
        self.tree_root_path = self._compute_tree_root_path()
        view_root_path = self._compute_tree_view_root_path()
        self.file_proxy.set_sources(
            self.opened_files,
            self.opened_folders,
            self.excluded_paths,
            self.tree_root_path,
            view_root_path,
        )

        if not self.tree_root_path:
            self.tree_view.setRootIndex(QModelIndex())
            self.tree_view.clearSelection()
            return

        source_root = self.file_model.index(view_root_path)
        proxy_root = self.file_proxy.mapFromSource(source_root)
        self.tree_view.setRootIndex(proxy_root)
        for column in range(1, self.file_model.columnCount()):
            self.tree_view.hideColumn(column)
        self.tree_view.expand(proxy_root)
        if view_root_path != self.tree_root_path:
            self._expand_tree_path(self.tree_root_path)

        selected_path = preserve_selection or self.active_file_path
        if selected_path and self._path_is_visible(selected_path):
            self._select_tree_path(selected_path)
        else:
            self.tree_view.clearSelection()

    def _compute_tree_root_path(self) -> Optional[str]:
        candidates: List[str] = []
        candidates.extend(self.opened_folders)
        candidates.extend(os.path.dirname(path) for path in self.opened_files)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        try:
            return os.path.commonpath(candidates)
        except ValueError:
            drive, _ = os.path.splitdrive(candidates[0])
            return drive + os.sep if drive else os.sep

    def _compute_tree_view_root_path(self) -> str:
        if len(self.opened_folders) == 1 and not self.opened_files:
            folder = next(iter(self.opened_folders))
            parent = os.path.dirname(folder)
            return parent or folder
        return self.tree_root_path or ""

    def _select_tree_path(self, path: Optional[str]):
        if not path or not self.tree_root_path:
            return
        self._expand_tree_path(path)
        source_index = self.file_model.index(path)
        proxy_index = self.file_proxy.mapFromSource(source_index)
        if not proxy_index.isValid():
            return
        selection_model = self.tree_view.selectionModel()
        if selection_model is not None:
            selection_model.blockSignals(True)
        self.tree_view.setCurrentIndex(proxy_index)
        self.tree_view.scrollTo(proxy_index)
        if selection_model is not None:
            selection_model.blockSignals(False)

    def _expand_tree_path(self, path: Optional[str]):
        if not path:
            return
        current_path = os.path.abspath(path)
        while current_path:
            source_index = self.file_model.index(current_path)
            proxy_index = self.file_proxy.mapFromSource(source_index)
            if proxy_index.isValid():
                self.tree_view.expand(proxy_index)
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:
                break
            current_path = parent_path

    def _path_from_proxy_index(self, index: QModelIndex) -> Optional[str]:
        if not index.isValid():
            return None
        source_index = self.file_proxy.mapToSource(index)
        if not source_index.isValid():
            return None
        path = self.file_model.filePath(source_index)
        return os.path.abspath(path) if path else None

    def _selected_tree_paths(self) -> List[str]:
        selection_model = self.tree_view.selectionModel()
        if selection_model is None:
            return []
        paths = []
        seen = set()
        for index in selection_model.selectedRows(0):
            path = self._path_from_proxy_index(index)
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
        return paths

    def _path_is_visible(self, path: Optional[str]) -> bool:
        if not path or not self.tree_root_path:
            return False
        normalized = os.path.abspath(path)
        if os.path.isdir(normalized):
            return any(
                self._is_under(normalized, folder) or self._is_under(folder, normalized)
                for folder in self.opened_folders
            ) or any(
                self._is_under(file_path, normalized) for file_path in self.opened_files
            )
        if not normalized.lower().endswith(SUPPORTED_EXTS):
            return False
        return normalized in self.opened_files or any(
            self._is_under(normalized, folder) for folder in self.opened_folders
        )

    def _visible_tif_paths(self) -> List[str]:
        if not self.tree_root_path:
            return []

        source_root = self.file_model.index(self.tree_root_path)
        if self.file_model.canFetchMore(source_root):
            self.file_model.fetchMore(source_root)
        proxy_root = self.file_proxy.mapFromSource(source_root)
        paths: List[str] = []

        def walk(parent: QModelIndex):
            row_count = self.file_proxy.rowCount(parent)
            for row in range(row_count):
                index = self.file_proxy.index(row, 0, parent)
                source_index = self.file_proxy.mapToSource(index)
                if self.file_model.isDir(source_index):
                    if self.file_model.canFetchMore(source_index):
                        self.file_model.fetchMore(source_index)
                    walk(index)
                    continue
                path = self._path_from_proxy_index(index)
                if path and path.lower().endswith(SUPPORTED_EXTS):
                    paths.append(path)

        walk(proxy_root)
        return paths

    def _find_first_tif_path(self, folder: str) -> Optional[str]:
        for root, dirs, files in os.walk(folder):
            dirs.sort(key=natural_key)
            tif_names = sorted(
                [name for name in files if name.lower().endswith(SUPPORTED_EXTS)],
                key=natural_key,
            )
            if tif_names:
                return os.path.join(root, tif_names[0])
        return None

    @staticmethod
    def _is_under(path: str, parent: str) -> bool:
        try:
            p = os.path.abspath(path)
            par = os.path.abspath(parent)
            return os.path.commonpath([p, par]) == par and p != par
        except ValueError:
            return False

    @staticmethod
    def _is_under_or_equal(path: str, parent: str) -> bool:
        try:
            p = os.path.abspath(path)
            par = os.path.abspath(parent)
            return os.path.commonpath([p, par]) == par
        except ValueError:
            return False

    def _clear_loaded_image(self, status_text: str):
        path = self.active_file_path
        self.loaded = None
        self.total_slices = 1
        self.current_slice = 0
        self.active_file_path = None
        if path:
            self.calibrations_by_file.pop(path, None)
            self.changed_files.discard(path)
        self.viewer.set_scale_calibration(None)
        self.viewer.set_image(QPixmap())
        self.viewer.set_rois([], None)
        self.slice_controls.hide()
        self._refresh_roi_list()
        self._update_roi_stats()
        if status_text:
            self.status.setText(status_text)

    def _set_theme(self, theme: str):
        self.settings.setValue("theme", theme)
        qdarktheme.setup_theme(
            theme,
            custom_colors={
                "primary": PRIMARY_COLOR,
                "border": BORDER_COLOR,
            },
        )
        # Update check marks
        self.act_theme_auto.setChecked(theme == "auto")
        self.act_theme_light.setChecked(theme == "light")
        self.act_theme_dark.setChecked(theme == "dark")


def main():
    app = QApplication(sys.argv)

    # Initial setup based on settings
    settings = QSettings(ORG_NAME, ORG_DOMAIN)
    current_theme = settings.value("theme", "auto")

    qdarktheme.setup_theme(
        current_theme,
        custom_colors={
            "primary": PRIMARY_COLOR,
            "border": BORDER_COLOR,
        },
    )

    # Add custom style for better borders on buttons and combo boxes
    app.setStyleSheet(app.styleSheet() + CUSTOM_APP_STYLESHEET)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

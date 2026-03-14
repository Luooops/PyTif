"""
Centralized constants for the PyTIF application.
"""

# File Handling
SUPPORTED_EXTS = (".tif", ".tiff")
ROI_JSON_EXTENSION = ".roi.json"

# Application Metadata
APP_NAME = "PyTIF"
APP_TITLE = "PyTIF Viewer"
ORG_NAME = "PyTIF"
ORG_DOMAIN = "Viewer"

# UI Defaults
DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 850
SIDEBAR_MIN_WIDTH = 250
ROI_PANEL_MARGIN = 10

# ROI Types (Mirroring ImageViewer constants)
ROI_NONE = "none"
ROI_POLYGON = "polygon"
ROI_RECT = "rect"
ROI_ELLIPSE = "ellipse"

# UI Styles
DARK_OVERLAY_STYLE = (
    "#overlayPanel { background: rgba(40,40,40,150); border: 1px solid #555; border-radius: 4px; }"
    "QPushButton { background: rgba(60,60,60,200); border: 1px solid #777; border-radius: 2px; color: white; min-width: 32px; min-height: 32px; font-weight: bold; font-size: 16px; }"
    "QPushButton:hover { background: rgba(80,80,80,220); border-color: #999; }"
    "QPushButton:pressed { background: rgba(45,123,216,200); }"
    "QPushButton:checked { background: rgba(45,123,216,220); border-color: #64a7ff; }"
)

ROI_PANEL_STYLE = (
    "#roiPanel { background: rgba(40,40,40,220); border: 1px solid #666; border-radius: 8px; }"
    "#roiPanel QLabel { color: #ddd; }"
    "#roiPanel QToolButton { background: transparent; border: 1px solid #888; border-radius: 6px; padding: 2px; }"
    "#roiPanel QToolButton:checked { background: #2d7bd8; border-color: #64a7ff; }"
)

ROI_LIST_STYLE = (
    "QWidget { background: rgba(28,28,28,235); color: #ddd; }"
    "QListWidget { background: rgba(20,20,20,220); border: 1px solid #555; }"
    "QPushButton { background: #3a3a3a; border: 1px solid #666; padding: 4px 8px; }"
    "QPushButton:hover { background: #4a4a4a; }"
)

TREE_VIEW_STYLE = (
    "QTreeView::item:selected { background-color: #3d5a80; color: white; }"
    "QTreeView::item:selected:!active { background-color: #3d5a80; color: white; }"
)

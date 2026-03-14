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

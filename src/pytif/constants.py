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

# Styling
PRIMARY_COLOR = "#3399ff"
BORDER_COLOR = "#555555"
CONTROL_BORDER_COLOR = "#666666"
CONTROL_HOVER_BORDER_COLOR = "#888888"

CUSTOM_APP_STYLESHEET = f"""
    QMenuBar::item {{
        padding: 4px 12px;
        margin: 0px;
    }}
    QMenu {{
        min-width: 200px;
    }}
    QMenu::item {{
        min-width: 180px;
        padding: 4px 20px;
    }}
    QPushButton, QToolButton, QComboBox, QSpinBox, QDoubleSpinBox {{
        border: 1px solid {CONTROL_BORDER_COLOR};
        border-radius: 4px;
        padding: 4px;
    }}
    QToolButton::menu-button {{
        border: 1px solid {CONTROL_BORDER_COLOR};
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
        width: 16px;
    }}
    QPushButton:hover, QToolButton:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QToolButton::menu-button:hover {{
        border: 1px solid {CONTROL_HOVER_BORDER_COLOR};
    }}
    QPushButton:pressed, QToolButton:pressed {{
        background-color: rgba(255, 255, 255, 0.1);
    }}
"""

# ROI Types (Mirroring ImageViewer constants)
ROI_NONE = "none"
ROI_POLYGON = "polygon"
ROI_RECT = "rect"
ROI_ELLIPSE = "ellipse"

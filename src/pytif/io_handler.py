import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tifffile
from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QPixmap

from .constants import SUPPORTED_EXTS
from .roi import serialize_roi_geometry
from .utils import (
    extract_tiff_scale_calibration,
    flatten_to_slices,
    numpy_to_qimage,
    rgb_like_to_gray,
)


def load_tiff_data(path: str) -> Tuple[np.ndarray, int, Optional[Dict[str, Any]]]:
    """
    Loads TIFF array, flattens to (H,W) or (S,H,W), and extracts metadata.
    Returns: (loaded_array, total_slices, scale_calibration)
    """
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
        scale_calibration = extract_tiff_scale_calibration(tif)

    arr = rgb_like_to_gray(arr)
    flat, slices = flatten_to_slices(arr)
    return flat, slices, scale_calibration


class TiffLoaderSignals(QObject):
    finished = Signal(
        str, object, int, object, object
    )  # path, flat, slices, calibration, initial_pixmap
    error = Signal(str, str)  # path, error_msg


class TiffLoaderWorker(QRunnable):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.signals = TiffLoaderSignals()

    def run(self):
        try:
            flat, slices, calibration = load_tiff_data(self.path)

            # Pre-render the first slice to a QPixmap to avoid UI freeze during display
            if flat.ndim == 2:
                img = flat
            else:
                img = flat[0]

            qimg = numpy_to_qimage(img)
            pix = QPixmap.fromImage(qimg)

            self.signals.finished.emit(self.path, flat, slices, calibration, pix)
        except Exception as e:
            self.signals.error.emit(self.path, str(e))


def save_rois_to_json(path: str, image_path: str, rois: List[Dict[str, Any]]) -> int:
    """
    Serializes ROIs and saves to a JSON file.
    Returns the count of saved ROIs.
    """
    data_rois = []
    for roi in rois:
        s = serialize_roi_geometry(roi)
        if s is not None:
            data_rois.append(s)

    payload = {
        "version": 1,
        "image_path": os.path.abspath(image_path),
        "image_name": os.path.basename(image_path),
        "rois": data_rois,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return len(data_rois)


def load_rois_from_json(path: str) -> List[Dict[str, Any]]:
    """
    Loads ROIs from a JSON file and returns a list of ROI dictionaries.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw = payload.get("rois", [])
    if not isinstance(raw, list):
        raise ValueError("Invalid ROI file: 'rois' must be a list.")

    rois = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        typ = r.get("type")
        if typ == "polygon":
            pts = r.get("points", [])
            if isinstance(pts, list) and len(pts) >= 3:
                rois.append(
                    {"type": typ, "points": [(float(x), float(y)) for x, y in pts]}
                )
        elif typ in ("rect", "ellipse"):
            rois.append(
                {
                    "type": typ,
                    "x": float(r.get("x", 0.0)),
                    "y": float(r.get("y", 0.0)),
                    "w": float(r.get("w", 0.0)),
                    "h": float(r.get("h", 0.0)),
                }
            )
    return rois


def save_tiff_with_metadata(
    path: str,
    data: np.ndarray,
    units_per_pixel: float,
    unit_label: str,
    description: Optional[str] = None,
):
    """
    Saves image data to a TIFF file with resolution and unit metadata.
    """
    # resolution is pixels per unit
    res = 1.0 / units_per_pixel
    tifffile.imwrite(
        path,
        data,
        resolution=(res, res),
        resolutionunit=1,
        imagej=True,
        metadata=(
            {"unit": unit_label, "description": description}
            if description
            else {"unit": unit_label}
        ),
        photometric="minisblack",
    )

import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tifffile
from PySide6.QtGui import QImage

_UNIT_ALIASES = {
    "m": ("m", 1.0),
    "meter": ("m", 1.0),
    "meters": ("m", 1.0),
    "metre": ("m", 1.0),
    "metres": ("m", 1.0),
    "cm": ("cm", 1e-2),
    "centimeter": ("cm", 1e-2),
    "centimeters": ("cm", 1e-2),
    "centimetre": ("cm", 1e-2),
    "centimetres": ("cm", 1e-2),
    "mm": ("mm", 1e-3),
    "millimeter": ("mm", 1e-3),
    "millimeters": ("mm", 1e-3),
    "millimetre": ("mm", 1e-3),
    "millimetres": ("mm", 1e-3),
    "um": ("um", 1e-6),
    "micrometer": ("um", 1e-6),
    "micrometers": ("um", 1e-6),
    "micrometre": ("um", 1e-6),
    "micrometres": ("um", 1e-6),
    "µm": ("um", 1e-6),
    "μm": ("um", 1e-6),
    "nm": ("nm", 1e-9),
    "nanometer": ("nm", 1e-9),
    "nanometers": ("nm", 1e-9),
    "nanometre": ("nm", 1e-9),
    "nanometres": ("nm", 1e-9),
    "in": ("in", 0.0254),
    "inch": ("in", 0.0254),
    "inches": ("in", 0.0254),
}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def to_8bit_grayscale(img2d: np.ndarray, auto_contrast: bool = False) -> np.ndarray:
    if not auto_contrast:
        # If it's already 8-bit, return as is
        if img2d.dtype == np.uint8:
            return img2d
        # If it's 16-bit, downscale to 8-bit by bit-shifting (preserving raw value ratios)
        if img2d.dtype == np.uint16:
            return (img2d >> 8).astype(np.uint8)
        # For other types (float, etc.), we still need a fallback, but let's default to minmax
        # though user asked for raw. For float, "raw" is ambiguous without a range.
        # We'll use 0-1 range if it's float, otherwise minmax as a safe fallback.
        if np.issubdtype(img2d.dtype, np.floating):
            return (np.clip(img2d, 0, 1) * 255).astype(np.uint8)

    x = img2d.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.uint8)

    if auto_contrast:
        vmin = np.percentile(x[finite], 0.2)
        vmax = np.percentile(x[finite], 99.8)
    else:
        vmin = np.min(x[finite])
        vmax = np.max(x[finite])

    if vmax <= vmin:
        vmax = vmin + 1

    y = (x - vmin) / (vmax - vmin)
    y = np.clip(y, 0, 1)
    return (y * 255).astype(np.uint8)


def numpy_to_qimage(img: np.ndarray, auto_contrast: bool = False) -> QImage:
    if img.ndim == 2:
        u8 = to_8bit_grayscale(img, auto_contrast=auto_contrast)
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


def rgb_like_to_gray(arr: np.ndarray) -> np.ndarray:
    """
    Convert common RGB/RGBA TIFF layouts to grayscale.
    Supported examples:
      (H, W, 3/4), (S, H, W, 3/4), (3/4, H, W), (S, 3/4, H, W)
    """
    x = arr

    # Planar channel-first single image: (C,H,W) -> (H,W,C)
    if x.ndim == 3 and x.shape[0] in (3, 4) and x.shape[-1] not in (3, 4):
        x = np.moveaxis(x, 0, -1)

    # Planar channel-first stack: (...,C,H,W) -> (...,H,W,C)
    if x.ndim >= 4 and x.shape[-3] in (3, 4) and x.shape[-1] not in (3, 4):
        x = np.moveaxis(x, -3, -1)

    # Interleaved channel-last RGB/RGBA
    if x.ndim >= 3 and x.shape[-1] in (3, 4):
        rgb = x[..., :3].astype(np.float32, copy=False)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return gray

    return x


def _normalize_physical_unit(raw_unit: Any) -> Tuple[Optional[str], Optional[float]]:
    if raw_unit is None:
        return None, None

    text = str(raw_unit).strip()
    if not text:
        return None, None

    normalized = text.lower()
    if normalized in {"none", "pixel", "pixels", "px"}:
        return None, None

    unit = _UNIT_ALIASES.get(normalized)
    if unit is not None:
        return unit
    return text, None


def _coerce_positive_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if result > 0 else None


def _rational_to_float(value: Any) -> Optional[float]:
    result = _coerce_positive_float(value)
    if result is not None:
        return result
    if isinstance(value, tuple) and len(value) == 2:
        num = _coerce_positive_float(value[0])
        den = _coerce_positive_float(value[1])
        if num is None or den is None:
            return None
        return num / den
    return None


def extract_tiff_scale_calibration(
    tif: tifffile.TiffFile,
) -> Optional[Dict[str, Any]]:
    if not tif.pages:
        return None

    page = tif.pages[0]
    resolution = getattr(page, "resolution", None)
    x_resolution = None
    y_resolution = None
    if isinstance(resolution, tuple) and len(resolution) >= 2:
        x_resolution = _coerce_positive_float(resolution[0])
        y_resolution = _coerce_positive_float(resolution[1])

    if x_resolution is None or y_resolution is None:
        tags = page.tags
        if x_resolution is None and "XResolution" in tags:
            x_resolution = _rational_to_float(tags["XResolution"].value)
        if y_resolution is None and "YResolution" in tags:
            y_resolution = _rational_to_float(tags["YResolution"].value)

    if x_resolution is None or y_resolution is None:
        return None

    unit_label: Optional[str] = None
    meters_per_unit: Optional[float] = None

    imagej_meta = getattr(tif, "imagej_metadata", None) or {}
    if isinstance(imagej_meta, dict):
        unit_label, meters_per_unit = _normalize_physical_unit(imagej_meta.get("unit"))

    if unit_label is None:
        raw_unit = None
        if "ResolutionUnit" in page.tags:
            raw_unit = page.tags["ResolutionUnit"].value
        unit_name = getattr(raw_unit, "name", "").lower()
        unit_code = None
        try:
            unit_code = int(raw_unit) if raw_unit is not None else None
        except (TypeError, ValueError):
            unit_code = None
        if unit_name == "inch" or unit_code == 2:
            unit_label, meters_per_unit = "in", 0.0254
        elif unit_name == "centimeter" or unit_code == 3:
            unit_label, meters_per_unit = "cm", 1e-2

    if unit_label is None:
        return None

    return {
        "x_units_per_pixel": 1.0 / x_resolution,
        "y_units_per_pixel": 1.0 / y_resolution,
        "unit_label": unit_label,
        "meters_per_unit": meters_per_unit,
    }

import re
from typing import Tuple
import numpy as np
from PySide6.QtGui import QImage


def natural_key(s: str):
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

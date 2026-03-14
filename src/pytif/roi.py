from typing import Any, Dict, Optional, Tuple

import numpy as np


def roi_geometry(state: Dict[str, Any]) -> Tuple[float, float]:
    typ = state.get("type")
    if typ == "polygon":
        pts = np.array(state.get("points", []), dtype=np.float64)
        if len(pts) < 3:
            return 0.0, 0.0
        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        perimeter = float(
            np.sum(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]])))
        )
        return float(area), perimeter
    if typ == "rect":
        w = float(state.get("w", 0.0))
        h = float(state.get("h", 0.0))
        return w * h, 2.0 * (w + h)
    if typ == "ellipse":
        a = float(state.get("w", 0.0)) / 2.0
        b = float(state.get("h", 0.0)) / 2.0
        area = np.pi * a * b
        perimeter = (
            float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))
            if a > 0 and b > 0
            else 0.0
        )
        return float(area), perimeter
    return 0.0, 0.0


def roi_mask(state: Dict[str, Any], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    typ = state.get("type")
    mask = np.zeros((h, w), dtype=bool)
    if typ == "rect":
        x0 = float(state.get("x", 0.0))
        y0 = float(state.get("y", 0.0))
        rw = float(state.get("w", 0.0))
        rh = float(state.get("h", 0.0))
        xmin = max(0, int(np.floor(min(x0, x0 + rw))))
        xmax = min(w, int(np.ceil(max(x0, x0 + rw))))
        ymin = max(0, int(np.floor(min(y0, y0 + rh))))
        ymax = min(h, int(np.ceil(max(y0, y0 + rh))))
        if xmin >= xmax or ymin >= ymax:
            return mask
        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        x = xx + 0.5
        y = yy + 0.5
        mask[ymin:ymax, xmin:xmax] = (
            (x >= x0) & (x <= x0 + rw) & (y >= y0) & (y <= y0 + rh)
        )
        return mask
    if typ == "ellipse":
        x0 = float(state.get("x", 0.0))
        y0 = float(state.get("y", 0.0))
        rw = float(state.get("w", 0.0))
        rh = float(state.get("h", 0.0))
        xmin = max(0, int(np.floor(min(x0, x0 + rw))))
        xmax = min(w, int(np.ceil(max(x0, x0 + rw))))
        ymin = max(0, int(np.floor(min(y0, y0 + rh))))
        ymax = min(h, int(np.ceil(max(y0, y0 + rh))))
        if xmin >= xmax or ymin >= ymax:
            return mask
        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        x = xx + 0.5
        y = yy + 0.5
        cx = x0 + rw / 2.0
        cy = y0 + rh / 2.0
        a = max(rw / 2.0, 1e-9)
        b = max(rh / 2.0, 1e-9)
        mask[ymin:ymax, xmin:xmax] = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0
        return mask
    if typ == "polygon":
        pts = np.array(state.get("points", []), dtype=np.float64)
        if len(pts) < 3:
            return mask
        xmin = max(0, int(np.floor(np.min(pts[:, 0]))))
        xmax = min(w, int(np.ceil(np.max(pts[:, 0]))))
        ymin = max(0, int(np.floor(np.min(pts[:, 1]))))
        ymax = min(h, int(np.ceil(np.max(pts[:, 1]))))
        if xmin >= xmax or ymin >= ymax:
            return mask
        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        x = xx + 0.5
        y = yy + 0.5
        px = pts[:, 0]
        py = pts[:, 1]
        inside = np.zeros((ymax - ymin, xmax - xmin), dtype=bool)
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            xi, yi = px[i], py[i]
            xj, yj = px[j], py[j]
            # Jordan Curve Theorem based scanline
            intersect = ((yi > y) != (yj > y)) & (
                x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
            )
            inside ^= intersect
        mask[ymin:ymax, xmin:xmax] = inside
        return mask
    return mask


def serialize_roi_geometry(roi: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    typ = roi.get("type")
    if typ == "polygon":
        pts = roi.get("points", [])
        return {
            "type": typ,
            "points": [[float(x), float(y)] for x, y in pts],
        }
    if typ in ("rect", "ellipse"):
        return {
            "type": typ,
            "x": float(roi.get("x", 0.0)),
            "y": float(roi.get("y", 0.0)),
            "w": float(roi.get("w", 0.0)),
            "h": float(roi.get("h", 0.0)),
        }
    return None

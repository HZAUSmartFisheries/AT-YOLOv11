"""Turbidity pseudo-label computation and calibration utilities."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch


DEFAULT_CALIBRATION = Path("turbidity_pseudo_calibration.json")


def raw_turbidity(images: torch.Tensor) -> torch.Tensor:
    """Compute the brightness-to-gradient-variance turbidity indicator for BCHW images in the 0-255 range."""
    images = images.float()
    batch_size = images.shape[0]
    gray = images.mean(dim=1, keepdim=True)
    diff_y = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    diff_x = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    grad_var = diff_x.reshape(batch_size, -1).var(dim=1) + diff_y.reshape(batch_size, -1).var(dim=1)
    mean_brightness = gray.reshape(batch_size, -1).mean(dim=1)
    return (mean_brightness + 1e-5) / (grad_var + 1e-5)


@lru_cache
def load_calibration(path: str) -> tuple[float, float]:
    """Load fixed training-set log-space quantiles from a calibration JSON file."""
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Missing turbidity calibration file: {calibration_path.resolve()}. "
            "Run tools/calibrate_turbidity_pseudo.py before training, or set TURBIDITY_PSEUDO_MODE=legacy."
        )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    q05, q95 = float(calibration["log_raw_q05"]), float(calibration["log_raw_q95"])
    if q95 <= q05:
        raise ValueError(f"Invalid turbidity calibration: q95={q95} must be greater than q05={q05}")
    return q05, q95


def normalize_turbidity(raw_t: torch.Tensor) -> torch.Tensor:
    """Map raw turbidity to [0, 1] using fixed robust training-set quantiles."""
    calibration_path = os.getenv("TURBIDITY_CALIBRATION", str(DEFAULT_CALIBRATION))
    q05, q95 = load_calibration(calibration_path)
    log_raw_t = torch.log1p(raw_t)
    return torch.clamp((log_raw_t - q05) / (q95 - q05 + 1e-6), 0.0, 1.0)


def soft_normalize_turbidity(raw_t: torch.Tensor) -> torch.Tensor:
    """Map raw turbidity smoothly to (0, 1) without hard endpoint clipping.

    The training-set q05 and q95 anchors map to approximately 0.25 and 0.75,
    leaving headroom for degradation levels outside the calibration interval.
    """
    calibration_path = os.getenv("TURBIDITY_CALIBRATION", str(DEFAULT_CALIBRATION))
    q05, q95 = load_calibration(calibration_path)
    log_raw_t = torch.log1p(raw_t)
    center = (q05 + q95) / 2.0
    slope = 2.0 * torch.log(torch.tensor(3.0, device=raw_t.device, dtype=raw_t.dtype)) / (q95 - q05)
    return torch.sigmoid(slope * (log_raw_t - center))

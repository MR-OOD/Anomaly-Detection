#!/usr/bin/env python3
"""Post-processing utilities for FastFlow anomaly maps.

This module applies body masks to per-image anomaly maps prior to evaluation.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

ARRAY_EXTENSIONS = {".npy", ".npz"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VALID_EXTENSIONS = ARRAY_EXTENSIONS | IMAGE_EXTENSIONS


@dataclass
class ArrayWithMeta:
    """Container that tracks array data and its original dtype."""

    data: np.ndarray
    dtype: np.dtype


def _load_array(path: Path) -> ArrayWithMeta:
    """Load an anomaly map from disk."""
    suffix = path.suffix.lower()
    if suffix not in VALID_EXTENSIONS:
        raise ValueError(f"Unsupported anomaly map format: {path}")

    if suffix in ARRAY_EXTENSIONS:
        array = np.load(path)
        if isinstance(array, np.lib.npyio.NpzFile):
            arr = array.get("arr_0")
            if arr is None:
                raise ValueError(f"NPZ file {path} does not contain 'arr_0'.")
            array = arr
    else:
        image = Image.open(path)
        array = np.array(image)

    return ArrayWithMeta(data=array.astype(np.float32, copy=False), dtype=array.dtype)


def _load_mask(path: Path, threshold: float) -> np.ndarray:
    """Load body mask and convert to a binary tensor."""
    suffix = path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError(f"Body mask must be an image file, found: {path}")
    image = Image.open(path).convert("F")
    mask = np.array(image, dtype=np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    binary_mask = (mask >= threshold).astype(np.float32)
    return binary_mask


def _broadcast_mask(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast mask to match anomaly map shape."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    if len(target_shape) == 2:
        return mask
    if len(target_shape) == 3:
        return np.broadcast_to(mask[..., None], target_shape)
    raise ValueError(f"Unsupported anomaly map shape {target_shape} for mask application")


def _apply_mask(anomaly_map: ArrayWithMeta, mask: np.ndarray) -> np.ndarray:
    """Apply binary mask to anomaly map."""
    masked = anomaly_map.data * mask
    target_dtype = anomaly_map.dtype
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        masked = np.clip(masked, info.min, info.max)
        return masked.astype(target_dtype)
    return masked.astype(target_dtype)


def _parse_replacements(replacements: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in replacements:
        if ":" not in item:
            raise ValueError(f"Invalid replacement '{item}'. Expected format 'src:dst'.")
        src, dst = item.split(":", 1)
        mapping[src] = dst
    return mapping


def _apply_replacements(relative_path: Path, mapping: dict[str, str]) -> Path:
    if not mapping:
        return relative_path
    parts = [mapping.get(part, part) for part in relative_path.parts]
    return Path(*parts)


def _resolve_mask_path(
    anomaly_path: Path,
    anomaly_root: Path,
    mask_root: Path,
    replacements: dict[str, str],
) -> Path:
    relative = anomaly_path.relative_to(anomaly_root)
    mapped_relative = _apply_replacements(relative, replacements)
    return mask_root / mapped_relative


def _save_array(path: Path, masked: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, masked)
    elif suffix == ".npz":
        np.savez_compressed(path, masked)
    else:
        image = Image.fromarray(masked)
        image.save(path)


def apply_body_mask(
    anomaly_root: Path,
    mask_root: Path,
    output_root: Path,
    *,
    threshold: float,
    replacements: dict[str, str],
    strict: bool,
) -> None:
    anomaly_files = [
        path
        for path in anomaly_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    if not anomaly_files:
        raise FileNotFoundError(f"No anomaly maps found in {anomaly_root}.")

    for anomaly_path in anomaly_files:
        mask_path = _resolve_mask_path(anomaly_path, anomaly_root, mask_root, replacements)
        if not mask_path.exists():
            message = f"Missing body mask for {anomaly_path.relative_to(anomaly_root)} at {mask_path}"
            if strict:
                raise FileNotFoundError(message)
            print(f"[WARN] {message}", file=sys.stderr)
            continue

        anomaly_array = _load_array(anomaly_path)
        mask = _load_mask(mask_path, threshold=threshold)
        mask_broadcast = _broadcast_mask(mask, anomaly_array.data.shape)
        masked = _apply_mask(anomaly_array, mask_broadcast)

        destination = output_root / anomaly_path.relative_to(anomaly_root)
        _save_array(destination, masked)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply body masks to FastFlow anomaly maps prior to evaluation."
    )
    parser.add_argument("--anomaly-dir", type=Path, required=True, help="Directory containing anomaly maps.")
    parser.add_argument("--body-mask-dir", type=Path, required=True, help="Directory with body mask images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for masked anomaly maps.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold (0-255 for 8-bit masks) applied before binarising the mask.",
    )
    parser.add_argument(
        "--path-replace",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="Optional path component replacement (e.g. 'img:bodymask'). Can be supplied multiple times.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip samples without body masks instead of raising an error.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    replacements = _parse_replacements(args.path_replace)
    threshold = args.mask_threshold
    if threshold > 1.0:
        # Assume input masks are 8-bit when threshold > 1
        threshold = threshold / 255.0
    apply_body_mask(
        anomaly_root=args.anomaly_dir,
        mask_root=args.body_mask_dir,
        output_root=args.output_dir,
        threshold=threshold,
        replacements=replacements,
        strict=not args.skip_missing,
    )


if __name__ == "__main__":
    main()

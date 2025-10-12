#!/usr/bin/env python3
"""Minimal FastFlow inference script that exports anomaly maps as .npy files."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import nibabel as nib
import torch
from lightning.pytorch import Trainer
from PIL import Image
from torch.nn import functional as F

from fastflow_dataset import prepare_dataset_root
from train_fastflow import _build_model, _resolve_eval_dirs  # reuse existing helpers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract FastFlow anomaly maps and prediction masks.")
    parser.add_argument("--data_root", type=Path, required=True, help="Dataset root to evaluate.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to FastFlow checkpoint.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store outputs.")
    parser.add_argument("--split", type=str, default="test", choices={"test", "valid"}, help="Dataset split to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="Prediction batch size.")
    parser.add_argument("--backbone", type=str, default="radimagenet_resnet50", help="Backbone identifier.")
    parser.add_argument("--radimagenet_ckpt", type=str, default=None, help="Optional RadImageNet checkpoint.")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index to use; -1 for CPU.")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Threshold for generating prediction masks when the model does not supply one.")
    parser.add_argument("--map_size", type=int, default=224, help="Spatial size (height/width) for saved anomaly maps and masks.")
    parser.add_argument(
        "--dataset-format",
        type=str,
        choices={"auto", "png", "nifti"},
        default="auto",
        help="Input dataset format. Use 'nifti' to force conversion, 'png' to skip detection.",
    )
    parser.add_argument(
        "--conversion-cache-dir",
        type=Path,
        default=None,
        help="Optional directory where converted PNG datasets will be stored when using NIfTI inputs.",
    )
    parser.add_argument(
        "--mask-output-format",
        type=str,
        choices={"png", "nifti"},
        default="png",
        help="File format used when saving prediction masks.",
    )
    parser.add_argument(
        "--repair-layout",
        action="store_true",
        help="Reorganise existing outputs into dataset-style folders after extraction.",
    )
    return parser.parse_args()


def _coerce_list(obj: Sequence | str | None) -> list[str]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        flattened: list[str] = []
        for item in obj:
            flattened.extend(_coerce_list(item))
        return flattened
    return [str(obj)]


def _derive_output_relative_path(rel_image_path: Path, split: str) -> Path:
    """Recreate a dataset-style relative path rooted at <split>/<good|Ungood>/..."""
    parts = list(rel_image_path.parts)
    label_idx = next((idx for idx, part in enumerate(parts) if part.lower() in {"good", "ungood"}), None)
    if label_idx is None:
        # Fall back to preserving the existing layout under the split.
        return Path(split) / rel_image_path

    label = parts[label_idx]
    suffix_parts = parts[label_idx + 1 :]

    dest_parts: list[str] = [split, label]
    if suffix_parts:
        # Preserve subfolders such as img/, label/, etc.
        dest_parts.extend(suffix_parts)
    else:
        dest_parts.append(rel_image_path.name)

    return Path(*dest_parts)


def _extract(outputs, keys: Iterable[str]):
    if outputs is None:
        return None

    if isinstance(outputs, dict):
        for key in keys:
            if key in outputs and outputs[key] is not None:
                return outputs[key]

    for key in keys:
        if hasattr(outputs, key):
            value = getattr(outputs, key)
            if value is not None:
                return value

    if isinstance(outputs, (list, tuple)):
        collected = []
        for item in outputs:
            value = _extract(item, keys)
            if value is not None:
                collected.append(value)
        if collected:
            return collected

    return None


def _resolve_split_dirs(root: Path, split: str) -> tuple[str, str, str | None]:
    base = root / split
    if not base.exists():
        raise FileNotFoundError(f"Split '{split}' not found under {root}")

    good = base / "good"
    bad = base / "Ungood"
    has_img_subdirs = (good / "img").exists() and (bad / "img").exists()

    if has_img_subdirs:
        normal_test_dir = f"{split}/good/img"
        abnormal_dir = f"{split}/Ungood/img"
        mask_dir = f"{split}/Ungood/label" if (bad / "label").exists() else None
    else:
        normal_test_dir = f"{split}/good"
        abnormal_dir = f"{split}/Ungood"
        mask_dir = f"{split}/Ungood/label" if (bad / "label").exists() else None

    return normal_test_dir, abnormal_dir, mask_dir


def _normalize_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def _index_dataset_stems(data_root: Path, split: str) -> dict[str, tuple[str, Path]]:
    """Map base stem -> (label, relative dataset path) for quick lookup."""
    split_root = data_root / split
    index: dict[str, tuple[str, Path]] = {}

    def register(path: Path) -> None:
        try:
            relative_to_split = path.relative_to(split_root)
        except ValueError:
            return
        parts = relative_to_split.parts
        if not parts:
            return
        label = parts[0]
        if label not in {"good", "Ungood"}:
            return
        stem = _normalize_stem(path)
        if stem not in index or ("img" in relative_to_split.parts and "img" not in index[stem][1].parts):
            index[stem] = (label, path.relative_to(split_root / label))

    for label in ("good", "Ungood"):
        img_dir = split_root / label / "img"
        if img_dir.exists():
            for file in img_dir.iterdir():
                if file.is_file():
                    register(file)

    for label in ("good", "Ungood"):
        label_root = split_root / label
        if label_root.exists():
            for file in label_root.rglob("*"):
                if file.is_file():
                    register(file)

    return index


def _derive_destination_path(
    category: str,
    filename: str,
    index: dict[str, tuple[str, Path]],
    split: str,
) -> tuple[Path | None, str | None]:
    suffix_map = {
        "anomaly_maps": ["_anomaly_map.npy", "_anomaly_map.npz", "_anomaly_map.nii.gz", "_anomaly_map.nii"],
        "prediction_masks": ["_pred_mask.npy", "_pred_mask.npz", "_pred_mask.nii.gz", "_pred_mask.nii", "_pred_mask.png"],
    }
    base_name = None
    for suffix in suffix_map.get(category, []):
        if filename.endswith(suffix):
            base_name = filename[: -len(suffix)]
            break
    if base_name is None:
        base_name = Path(filename).stem

    if base_name not in index:
        return None, base_name

    label, rel_dataset = index[base_name]
    parent = rel_dataset.parent
    dest_parts = [split, label]
    if str(parent) != ".":
        dest_parts.append(str(parent))
    dest = Path(*dest_parts) / filename
    return dest, None


def reorganize_existing_outputs(output_root: Path, data_root: Path, split: str) -> None:
    """Move previously exported files into dataset-like good/Ungood folders."""
    index = _index_dataset_stems(data_root, split)
    if not index:
        print(f"[WARN] Unable to index dataset stems under {data_root}/{split}; skipping reorganisation.")
        return

    relocated = 0
    skipped: list[str] = []
    for category in ("anomaly_maps", "prediction_masks"):
        category_base = output_root / category
        split_root = category_base / split
        if not split_root.exists():
            continue

        for file_path in list(split_root.rglob("*")):
            if file_path.is_dir():
                continue
            rel_parts = file_path.relative_to(split_root).parts
            if any(part in {"good", "Ungood"} for part in rel_parts):
                continue
            destination_rel, missing_key = _derive_destination_path(category, file_path.name, index, split)
            if destination_rel is None:
                skipped.append(missing_key or file_path.stem)
                continue
            destination_path = category_base / destination_rel
            if destination_path == file_path:
                continue
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if destination_path.exists():
                destination_path.unlink()
            file_path.rename(destination_path)
            relocated += 1

    if relocated:
        print(f"[INFO] Relocated {relocated} files into dataset-style folders under {output_root}")
    if skipped:
        unique_skipped = sorted(set(skipped))
        print(
            f"[WARN] Unable to determine destination for {len(unique_skipped)} files (example: {unique_skipped[:5]}). "
            "They were left in their original location."
        )


def _prepare_datamodule(args: argparse.Namespace):
    from anomalib.data import Folder

    data_root = Path(args.data_root)
    try:
        normal_test_dir, abnormal_dir, mask_dir = _resolve_split_dirs(data_root, args.split)
    except FileNotFoundError:
        normal_test_dir, abnormal_dir, mask_dir = _resolve_eval_dirs(data_root)

    return Folder(
        name=f"{args.dataset_name}_fastflow_{args.split}",
        root=str(args.data_root),
        normal_dir="train/good",
        normal_test_dir=normal_test_dir,
        abnormal_dir=abnormal_dir,
        mask_dir=mask_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=8,
        extensions=(".png",),
        val_split_mode="none",
        test_split_mode="from_dir",
    )


def _load_model(args: argparse.Namespace):
    try:
        from anomalib.models import Fastflow

        return Fastflow.load_from_checkpoint(
            checkpoint_path=str(args.checkpoint),
            backbone=args.backbone,
            strict=True,
        )
    except Exception:
        pass

    checkpoint_blob = torch.load(args.checkpoint, map_location="cpu")
    base_model = _build_model(args.backbone, args.radimagenet_ckpt)
    if isinstance(checkpoint_blob, dict) and "state_dict" in checkpoint_blob:
        base_model.load_state_dict(checkpoint_blob["state_dict"], strict=False)
    elif isinstance(checkpoint_blob, dict):
        base_model.load_state_dict(checkpoint_blob, strict=False)
    else:
        base_model.load_state_dict(checkpoint_blob.state_dict(), strict=False)
    return base_model


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    args.original_data_root = data_root
    cache_root = args.conversion_cache_dir.resolve() if args.conversion_cache_dir else None
    prepared_root, converted = prepare_dataset_root(
        data_root, format_hint=args.dataset_format, cache_root=cache_root
    )
    if converted:
        print(f"[INFO] Converted NIfTI dataset at {data_root} -> PNG cache at {prepared_root}")

    args.dataset_name = data_root.name
    args.data_root = prepared_root

    datamodule = _prepare_datamodule(args)
    model = _load_model(args)

    if hasattr(model, "evaluator"):
        model.evaluator = None

    if args.gpu >= 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = [args.gpu]
    else:
        accelerator = "cpu"
        devices = 1

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_progress_bar=True,
    )

    predictions = trainer.predict(model=model, datamodule=datamodule)
    output_root = Path(args.output_dir).resolve()
    maps_dir = output_root / "anomaly_maps"
    masks_dir = output_root / "prediction_masks"
    maps_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    mirror_root = Path("maps_fastflow") / args.split
    mirror_maps = mirror_root / "anomaly_maps"
    mirror_masks = mirror_root / "prediction_masks"
    mirror_maps.mkdir(parents=True, exist_ok=True)
    mirror_masks.mkdir(parents=True, exist_ok=True)

    saved = 0
    threshold = args.mask_threshold
    target_size = (args.map_size, args.map_size)

    for batch_outputs in predictions:
        if batch_outputs is None:
            continue

        anomaly_maps = _extract(batch_outputs, ("anomaly_maps", "anomaly_map"))
        pred_masks = _extract(batch_outputs, ("pred_masks", "pred_mask", "anomaly_masks", "anomaly_mask"))
        image_paths = _coerce_list(_extract(batch_outputs, ("image_paths", "image_path")))

        if anomaly_maps is None or not image_paths:
            continue

        anomaly_tensor = torch.as_tensor(anomaly_maps).detach().cpu()
        if anomaly_tensor.ndim == 4 and anomaly_tensor.shape[1] == 1:
            anomaly_tensor = anomaly_tensor.squeeze(1)
        if anomaly_tensor.ndim == 4 and anomaly_tensor.shape[0] == 1:
            anomaly_tensor = anomaly_tensor.squeeze(0)
        if anomaly_tensor.ndim == 3 and anomaly_tensor.shape[-2:] != target_size:
            anomaly_tensor = F.interpolate(
                anomaly_tensor.unsqueeze(1),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        if pred_masks is not None:
            mask_tensor = torch.as_tensor(pred_masks).detach().cpu()
            if mask_tensor.ndim == 4 and mask_tensor.shape[1] == 1:
                mask_tensor = mask_tensor.squeeze(1)
            if mask_tensor.ndim == 4 and mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.squeeze(0)
            if mask_tensor.ndim == 3 and mask_tensor.shape[-2:] != target_size:
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(1).to(torch.float32),
                    size=target_size,
                    mode="nearest",
                ).squeeze(1)
        else:
            mask_tensor = (anomaly_tensor >= threshold).to(torch.uint8)

        if mask_tensor.dtype != torch.uint8:
            mask_tensor = (mask_tensor >= threshold).to(torch.uint8)

        for idx, path_str in enumerate(image_paths):
            try:
                anomaly_map = anomaly_tensor[idx].numpy().astype("float32")
                mask = mask_tensor[idx].numpy()
            except Exception:
                continue
            image_path_obj = Path(path_str)
            try:
                rel_image_path = image_path_obj.relative_to(data_root)
            except ValueError:
                original_root = getattr(args, "original_data_root", None)
                if original_root is not None:
                    try:
                        rel_image_path = image_path_obj.relative_to(original_root)
                    except ValueError:
                        rel_image_path = Path(image_path_obj.name)
                else:
                    rel_image_path = Path(image_path_obj.name)

            destination_rel = _derive_output_relative_path(rel_image_path, args.split)
            stem = image_path_obj.stem
            dest_parent = destination_rel.parent
            map_rel_path = dest_parent / f"{stem}_anomaly_map.npy"
            mask_extension = ".png" if args.mask_output_format == "png" else ".nii.gz"
            mask_rel_path = dest_parent / f"{stem}_pred_mask{mask_extension}"

            map_path = maps_dir / map_rel_path
            mask_path = masks_dir / mask_rel_path
            mirror_map_path = mirror_maps / map_rel_path
            mirror_mask_path = mirror_masks / mask_rel_path

            for path in (map_path.parent, mirror_map_path.parent, mask_path.parent, mirror_mask_path.parent):
                path.mkdir(parents=True, exist_ok=True)

            np.save(map_path, anomaly_map)
            np.save(mirror_map_path, anomaly_map)

            if args.mask_output_format == "png":
                mask_arr = mask.astype(np.uint8)
                if mask_arr.max() <= 1:
                    mask_arr = mask_arr * 255
                mask_img = Image.fromarray(mask_arr)
                mask_img.save(mask_path)
                mask_img.save(mirror_mask_path)
            else:
                mask_arr = mask.astype(np.float32)
                nifti_mask = nib.Nifti1Image(mask_arr, affine=np.eye(4))
                nib.save(nifti_mask, mask_path)
                nib.save(nifti_mask, mirror_mask_path)
            saved += 1

    print(f"[INFO] Saved anomaly maps and prediction masks for {saved} samples to {output_root}")
    if args.repair_layout:
        reorganize_existing_outputs(output_root, args.original_data_root, args.split)


if __name__ == "__main__":
    main()

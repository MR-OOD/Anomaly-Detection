from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger

from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.metrics.evaluator import Evaluator
from radimagenet_utils import load_radimagenet_resnet_weights


DEFAULT_DATA_ROOT = "/local/scratch/koepchen/synth23_pelvis_v6_png/synth23_pelvis_v6_png"
DEFAULT_LOG_DIR = "/home/user/koepchen/OOD_tests"


def _build_model(backbone: str, radimagenet_ckpt: str | None) -> Fastflow:
    name = backbone.lower()
    suffix = name.replace("radimagenet", "").strip("-_ ") or "resnet50"
    mapping = {"resnet50": "resnet50", "50": "resnet50", "resnet18": "resnet18", "18": "resnet18"}
    target = mapping.get(suffix, suffix)

    try:
        model = Fastflow(backbone=target, pre_trained=False)
    except TypeError:
        model = Fastflow()

    if target.startswith("resnet") and radimagenet_ckpt:
        feature_extractor = model.model.feature_extractor
        load_radimagenet_resnet_weights(feature_extractor, radimagenet_ckpt, strict=False)

    return model


def _resolve_eval_dirs(root: Path) -> tuple[str, str, str | None]:
    valid_good = root / "valid" / "good" / "img"
    valid_bad = root / "valid" / "Ungood" / "img"
    if valid_good.exists() and valid_bad.exists():
        return "valid/good/img", "valid/Ungood/img", None

    test_good = root / "test" / "good" / "img"
    test_bad = root / "test" / "Ungood" / "img"
    if test_good.exists() and test_bad.exists():
        return "test/good/img", "test/Ungood/img", None

    return "test/good", "test/Ungood", None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal FastFlow training script (PNG only)")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--backbone", type=str, default="radimagenet_resnet50")
    parser.add_argument("--radimagenet_ckpt", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index to use; -1 forces CPU")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    os.environ.setdefault("ANOMALIB_LOG_DIR", args.log_dir)

    normal_test_dir, abnormal_dir, mask_dir = _resolve_eval_dirs(data_root)

    datamodule = Folder(
        name=f"{data_root.name}_fastflow",
        root=str(data_root),
        normal_dir="train/good",
        normal_test_dir=normal_test_dir,
        abnormal_dir=abnormal_dir,
        mask_dir=mask_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=8,
        extensions=(".png",),
    )

    model = _build_model(args.backbone, args.radimagenet_ckpt)
    if hasattr(model, "evaluator") and isinstance(model.evaluator, Evaluator):
        if hasattr(model.evaluator, "pixel_metrics"):
            model.evaluator.pixel_metrics = torch.nn.ModuleList([])

        original_configure = getattr(model, "configure_callbacks", None)

        def _configure_callbacks_without_evaluator():
            callbacks = original_configure() if callable(original_configure) else []
            return [cb for cb in callbacks if not isinstance(cb, Evaluator)]

        model.configure_callbacks = _configure_callbacks_without_evaluator

    if args.gpu >= 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = [args.gpu]
    else:
        accelerator = "cpu"
        devices = 1

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        logger=CSVLogger(save_dir=args.log_dir, name="fastflow_logs"),
    )

    trainer.fit(model=model, datamodule=datamodule)

    log_dir = None
    logger = getattr(trainer, "logger", None)
    if logger is not None and getattr(logger, "log_dir", None):
        log_dir = Path(logger.log_dir)
    if log_dir is None:
        log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = log_dir / "training_run_metadata.json"
    epochs_completed = trainer.current_epoch + 1 if trainer.current_epoch is not None else args.epochs
    metadata = {
        "data_root": str(data_root.resolve()),
        "backbone": args.backbone,
        "radimagenet_ckpt": args.radimagenet_ckpt,
        "epochs_target": args.epochs,
        "epochs_completed": epochs_completed,
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


if __name__ == "__main__":
    main()

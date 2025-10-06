import argparse
import os
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger

from anomalib.models.image.cflow import Cflow
from anomalib.data import Folder
from radimagenet_utils import load_radimagenet_resnet_weights


DEFAULT_DATA_ROOT = "/local/scratch/koepchen/synth23_pelvis_v6_png"
DEFAULT_LOG_DIR = "/home/user/koepchen/OOD_tests"


def _build_model(backbone: str, radimagenet_ckpt: str | None) -> Cflow:
    name = backbone.lower()
    suffix = name.replace("radimagenet", "").strip("-_ ") or "resnet50"
    mapping = {"resnet50": "resnet50", "50": "resnet50", "resnet18": "resnet18", "18": "resnet18"}
    target = mapping.get(suffix, suffix)

    model = Cflow(backbone=target, pre_trained=False)

    if target.startswith("resnet") and radimagenet_ckpt:
        feature_extractor = model.model.encoder.feature_extractor
        load_radimagenet_resnet_weights(feature_extractor, radimagenet_ckpt, strict=False)

    return model


def _resolve_eval_dirs(root: Path) -> tuple[str, str, str | None]:
    valid_good = root / "valid" / "good" / "img"
    valid_bad = root / "valid" / "Ungood" / "img"
    valid_mask = root / "valid" / "Ungood" / "label"
    if valid_good.exists() and valid_bad.exists():
        mask_dir = "valid/Ungood/label" if valid_mask.exists() else None
        return "valid/good/img", "valid/Ungood/img", mask_dir

    test_good = root / "test" / "good" / "img"
    test_bad = root / "test" / "Ungood" / "img"
    test_mask = root / "test" / "Ungood" / "label"
    if test_good.exists() and test_bad.exists():
        mask_dir = "test/Ungood/label" if test_mask.exists() else None
        return "test/good/img", "test/Ungood/img", mask_dir

    return "test/good", "test/Ungood", None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal CFlow training script (PNG only)")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--backbone", type=str, default="radimagenet_resnet50")
    parser.add_argument("--radimagenet_ckpt", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
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
        name=f"{data_root.name}_cflow",
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
        logger=CSVLogger(save_dir=args.log_dir, name="cflow_logs"),
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()

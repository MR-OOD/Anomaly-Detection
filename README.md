# OOD Flow Training Utilities

This repository contains training entry points for flow-based anomaly detection models built on [anomalib](https://github.com/openvinotoolkit/anomalib). They are tailored for datasets organised with separate `img/` and `label/` folders (PNG images plus optional pixel-level masks) and focus on two architectures:

- **CFlow** (`train_cflow.py`)
- **FastFlow** (`train_fastflow.py`)

Both scripts assume a RadImageNet-pretrained backbone but can be used with other backbones exposed by anomalib.

## Dataset Layout

Each dataset root is expected to follow the anomalib folder convention:

```
<data_root>/
  train/
    good/
      img/*.png
      label/*.png

  valid/                
    good/img/*.png
    good/label/*.png
    Ungood/img/*.png
    Ungood/label/*.png  
  test/                 
    good/img/*.png
    good/label/*.png
    Ungood/img/*.png
    Ungood/label/*.png
```

If `valid/` exists it is used for evaluation; otherwise the scripts fall back to `test/`. Masks in `label/` are optional but enable pixel-level metrics during FastFlow training.

## Common Arguments

Both scripts share a similar CLI:

- `--data_root`: Path to the dataset root (default: `/local/scratch/koepchen/synth23_pelvis_v6_png`).
- `--log_dir`: Base directory where Lightning logs and checkpoints are stored (default: `/home/user/koepchen/OOD_tests`).
- `--backbone`: Backbone identifier. Values like `radimagenet_resnet50`, `radimagenet_resnet18`, or any backbone supported by the respective model.
- `--radimagenet_ckpt`: Optional path to a RadImageNet `.pt` checkpoint to seed the backbone weights.
- `--epochs`: Maximum training epochs.
- `--batch_size`: Training and evaluation batch size.
- `--gpu`: GPU index to use (`-1` forces CPU).

## Training CFlow on PNGs

`train_cflow.py` runs CFlow with PNG inputs using anomalib’s `Folder` datamodule.

Example command:

```
python new_train_cflow.py \
  --data_root /local/scratch/koepchen/synth23_pelvis_v6_png/synth23_pelvis_v6_png \
  --backbone radimagenet_resnet50 \
  --radimagenet_ckpt RadImageNet_pytorch/ResNet50.pt \
  --epochs 10 \
  --batch_size 16 \
  --gpu 0
```

- Logs and checkpoints appear under `<log_dir>/lightning_logs/version_*` by default.
- Since no custom callbacks are registered, Lightning keeps `best.ckpt` (min `val_loss`) and `last.ckpt` in the checkpoints directory.

## Training FastFlow on PNGs with Pixel Metrics

`train_fastflow.py` provides a streamlined FastFlow trainer. It automatically looks for anomaly masks in `label/` directories to enable pixel-level evaluation.

Example command:

```
python train_fastflow.py \
  --data_root /local/scratch/koepchen/synth23_pelvis_v6_png/synth23_pelvis_v6_png \
  --backbone radimagenet_resnet50 \
  --radimagenet_ckpt RadImageNet_pytorch/ResNet50.pt \
  --epochs 50 \
  --batch_size 32 \
  --gpu 2
```

The default Lightning checkpointing policy applies (saves best `val_loss` and the latest weights inside `lightning_logs/version_*/checkpoints/`). If you prefer a custom location or metric, add a `ModelCheckpoint` callback in the script.

## RadImageNet Weight Loading

Both scripts reuse `radimagenet_utils.py` to remap RadImageNet state dicts to standard torchvision ResNet module names before loading them into the anomalib models. Provide the `.pt` file via `--radimagenet_ckpt` or skip the flag to train from scratch.

## Notes

- GPU selection is index based (`--gpu 0` uses the first visible CUDA device). If CUDA is unavailable or `--gpu -1`, training runs on CPU.
- Lightning’s default loggers create CSV metrics inside `<log_dir>/<model>_logs/version_*/metrics.csv` for convenient inspection.
- FastFlow’s evaluator requires masks only when pixel metrics are enabled; absence of `label/` directories will disable pixel metrics automatically inside anomalib.

For more advanced usage (additional callbacks, mixed precision, distributed training), treat these scripts as starting points and extend them with standard Lightning components.


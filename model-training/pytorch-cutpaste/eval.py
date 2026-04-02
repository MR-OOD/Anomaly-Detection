from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
from torch.utils.data import DataLoader
from torchvision import transforms

from data import TrainDataset, ValidDataset, TestDataset
from density import GaussianDensityTorch
from model import ProjectionNet


def get_train_embeds(model, data_root, transform, device):
    train_data = TrainDataset(data_root=data_root, transform=transform)
    dataloader_train = DataLoader(
        train_data,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    train_embed = []
    with torch.no_grad():
        for x in dataloader_train:
            embed, _ = model(x.to(device))
            train_embed.append(embed.cpu())

    train_embed = torch.cat(train_embed)
    return train_embed


def find_optimal_threshold(labels, distances):
    precision, recall, thresholds = precision_recall_curve(labels, distances)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)

    if len(thresholds) == 0:
        return 0.0, 0.0

    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
    optimal_f1 = f1_scores[optimal_idx]
    return optimal_threshold, optimal_f1


def evaluate_f1(labels, distances, threshold):
    preds = (distances > threshold).astype(int)
    f1 = f1_score(labels, preds)
    return f1, preds


def eval_model(modelname, data_root, device="cpu", save_plots=False, size=256, show_training_data=True, model=None, train_embed=None, head_layer=2, density=GaussianDensityTorch()):
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size, size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )

    test_data_eval = ValidDataset(
        data_root=data_root,
        transform=test_transform,
        size=size,
    )

    dataloader_test = DataLoader(
        test_data_eval,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    if model is None:
        print(f"loading model {modelname}")
        head_layers = [512] * head_layer + [128]
        weights = torch.load(modelname, map_location=device)
        classes = weights["out.weight"].shape[0]
        model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

    labels = []
    embeds = []
    with torch.no_grad():
        for sample in dataloader_test:
            img = sample["image"].to(device)
            label = sample["label"]
            embed, _ = model(img)

            embeds.append(embed.cpu())
            labels.append(label.cpu())

    labels = torch.cat(labels).numpy()
    embeds = torch.cat(embeds)

    if train_embed is None:
        train_embed = get_train_embeds(model, data_root, test_transform, device)

    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

    density.fit(train_embed)
    distances = density.predict(embeds)
    if isinstance(distances, torch.Tensor):
        distances = distances.numpy()

    fpr, tpr, _ = roc_curve(labels, distances)
    roc_auc = auc(fpr, tpr)

    optimal_threshold, optimal_f1 = find_optimal_threshold(labels, distances)

    return roc_auc, optimal_threshold, optimal_f1, len(dataloader_test)


def test_model(modelname, data_root, device="cpu", save_plots=True, size=256, show_training_data=True, model=None, train_embed=None, head_layer=2, density=GaussianDensityTorch(), threshold=None):
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size, size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )

    test_data_eval = TestDataset(
        data_root=data_root,
        transform=test_transform,
        size=size,
    )

    dataloader_test = DataLoader(
        test_data_eval,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    if model is None:
        print(f"loading model {modelname}")
        head_layers = [512] * head_layer + [128]
        weights = torch.load(modelname, map_location=device)
        classes = weights["out.weight"].shape[0]
        model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

    labels = []
    embeds = []
    with torch.no_grad():
        for sample in dataloader_test:
            img = sample["image"].to(device)
            label = sample["label"]
            embed, _ = model(img)

            embeds.append(embed.cpu())
            labels.append(label.cpu())

    labels = torch.cat(labels).numpy()
    embeds = torch.cat(embeds)

    if train_embed is None:
        train_embed = get_train_embeds(model, data_root, test_transform, device)

    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

    density.fit(train_embed)
    distances = density.predict(embeds)
    if isinstance(distances, torch.Tensor):
        distances = distances.numpy()

    model_stem = Path(modelname).stem
    eval_dir = Path("eval") / model_stem
    if save_plots:
        eval_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(labels, distances)
    roc_auc = auc(fpr, tpr)

    if threshold is None:
        threshold, _ = find_optimal_threshold(labels, distances)
        print(f"No threshold provided, using optimal threshold from test set: {threshold:.4f}")

    test_f1, _ = evaluate_f1(labels, distances, threshold)

    if save_plots:
        roc_auc = plot_roc(
            labels,
            distances,
            eval_dir / "roc_plot_exp1.png",
            modelname=modelname,
            save_plots=save_plots,
        )

    return roc_auc, test_f1, len(dataloader_test)


def plot_roc(labels, scores, filename, modelname="", save_plots=True):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    if save_plots:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.legend(loc="lower right")
        plt.savefig(filename)
        plt.close()

    return roc_auc
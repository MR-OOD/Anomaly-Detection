from pathlib import Path
import datetime
import argparse
import yaml

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms

from data import TrainDataset
from dataset import Repeat
from cutpaste import CutPasteNormal, CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn
from model import ProjectionNet
from eval import eval_model, test_model


def run_training(
    data_type,
    data_root,
    model_dir,
    epochs,
    pretrained,
    test_epochs,
    freeze_resnet,
    learninig_rate,
    optim_name,
    batch_size,
    head_layer,
    cutpate_type,
    device,
    workers,
    size,
):
    torch.multiprocessing.freeze_support()

    weight_decay = 1e-4
    momentum = 0.9
    model_name = f"model-{data_type}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())

    min_scale = 0.5

    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    after_cutpaste_transform.transforms.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )

    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale, 1)))
    train_transform.transforms.append(transforms.GaussianBlur(int(size / 10), sigma=(0.1, 2.0)))
    train_transform.transforms.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.1))
    train_transform.transforms.append(cutpate_type(transform=after_cutpaste_transform))

    train_data = TrainDataset(data_root=data_root, transform=train_transform)

    dataloader_kwargs = dict(
        dataset=Repeat(train_data, 3000),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=workers,
        collate_fn=cut_paste_collate_fn,
        pin_memory=True,
    )
    if workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 5

    dataloader = DataLoader(**dataloader_kwargs)

    head_layers = [512] * head_layer + [128]
    num_classes = 2 if cutpate_type is not CutPaste3Way else 3
    model = ProjectionNet(pretrained=pretrained, head_layers=head_layers, num_classes=num_classes)
    model.to(device)

    if freeze_resnet > 0 and pretrained:
        model.freeze_resnet()

    loss_fn = torch.nn.CrossEntropyLoss()
    if optim_name.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learninig_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)
    elif optim_name.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learninig_rate,
            weight_decay=weight_decay,
        )
        scheduler = None
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")

    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
                yield out

    dataloader_inf = get_data_inf()

    for step in range(epochs):
        epoch = int(step / 1)
        if epoch == freeze_resnet:
            model.unfreeze()

        _, data = next(dataloader_inf)
        xs = [x.to(device) for x in data]

        optimizer.zero_grad()

        xc = torch.cat(xs, axis=0)
        embeds, logits = model(xc)

        y = torch.arange(len(xs), device=device)
        y = y.repeat_interleave(xs[0].size(0))
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step(epoch)

        model.eval()

        valid_auc, optimal_threshold, valid_f1, len_valid = eval_model(
            model_name,
            data_root=data_root,
            device=device,
            save_plots=False,
            size=size,
            show_training_data=False,
            model=model,
            head_layer=head_layer,
        )

        test_auc, test_f1, len_test = test_model(
            model_name,
            data_root=data_root,
            device=device,
            save_plots=True,
            size=size,
            show_training_data=False,
            model=model,
            head_layer=head_layer,
            threshold=optimal_threshold,
        )

        model.train()

        print(
            f"[epoch {step} loss: {loss.item()} "
            f"valid_auc (f{len_valid}): {valid_auc} "
            f"valid_f1: {valid_f1} "
            f"test_auc (f{len_test}): {test_auc} "
            f"test_f1: {test_f1}]"
        )

    torch.save(model.state_dict(), model_dir / f"{model_name}.tch")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="custom")
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--no-pretrained',
        dest='pretrained',
        default=True,
        action='store_false',
        help='use pretrained values to initialize ResNet18',
    )

    args = parser.parse_args()
    print(args)

    config_path = args.config
    print(f"reading config {config_path}...")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config)

    variant_map = {
        'normal': CutPasteNormal,
        'scar': CutPasteScar,
        '3way': CutPaste3Way,
        'union': CutPasteUnion,
    }
    variant = variant_map[config['model']['variant']]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    model_dir = Path(config['train']['output_root'])
    model_dir.mkdir(exist_ok=True, parents=True)

    data_type = args.type
    print(f"training {data_type}")
    print(f"data_root: {args.data_root}")

    run_training(
        data_type=data_type,
        data_root=args.data_root,
        model_dir=model_dir,
        epochs=config['train']['epochs'],
        pretrained=config['train']['weights'] if args.pretrained else False,
        test_epochs=config['train']['test_epochs'],
        freeze_resnet=config['train']['freeze_resnet'],
        learninig_rate=config['train']['lr'],
        optim_name=config['train']['optim'],
        batch_size=config['train']['batch_size'],
        head_layer=config['model']['head_layer'],
        device=device,
        cutpate_type=variant,
        workers=config['train']['workers'],
        size=256,
    )
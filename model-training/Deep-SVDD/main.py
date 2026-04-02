import click
import torch
import logging
import random
import numpy as np
import os
import yaml

from sklearn.metrics import roc_curve, auc

from data import TrainDataset, ValidDataset, get_train_transforms, get_valid_transforms, TestDataset
from deepSVDD import DeepSVDD


@click.command()
@click.argument('dataset_name', type=str)
@click.argument('net_name', type=click.Choice([
    'mnist_LeNet',
    'cifar10_LeNet',
    'cifar10_LeNet_ELU',
    'resnet50_rad',
    'resnet50_imagenet'
]))
@click.argument('xp_path')
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', '--config_path', 'load_config',
              type=click.Path(exists=True), default=None,
              help='YAML config file path.')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--device', type=str, default='cuda',
              help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--pretrain', type=bool, default=None,
              help='Override whether to pretrain via autoencoder.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading.')
@click.option('--normal_class', type=int, default=0,
              help='Unused placeholder kept for compatibility.')
def main(dataset_name, net_name, xp_path, data_path, load_config, load_model,
         device, pretrain, n_jobs_dataloader, normal_class):
    config_path = load_config or "config/deepsvdd.yaml"
    print(f"reading config {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(config)

    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})

    objective = model_cfg["objective"]
    nu = model_cfg["nu"]
    optimizer_name = model_cfg["optimizer_name"]
    ae_optimizer_name = model_cfg["ae_optimizer_name"]

    lr = train_cfg["lr"]
    n_epochs = train_cfg["n_epochs"]
    lr_milestone = train_cfg["lr_milestones"]
    batch_size = train_cfg["batch_size"]
    weight_decay = train_cfg["weight_decay"]

    ae_lr = train_cfg["ae_lr"]
    ae_n_epochs = train_cfg["ae_n_epochs"]
    ae_lr_milestone = train_cfg["ae_lr_milestone"]
    ae_batch_size = train_cfg["ae_batch_size"]
    ae_weight_decay = train_cfg["ae_weight_decay"]

    seed = train_cfg.get("seed", -1)

    if pretrain is None:
        pretrain = bool(model_cfg.get("pretrained", True))

    os.makedirs(xp_path, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = os.path.join(xp_path, 'log4.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    logger.addHandler(file_handler)

    logger.info('Loaded configuration from %s.', config_path)
    logger.info('Deep SVDD objective: %s', objective)
    logger.info('Nu-parameter: %.2f', nu)

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info('Set seed to %d.', seed)

    if not torch.cuda.is_available():
        device = 'cpu'

    logger.info('Computation device: %s', device)
    logger.info('Number of dataloader workers: %d', n_jobs_dataloader)
    logger.info('Dataset tag: %s', dataset_name)
    logger.info('Dataset path: %s', data_path)

    train_dataset = TrainDataset(data_path, transform=get_train_transforms())
    valid_dataset = ValidDataset(data_path, transform=get_valid_transforms())
    test_dataset = TestDataset(data_path, transform=get_valid_transforms())

    logger.info('Train samples: %d', len(train_dataset))
    logger.info('Valid samples: %d', len(valid_dataset))
    logger.info('Test samples: %d', len(test_dataset))

    deep_SVDD = DeepSVDD(objective, nu)
    deep_SVDD.set_network(net_name)

    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.', load_model)

    logger.info('Pretraining: %s', pretrain)
    if pretrain:
        logger.info('Pretraining optimizer: %s', ae_optimizer_name)
        logger.info('Pretraining learning rate: %g', ae_lr)
        logger.info('Pretraining epochs: %d', ae_n_epochs)
        logger.info('Pretraining learning rate scheduler milestones: %s', (ae_lr_milestone,))
        logger.info('Pretraining batch size: %d', ae_batch_size)
        logger.info('Pretraining weight decay: %g', ae_weight_decay)

        deep_SVDD.pretrain(
            train_dataset,
            optimizer_name=ae_optimizer_name,
            lr=ae_lr,
            n_epochs=ae_n_epochs,
            lr_milestones=(ae_lr_milestone,),
            batch_size=ae_batch_size,
            weight_decay=ae_weight_decay,
            device=device,
            n_jobs_dataloader=n_jobs_dataloader
        )

    logger.info('Training optimizer: %s', optimizer_name)
    logger.info('Training learning rate: %g', lr)
    logger.info('Training epochs: %d', n_epochs)
    logger.info('Training learning rate scheduler milestones: %s', (lr_milestone,))
    logger.info('Training batch size: %d', batch_size)
    logger.info('Training weight decay: %g', weight_decay)

    deep_SVDD.train(
        train_dataset,
        optimizer_name=optimizer_name,
        lr=lr,
        n_epochs=n_epochs,
        lr_milestones=(lr_milestone,),
        batch_size=batch_size,
        weight_decay=weight_decay,
        device=device,
        n_jobs_dataloader=n_jobs_dataloader
    )

    logger.info('=' * 50)
    logger.info('Testing on validation set...')
    deep_SVDD.test(
        valid_dataset,
        device=device,
        n_jobs_dataloader=n_jobs_dataloader
    )
    logger.info('Validation-set AUC (reported via test API): %.4f', deep_SVDD.results['test_auc'])

    logger.info('=' * 50)
    logger.info('Testing on test set...')
    deep_SVDD.test(
        test_dataset,
        device=device,
        n_jobs_dataloader=n_jobs_dataloader
    )

    logger.info('Test Results:')
    logger.info('  AUC: %.4f', deep_SVDD.results['test_auc'])
    logger.info('=' * 50)

    labels, scores = zip(*deep_SVDD.results['test_scores'])
    labels, scores = np.array(labels), np.array(scores)

    deep_SVDD.save_results(export_json=os.path.join(xp_path, 'results.json'))
    deep_SVDD.save_model(export_model=os.path.join(xp_path, f'model_{dataset_name}.tar'))

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    logger.info("ROC AUC on test set: %.4f", roc_auc)

    with open(os.path.join(xp_path, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test AUC: {roc_auc:.4f}\n")

    logger.info('Test metrics saved to %s', os.path.join(xp_path, 'test_metrics.txt'))


if __name__ == '__main__':
    main()
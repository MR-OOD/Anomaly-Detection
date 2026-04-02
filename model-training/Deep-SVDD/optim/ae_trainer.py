from base.base_trainer import BaseTrainer
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class AETrainer(BaseTrainer):

    def __init__(
        self,
        optimizer_name: str = 'adam',
        lr: float = 0.001,
        n_epochs: int = 150,
        lr_milestones: tuple = (),
        batch_size: int = 128,
        weight_decay: float = 1e-6,
        device: str = 'cuda',
        n_jobs_dataloader: int = 0
    ):
        super().__init__(
            optimizer_name,
            lr,
            n_epochs,
            lr_milestones,
            batch_size,
            weight_decay,
            device,
            n_jobs_dataloader
        )

    def train(self, dataset, ae_net: BaseNet):
        logger = logging.getLogger()

        ae_net = ae_net.to(self.device)

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs_dataloader
        )

        optimizer = optim.Adam(
            ae_net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=self.optimizer_name == 'amsgrad'
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.lr_milestones,
            gamma=0.1
        )

        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()

        for epoch in range(self.n_epochs):
            # 这里保留你原来的调度器调用位置，虽然有 warning，
            # 但先不改行为，避免影响结果复现
            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            for sample in train_loader:
                inputs = sample['image'].to(self.device)

                optimizer.zero_grad()

                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            epoch_train_time = time.time() - epoch_start_time
            logger.info(
                '  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(
                    epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches
                )
            )

        pretrain_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' % pretrain_time)
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset, ae_net: BaseNet):
        logger = logging.getLogger()

        ae_net = ae_net.to(self.device)

        # 关键修复：直接使用传进来的 dataset，不要再覆盖成 ValidDataset('camelyon', ...)
        valid_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs_dataloader
        )

        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []

        ae_net.eval()
        with torch.no_grad():
            for sample in valid_dataloader:
                inputs = sample['image'].to(self.device)
                labels = sample['label'].to(self.device)

                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                idx_label_score += list(
                    zip(
                        labels.cpu().data.numpy().tolist(),
                        scores.cpu().data.numpy().tolist()
                    )
                )

                loss_epoch += loss.item()
                n_batches += 1

        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * auc))

        test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')
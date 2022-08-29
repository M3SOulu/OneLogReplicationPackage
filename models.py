from abc import abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric
from tqdm import tqdm

from torchmetrics.classification import F1Score, Precision, Recall
from losses import MarginLoss, F1ScoreWithEmbeddings, PrecisionWithEmbeddings, RecallWithEmbeddings


class BaseModel(pl.LightningModule):

    def __init__(self, lr=0.001, testset_count=1, loss_type='marginal', mn=1, ma=5, positive_class_weight=1, *args, **kwargs):
        assert loss_type in {'marginal', 'logloss'}
        super().__init__(*args, **kwargs)
        if loss_type == 'marginal':
            self.testset_count = testset_count
            self.train_f1 = F1ScoreWithEmbeddings(mn, ma)
            self.val_loss = MeanMetric()
            self.val_f1 = F1ScoreWithEmbeddings(mn, ma)
            for i in range(testset_count):
                setattr(self, f'test_precision_{i}', PrecisionWithEmbeddings(mn, ma))
                setattr(self, f'test_recall_{i}', RecallWithEmbeddings(mn, ma))
                setattr(self, f'test_f1_{i}', F1ScoreWithEmbeddings(mn, ma))
            self.criterion = MarginLoss(mn, ma)
        elif loss_type == 'logloss':
            self.testset_count = testset_count
            self.train_f1 = F1Score(threshold=0)
            self.val_loss = MeanMetric()
            self.val_f1 = F1Score(threshold=0)
            for i in range(testset_count):
                setattr(self, f'test_precision_{i}', Precision(threshold=0))
                setattr(self, f'test_recall_{i}', Recall(threshold=0))
                setattr(self, f'test_f1_{i}', F1Score(threshold=0))
            self.criterion = nn.BCEWithLogitsLoss(torch.tensor([positive_class_weight]))
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, l, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.train_f1(y_hat, y.int())
        self.log("f1", self.train_f1.compute(), prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.log("training_epoch_f1", self.train_f1.compute(), sync_dist=True)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, l, y = batch
        y_hat = self(x)
        if dataloader_idx == 0:  # validation dataset
            loss = self.criterion(y_hat, y)
            self.val_loss.update(loss)
            self.val_f1.update(y_hat, y.int())
            return {"val_loss": self.val_loss.compute(), "val_f1": self.val_f1.compute()}
        else:  # test datasets
            p = getattr(self, f'test_precision_{dataloader_idx - 1}')
            r = getattr(self, f'test_recall_{dataloader_idx - 1}')
            f1 = getattr(self, f'test_f1_{dataloader_idx - 1}')
            p.update(y_hat, y.int())
            r.update(y_hat, y.int())
            f1.update(y_hat, y.int())
            return {
                f"test_precision_{self.trainer.datamodule.datasets_test[dataloader_idx - 1].name}": p.compute(),
                f"test_recall_{self.trainer.datamodule.datasets_test[dataloader_idx - 1].name}": r.compute(),
                f"test_f1_{self.trainer.datamodule.datasets_test[dataloader_idx - 1].name}": f1.compute()
            }

    def validation_epoch_end(self, outputs) -> None:
        self.log("val_f1", self.val_f1.compute(), prog_bar=True, sync_dist=True)
        for i in range(self.testset_count):
            p = getattr(self, f'test_precision_{i}')
            r = getattr(self, f'test_recall_{i}')
            f1 = getattr(self, f'test_f1_{i}')
            self.log(f'test_precision_{self.trainer.datamodule.datasets_test[i].name}', p.compute(), sync_dist=True)
            self.log(f'test_recall_{self.trainer.datamodule.datasets_test[i].name}', r.compute(), sync_dist=True)
            self.log(f'test_f1_{self.trainer.datamodule.datasets_test[i].name}', f1.compute(), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.lr)

    def predict(self, data_loader):
        results = []
        with tqdm(total=len(data_loader)) as tbar:
            for batch in data_loader:
                x, l = batch
                y_hat = F.sigmoid(self(x.to(self.device), l)).cpu().numpy()
                results.append(y_hat)
                tbar.update()
        return np.concatenate(results, axis=0)

    @property
    def loss_type(self):
        if isinstance(self.criterion, MarginLoss):
            return 'marginal'
        elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
            return 'logloss'
        else:
            raise ValueError("Unrecognized loss type")

    @abstractmethod
    def character_embedding_vectors(self):
        pass


class HierarchicalCnnModel(BaseModel):
    min_input_shape = (6, 15)

    def __init__(self, max_embedding_indices, dims=32, dropout=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.character_embedding = nn.Embedding(max_embedding_indices, dims, 0)
        self.message_embedding = nn.Sequential()
        self.sequence_embedding = nn.Sequential()
        self.message_embedding = nn.Sequential(
            nn.Conv2d(dims, dims, (1, 3), stride=(1, 2), bias=False),
            nn.BatchNorm2d(dims),
            nn.ReLU(),
            # nn.Dropout2d(0.5),

            nn.Conv2d(dims, dims, (1, 3), stride=(1, 2), bias=False),
            nn.BatchNorm2d(dims),
            nn.ReLU(),
            # nn.Dropout2d(0.5),

            nn.Conv2d(dims, dims * 2, (1, 3), stride=(1, 2), bias=False),
            nn.BatchNorm2d(dims * 2),
            nn.ReLU(),
            # nn.Dropout2d(0.5),
        )
        self.sequence_embedding = nn.Sequential(
            nn.Conv1d(dims * 2, dims * 2, 3, stride=1, bias=False),
            nn.BatchNorm1d(dims * 2),
            nn.ReLU(),
            # nn.Dropout(0.5),

            nn.Conv1d(dims * 2, dims * 3, 3, stride=1, bias=False),
            nn.BatchNorm1d(dims * 3),
            nn.ReLU(),
            # nn.Dropout(0.5),

            nn.Conv1d(dims * 3, dims * 4, 2, stride=1, bias=False),
            nn.BatchNorm1d(dims * 4),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.classification = torch.nn.Sequential(
            nn.Linear(dims * 4, dims * 2),
            nn.ReLU(),
            # nn.Dropout(0.5),

            nn.Linear(dims * 2, dims),
            nn.ReLU(),
            # nn.Dropout(0.5),

            nn.Linear(dims, dims if self.loss_type == 'marginal' else 1),
        )

    def forward(self, x: torch.Tensor):
        x = self.character_embedding(x).permute(0, 3, 1, 2)
        x = self.message_embedding(x)
        x, _ = torch.max(x, dim=-1)
        x = self.sequence_embedding(x)
        x, _ = torch.max(x, dim=-1)
        x = self.classification(x)
        x = torch.squeeze(x)
        return x

    def embed_message(self, x: torch.Tensor):
        x = self.character_embedding(x)
        x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        x = self.message_embedding(x)
        x, _ = torch.max(x, dim=-1)
        return x[:, :, 0]

    def character_embedding_vectors(self):
        return self.character_embedding.weight.data.cpu().numpy()

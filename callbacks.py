import csv
import os
from typing import List, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import F1, Precision, Recall
import tqdm

from utils import export_characters_tensorflow_projector_visualization


class OverallTestDataset(pl.Callback):
    def __init__(self):
        self.first_zero_epoch = False

    def on_validation_end(self, trainer, pl_module):
        if not self.first_zero_epoch:
            self.first_zero_epoch = True
            return
        print("\n" * 2, "-" * 40)
        print("Overall Test Callback")
        print("-" * 40)
        row = []
        for name, dl in trainer.datamodule.test_dataloaders():
            print("Testing on", name, ":")
            test_results = self.evaluate(pl_module, dl)
            print(name, "test results:\n", test_results)
            row.append(test_results[0])
        print("=" * 40, "\n")
        self.add_row_to_history(trainer.default_root_dir, row)

    @staticmethod
    def add_row_to_history(directory, row):
        with open(os.path.join(directory, 'history.csv'), 'a') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    @staticmethod
    def evaluate(model, dataloader):
        device = model.device
        f1, precision, recall = F1().to(device), Precision().to(device), Recall().to(device)
        with torch.no_grad():
            for x, l, y_true in tqdm.tqdm(dataloader):
                x, y_true = x.to(device), y_true.to(device).int()
                y_pred = model(x)
                precision.update(y_pred, y_true)
                recall.update(y_pred, y_true)
                f1.update(y_pred, y_true)

        return f1.compute().item(), precision.compute().item(), recall.compute().item()


class CharacterEmbeddingTensorflowExport(pl.Callback):
    def __init__(self, encoder):
        self.encoder = encoder

    def on_validation_end(self, trainer, pl_module):
        export_dir = os.path.join(trainer.default_root_dir, "visualization", f"epoch_{trainer.current_epoch}")
        if not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        export_vec = os.path.join(export_dir, "vectors.tsv")
        export_met = os.path.join(export_dir, "metadata.tsv")
        export_characters_tensorflow_projector_visualization(pl_module, self.encoder, export_vec,
                                                             export_met)

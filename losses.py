import torch
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.classification.precision_recall import Precision, Recall


class PrecisionWithEmbeddings(Precision):
    def __init__(self, mn, ma, *args, **kwargs):
        self.mn = 1
        self.ma = 5
        super().__init__(*args, **kwargs, threshold=(ma + mn) / 2)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        norm_pred = torch.linalg.norm(preds, dim=-1)
        super().update(norm_pred, target)


class RecallWithEmbeddings(Recall):
    def __init__(self, mn, ma, *args, **kwargs):
        self.mn = 1
        self.ma = 5
        super().__init__(*args, **kwargs, threshold=(ma + mn) / 2)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        norm_pred = torch.linalg.norm(preds, dim=-1)
        super().update(norm_pred, target)


class F1ScoreWithEmbeddings(F1Score):
    def __init__(self, mn, ma, *args, **kwargs):
        self.mn = 1
        self.ma = 5
        super().__init__(*args, **kwargs, threshold=(ma + mn) / 2)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        norm_pred = torch.linalg.norm(preds, dim=-1)
        super().update(norm_pred, target)


class MarginLoss(torch.nn.Module):
    def __init__(self, mn=1, ma=5):
        super().__init__()
        self.mn = mn
        self.ma = ma

    def forward(self, v, l):
        norm_v = torch.linalg.norm(v, dim=-1)
        zeros = torch.zeros_like(norm_v, requires_grad=False)
        j = torch.square((1 - l) * torch.maximum(norm_v - self.mn, zeros)) + torch.square(
            l * torch.minimum(norm_v - self.ma, zeros))
        return j.mean()

    def infer(self, v):
        norm_v = torch.linalg.norm(v, dim=-1)
        return (norm_v > ((self.ma - self.mn) / 2)).float()

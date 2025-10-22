import torch
import torch.nn.functional as F


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta)
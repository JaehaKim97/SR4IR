import torch.nn as nn

from torch.nn import functional as F


class AUXCELoss(nn.Module):
    """Cross Entropy loss with Auxilary loss.

    Args:
        loss_weight (float): Loss weight for CE loss. Default: 1.0.
        loss_weight (float): Loss weight for auxilrary loss. Default: 0.5.
    """
    def __init__(self, loss_weight=1.0, aux_loss_weight=0.5, ignore_index=-100):
        super().__init__()
        self.loss_weight = loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.ignore_index = ignore_index
    
    def forward(self, pred, target, **kwargs):
        losses = {}
        for name, x in pred.items():
            losses[name] = F.cross_entropy(x, target, ignore_index=self.ignore_index)

        if len(losses) == 1:
            return self.loss_weight * losses["out"]

        return self.loss_weight * (losses["out"] + self.aux_loss_weight * losses["aux"])

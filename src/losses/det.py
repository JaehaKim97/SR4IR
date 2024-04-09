import torch.nn as nn

class DETLoss(nn.Module):
    """Detection loss.

    Args:
        loss_weight (float): Loss weight for CE loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(DETLoss, self).__init__()
        self.loss_weight = loss_weight        

    def forward(self, loss_dict, **kwargs):
        losses = sum(loss for loss in loss_dict.values())
        return self.loss_weight * losses
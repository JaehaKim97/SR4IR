from torch import nn as nn


class CELoss(nn.Module):
    """Cross Entropy loss.

    Args:
        loss_weight (float): Loss weight for CE loss. Default: 1.0.
        ignore_idx (int): Specifies a target value that is ignored and
            does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, label_smoothing=0.0, reduction='mean', ignore_index=-100):
        super(CELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: [none, mean, sum]')

        self.loss_weight = loss_weight
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=ignore_index, reduction=reduction)
        

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return self.loss_weight * self.cross_entropy(pred, target)

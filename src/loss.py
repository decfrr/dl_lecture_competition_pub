import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha (float): Weighting factor for the class (1-alpha for the other class)
            gamma (float): Focusing parameter to down-weight easy samples and focus on hard ones
            reduction (str): Reduction method to apply to the output ('mean', 'sum', or 'none')
        """
        super(FocalCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Predictions from the model (logits)
            targets (torch.Tensor): Ground truth labels (class indices)

        Returns:
            torch.Tensor: Computed focal cross-entropy loss
        """
        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute softmax over the predictions
        probs = torch.softmax(inputs, dim=1)

        # Gather the probabilities corresponding to the correct labels
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).type_as(inputs)
        p_t = (probs * targets_one_hot).sum(dim=1)

        # Compute the focal loss component
        focal_weight = self.alpha * (1 - p_t) ** self.gamma

        # Apply the focal weight to the cross-entropy loss
        focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            loss = focal_loss.mean()
        elif self.reduction == 'sum':
            loss = focal_loss.sum()
        else:
            loss = focal_loss

        return loss



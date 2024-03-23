import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        mask_pred = (inputs > 0.5).float()
        intersection = (mask_pred * targets).sum()

        dice_loss = 1 - (2.*intersection + smooth)/(mask_pred.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets)
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

def dice_coefficient(predicted, target, smooth=1e-5):
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice
import torch
import torch.nn as nn
import torch.nn.functional as F


def unpack_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs["main"], outputs.get("aux", [])
    return outputs, []


def soft_dice_loss_from_logits(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def mask_to_boundary_band(mask, radius=1):
    kernel_size = radius * 2 + 1
    dilation = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)
    erosion = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=radius)
    return (dilation - erosion).clamp(0.0, 1.0)


class BoundaryBandConsistencyLoss(nn.Module):
    def forward(self, logits, targets):
        device_type = logits.device.type if logits.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            logits = logits.float()
            targets = targets.float()
            pred_band = mask_to_boundary_band(torch.sigmoid(logits))
            target_band = mask_to_boundary_band(targets)
            bce = F.binary_cross_entropy(pred_band.clamp(1e-6, 1.0 - 1e-6), target_band)

            pred_band = pred_band.flatten(1)
            target_band = target_band.flatten(1)
            intersection = (pred_band * target_band).sum(dim=1)
            denominator = pred_band.sum(dim=1) + target_band.sum(dim=1)
            dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return bce + dice


class BoundaryBandLoss(nn.Module):
    def __init__(self, aux_weight=0.3, boundary_weight=0.2):
        super().__init__()
        self.aux_weight = aux_weight
        self.boundary_weight = boundary_weight
        self.boundary_loss = BoundaryBandConsistencyLoss()

    def seg_loss(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = soft_dice_loss_from_logits(logits, targets)
        return bce + dice

    def forward(self, outputs, targets):
        main_logits, aux_outputs = unpack_outputs(outputs)
        total = self.seg_loss(main_logits, targets)
        if aux_outputs:
            aux_total = 0.0
            for aux_logits in aux_outputs:
                resized_targets = F.interpolate(targets, size=aux_logits.shape[-2:], mode="nearest")
                aux_total = aux_total + self.seg_loss(aux_logits, resized_targets)
            total = total + self.aux_weight * aux_total / len(aux_outputs)

        total = total + self.boundary_weight * self.boundary_loss(main_logits, targets)
        return total


class BinarySegLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, outputs, targets):
        logits, _ = unpack_outputs(outputs)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = soft_dice_loss_from_logits(logits, targets)
        total = self.bce_weight * bce + self.dice_weight * dice
        return total

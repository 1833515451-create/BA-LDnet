import torch.nn as nn
import torch.nn.functional as F

from losses import mask_to_boundary_band, soft_dice_loss_from_logits, unpack_outputs


class SingleModuleSegLoss(nn.Module):
    def __init__(self, aux_weight=0.2, boundary_weight=0.08, tiny_area=0.002, main_boundary_weight=0.0, boundary_radius=1):
        super().__init__()
        self.aux_weight = aux_weight
        self.boundary_weight = boundary_weight
        self.tiny_area = tiny_area
        self.main_boundary_weight = main_boundary_weight
        self.boundary_radius = boundary_radius

    def seg_loss(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets) + soft_dice_loss_from_logits(logits, targets)

    def resize_target(self, targets, logits):
        return F.interpolate(targets, size=logits.shape[-2:], mode="nearest") if logits.shape[-2:] != targets.shape[-2:] else targets

    def erode(self, targets, kernel_size=5):
        pad = kernel_size // 2
        return -F.max_pool2d(-targets, kernel_size=kernel_size, stride=1, padding=pad)

    def weighted_bce_dice(self, logits, targets, weight=None):
        if weight is None:
            return self.seg_loss(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, weight=weight)
        return bce + soft_dice_loss_from_logits(logits, targets)

    def boundary_loss(self, logits, targets):
        target_band = mask_to_boundary_band(targets, self.boundary_radius)
        target_band = self.resize_target(target_band, logits)
        return F.binary_cross_entropy_with_logits(logits, target_band) + soft_dice_loss_from_logits(logits, target_band)

    def rim_loss(self, logits, targets):
        inner_rim = (targets - self.erode(targets, kernel_size=5)).clamp(0.0, 1.0)
        inner_rim = self.resize_target(inner_rim, logits)
        return self.weighted_bce_dice(logits, inner_rim)

    def low_contrast_edge_loss(self, outputs, targets):
        logits = outputs["low_contrast_edge"]
        target_band = self.resize_target(mask_to_boundary_band(targets, self.boundary_radius), logits)
        prior = outputs.get("low_contrast_prior")
        if prior is not None and prior.shape[-2:] != logits.shape[-2:]:
            prior = F.interpolate(prior, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        weight = 1.0 + prior.clamp(0.0, 1.0) if prior is not None else None
        return self.weighted_bce_dice(logits, target_band, weight)

    def tiny_recall_loss(self, logits, targets):
        targets = self.resize_target(targets, logits)
        area = targets.flatten(1).mean(dim=1)
        tiny_mask = (area <= self.tiny_area).float().view(-1, 1, 1, 1)
        tiny_count = tiny_mask.sum().clamp_min(1.0)
        if tiny_mask.sum().item() <= 0:
            return logits.sum() * 0.0
        probs = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = (probs * (1.0 + 2.0 * targets) * tiny_mask).sum() / tiny_count / targets.shape[-1] / targets.shape[-2]
        pred = F.logsigmoid(logits).exp()
        tp = (pred * targets).flatten(1).sum(dim=1)
        fp = (pred * (1.0 - targets)).flatten(1).sum(dim=1)
        fn = ((1.0 - pred) * targets).flatten(1).sum(dim=1)
        tversky = 1.0 - ((tp + 1.0) / (tp + 0.3 * fp + 0.7 * fn + 1.0))
        return bce + (tversky * tiny_mask.flatten()).sum() / tiny_count

    def transition_loss(self, logits, targets):
        inner_wide = (targets - self.erode(targets, kernel_size=9)).clamp(0.0, 1.0)
        inner_core = (targets - self.erode(targets, kernel_size=3)).clamp(0.0, 1.0)
        transition = (0.7 * inner_wide + 0.3 * inner_core).clamp(0.0, 1.0)
        transition = self.resize_target(transition, logits)
        return self.weighted_bce_dice(logits, transition, 1.0 + transition)

    def forward(self, outputs, targets):
        main_logits, aux_outputs = unpack_outputs(outputs)
        total = self.seg_loss(main_logits, targets)
        if self.main_boundary_weight > 0:
            total = total + self.main_boundary_weight * self.boundary_loss(main_logits, targets)
        if aux_outputs:
            aux_total = 0.0
            for aux_logits in aux_outputs:
                aux_targets = F.interpolate(targets, size=aux_logits.shape[-2:], mode="nearest")
                aux_total = aux_total + self.seg_loss(aux_logits, aux_targets)
            total = total + self.aux_weight * aux_total / len(aux_outputs)
        if isinstance(outputs, dict) and "boundary" in outputs:
            total = total + self.boundary_weight * self.boundary_loss(outputs["boundary"], targets)
        if isinstance(outputs, dict) and "rim" in outputs:
            total = total + self.boundary_weight * self.rim_loss(outputs["rim"], targets)
        if isinstance(outputs, dict) and "low_contrast_edge" in outputs:
            total = total + self.boundary_weight * self.low_contrast_edge_loss(outputs, targets)
        if isinstance(outputs, dict) and "tiny" in outputs:
            total = total + self.boundary_weight * self.tiny_recall_loss(outputs["tiny"], targets)
        if isinstance(outputs, dict) and "transition" in outputs:
            total = total + self.boundary_weight * self.transition_loss(outputs["transition"], targets)
        return total

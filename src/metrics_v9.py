import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from metrics import evaluate_batch


def _dice_from_binary(pred, target):
    inter = (pred * target).flatten(1).sum(dim=1)
    denom = pred.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    return (2.0 * inter / (denom + 1e-8)).detach().cpu()


def _boundary(mask, radius=2):
    dilated = F.max_pool2d(mask.float(), kernel_size=radius * 2 + 1, stride=1, padding=radius)
    eroded = -F.max_pool2d(-mask.float(), kernel_size=radius * 2 + 1, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def _surface(mask):
    mask = mask.astype(np.uint8)
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return (mask - eroded).astype(np.uint8)


def _surface_distances(pred, target):
    pred_surface = _surface(pred)
    target_surface = _surface(target)
    h, w = target.shape
    diagonal = math.sqrt(h * h + w * w)
    if pred_surface.sum() == 0 and target_surface.sum() == 0:
        return np.array([0.0], dtype=np.float32)
    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return np.array([diagonal], dtype=np.float32)
    target_dist = cv2.distanceTransform((1 - target_surface).astype(np.uint8), cv2.DIST_L2, 5)
    pred_dist = cv2.distanceTransform((1 - pred_surface).astype(np.uint8), cv2.DIST_L2, 5)
    return np.concatenate([target_dist[pred_surface > 0], pred_dist[target_surface > 0]]).astype(np.float32)


def evaluate_v9_batch(prob_map, target, threshold=0.5, small_area=0.01, tiny_area=0.002, boundary_radius=2, surface_metrics=False):
    metrics = evaluate_batch(prob_map, target, threshold=threshold)
    counts = {name: int(prob_map.shape[0]) for name in metrics}
    pred = (prob_map > threshold).float()
    target = (target > 0.5).float()
    areas = target.flatten(1).mean(dim=1)
    dice_values = _dice_from_binary(pred, target)
    small_mask = (areas > 0) & (areas < small_area)
    tiny_mask = (areas > 0) & (areas < tiny_area)
    metrics["small_dice"] = float(dice_values[small_mask.cpu()].mean().item()) if small_mask.any() else float("nan")
    metrics["tiny_dice"] = float(dice_values[tiny_mask.cpu()].mean().item()) if tiny_mask.any() else float("nan")
    counts["small_dice"] = int(small_mask.sum().item())
    counts["tiny_dice"] = int(tiny_mask.sum().item())
    pred_boundary = _boundary(pred, boundary_radius)
    target_boundary = _boundary(target, boundary_radius)
    tp = (pred_boundary * target_boundary).flatten(1).sum(dim=1)
    precision = tp / (pred_boundary.flatten(1).sum(dim=1) + 1e-8)
    recall = tp / (target_boundary.flatten(1).sum(dim=1) + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    valid_boundary = target_boundary.flatten(1).sum(dim=1) > 0
    metrics["boundary_precision"] = float(precision[valid_boundary].mean().item()) if valid_boundary.any() else float("nan")
    metrics["boundary_recall"] = float(recall[valid_boundary].mean().item()) if valid_boundary.any() else float("nan")
    metrics["boundary_f1"] = float(f1[valid_boundary].mean().item()) if valid_boundary.any() else float("nan")
    counts["boundary_precision"] = counts["boundary_recall"] = counts["boundary_f1"] = int(valid_boundary.sum().item())
    if surface_metrics:
        pred_np = pred[:, 0].detach().cpu().numpy().astype(np.uint8)
        target_np = target[:, 0].detach().cpu().numpy().astype(np.uint8)
        distances = [_surface_distances(p, t) for p, t in zip(pred_np, target_np)]
        hd95 = [float(np.percentile(d, 95)) for d in distances]
        assd = [float(d.mean()) for d in distances]
        metrics["hd95"] = float(np.mean(hd95))
        metrics["assd"] = float(np.mean(assd))
        counts["hd95"] = counts["assd"] = len(distances)
    return metrics, counts

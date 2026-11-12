import torch


def _threshold(x, threshold=0.5):
    return (x > threshold).float()


def _per_image_confusion(output, target, threshold=0.5):
    pred = _threshold(output, threshold).flatten(1)
    target = (target > 0.5).float().flatten(1)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1.0 - target)).sum(dim=1)
    fn = ((1.0 - pred) * target).sum(dim=1)
    tn = ((1.0 - pred) * (1.0 - target)).sum(dim=1)
    return tp, fp, fn, tn


def _macro_mean(values):
    return values.mean().item()


def accuracy(output, target, threshold=0.5):
    tp, fp, fn, tn = _per_image_confusion(output, target, threshold)
    return _macro_mean((tp + tn) / (tp + fp + fn + tn + 1e-8))


def precision(output, target, threshold=0.5):
    tp, fp, _, _ = _per_image_confusion(output, target, threshold)
    return _macro_mean(tp / (tp + fp + 1e-8))


def recall(output, target, threshold=0.5):
    tp, _, fn, _ = _per_image_confusion(output, target, threshold)
    return _macro_mean(tp / (tp + fn + 1e-8))


def specificity(output, target, threshold=0.5):
    _, fp, _, tn = _per_image_confusion(output, target, threshold)
    return _macro_mean(tn / (tn + fp + 1e-8))


def intersection_over_union(output, target, threshold=0.5):
    tp, fp, fn, _ = _per_image_confusion(output, target, threshold)
    return _macro_mean(tp / (tp + fp + fn + 1e-8))


def dice_coefficient(output, target, threshold=0.5):
    tp, fp, fn, _ = _per_image_confusion(output, target, threshold)
    return _macro_mean((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))


def evaluate_batch(prob_map, target, threshold=0.5):
    """Return image-level macro metrics for one batch.

    Callers must weight these batch means by the batch sample count to obtain
    an exact dataset-level image macro average.
    """
    return {
        "accuracy": accuracy(prob_map, target, threshold),
        "precision": precision(prob_map, target, threshold),
        "recall": recall(prob_map, target, threshold),
        "specificity": specificity(prob_map, target, threshold),
        "iou": intersection_over_union(prob_map, target, threshold),
        "dice": dice_coefficient(prob_map, target, threshold),
    }

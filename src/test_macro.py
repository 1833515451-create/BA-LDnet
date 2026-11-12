from argparse import Namespace
import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from metrics import dice_coefficient
from train import run_valid_epoch


def main():
    masks = torch.zeros(3, 1, 16, 16)
    masks[0, 0, 2:7, 2:7] = 1
    masks[1, 0, 2:4, 2:4] = 1
    masks[2, 0, 1:6, 1:6] = 1
    logits = masks * 20 - 10
    logits[1].fill_(-10)
    args = Namespace(max_valid_batches=0, small_area=0.01, tiny_area=0.002, boundary_radius=2)
    device = torch.device('cpu')
    dataset = TensorDataset(logits, masks)
    model = torch.nn.Identity()
    expected = 2 / 3
    assert math.isclose(dice_coefficient(logits.sigmoid(), masks), expected, abs_tol=1e-7)
    results = []
    for batch_size in (1, 2, 3):
        _, metrics = run_valid_epoch(
            model, DataLoader(dataset, batch_size=batch_size), torch.nn.functional.binary_cross_entropy_with_logits,
            device, args, False,
        )
        assert math.isclose(metrics['dice'], expected, abs_tol=1e-7), metrics
        results.append(metrics)
    for metrics in results[1:]:
        for key, value in metrics.items():
            assert math.isclose(value, results[0][key], abs_tol=1e-5, rel_tol=1e-6), (key, value)
    print('Image macro validation: batch sizes 1, 2, and 3 passed.')


if __name__ == '__main__':
    main()

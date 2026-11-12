import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

from augmentations import build_preprocess_fn
from dataset import IMAGE_EXTENSIONS, build_valid_transforms, read_pair_manifest
from losses import unpack_outputs
from metrics_v9 import evaluate_v9_batch
from models import MiTB3SingleModuleNet
from utils import resolve_device


def external_pairs(root):
    def collect(folder):
        pairs = {}
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                if path.stem in pairs:
                    raise ValueError(f'Duplicate stem: {path.stem}')
                pairs[path.stem] = path
        return pairs

    images, masks = collect(root / 'train_images'), collect(root / 'train_masks')
    if not images or images.keys() != masks.keys():
        raise ValueError('Image and mask stems must match.')
    return [(images[key], masks[key]) for key in images]


def read_cv(path, flags):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f'Cannot decode {path}')
    return image


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['dfuc2022', 'fuseg'])
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--module', choices=['baseline', 'boundary_aux'], required=True)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--boundary_radius', type=int, default=2)
    args = parser.parse_args()
    if args.dataset == 'dfuc2022':
        images, masks = read_pair_manifest(root / 'protocols/dfuc2022/validation.csv', args.data_path)
        pairs = zip(images, masks)
    else:
        pairs = external_pairs(args.data_path)

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    saved_args = checkpoint.get('args', {})
    model = MiTB3SingleModuleNet(
        module=args.module,
        encoder_name='mit_b3',
        encoder_weights='none',
        refine_channels=int(saved_args.get('refine_channels', 32)),
    )
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
    model.to(device).eval()
    transform = build_valid_transforms(256)
    preprocess = build_preprocess_fn('mit_b3', 'imagenet')
    rows = []
    with torch.inference_mode():
        for image_path, mask_path in pairs:
            mask = read_cv(mask_path, cv2.IMREAD_GRAYSCALE) > 127
            if args.dataset == 'fuseg' and not mask.any():
                continue
            image = cv2.cvtColor(read_cv(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            sample = transform(image=image, mask=mask.astype(np.float32)[..., None])
            pixels = torch.from_numpy(preprocess(sample['image']).astype(np.float32)).permute(2, 0, 1)
            target = torch.from_numpy(sample['mask']).permute(2, 0, 1)
            logits, _ = unpack_outputs(model(pixels.unsqueeze(0).to(device)))
            metrics, _ = evaluate_v9_batch(
                torch.sigmoid(logits), target.unsqueeze(0).to(device),
                boundary_radius=args.boundary_radius, surface_metrics=True,
            )
            rows.append({'image': image_path.name, **metrics})
    if not rows:
        raise ValueError('No evaluable image pairs.')

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'per_image.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for key in rows[0].keys() - {'image'}:
        values = [r[key] for r in rows if math.isfinite(r[key])]
        summary[key] = mean(values) if values else None
    (args.output_dir / 'image_macro.json').write_text(
        json.dumps({'images': len(rows), 'metrics': summary}, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()

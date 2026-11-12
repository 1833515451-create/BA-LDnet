import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--model', required=True, choices=['mean_teacher', 'ba_ldnet'])
    parser.add_argument('--seed', required=True, type=int, choices=[2026, 3967, 9421])
    parser.add_argument('--label', required=True, type=int, choices=[5, 10, 20])
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()

    runs = json.loads((root / 'configs/training_runs.json').read_text(encoding='utf-8'))
    run = next(row for row in runs if (row['model'], row['seed'], row['label_percent'])
               == (args.model, args.seed, args.label))
    settings = dict(run['parameters'])
    settings.update(
        data_path=args.data_path,
        protocol_dir=root / 'protocols/dfuc2022',
        device=args.device,
        save_dir=args.output_dir / 'checkpoints',
        plot_dir=args.output_dir / 'plots',
        history_path=args.output_dir / 'history.json',
    )
    if args.output_dir.exists():
        parser.error('--output_dir already exists')
    command = [sys.executable, str(root / 'src/train.py')]
    for key, value in settings.items():
        command.extend([f'--{key}', str(value)])
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()

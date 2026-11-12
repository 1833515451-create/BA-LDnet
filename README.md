# BA-LDNet

Boundary-Decoupled Auxiliary Supervision for Low-Label Diabetic Foot Ulcer Segmentation

## Data

- [DFUC2022](https://dfuc2022.grand-challenge.org/dataset/)
- [FUSeg](https://fusc.grand-challenge.org/)

```text
data/
  DFUC2022/{train_images,train_masks}/
  FUSeg/{train_images,train_masks}/
```

## Environment

Python 3.9

```sh
python -m pip install -r requirements.txt
python src/test_macro.py
```

## Training

```sh
python src/run_training.py --data_path data/DFUC2022 --model mean_teacher --seed 2026 --label 20 --output_dir runs/mt20
python src/run_training.py --data_path data/DFUC2022 --model ba_ldnet --seed 2026 --label 20 --output_dir runs/ba20
```

## Evaluation

```sh
python src/evaluate.py --dataset dfuc2022 --data_path data/DFUC2022 --module boundary_aux --checkpoint runs/ba20/checkpoints/best_dice.pth --output_dir evaluations/dfuc2022_ba20
python src/evaluate.py --dataset fuseg --data_path data/FUSeg --module boundary_aux --checkpoint runs/ba20/checkpoints/best_dice.pth --output_dir evaluations/fuseg_ba20
```

## Results

- [Manuscript tables](results/manuscript_tables/)
- [Paired FUSeg metrics](results/fuseg_primary/FUSeg/)
- [Paired FUSeg figure](figures/FUSeg_paired_analysis.pdf)

```sh
python src/plot_fuseg.py --source results/fuseg_primary --output figures/regenerated
```

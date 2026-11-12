import argparse, itertools, json, os, random, stat, sys
from copy import deepcopy
from pathlib import Path

import albumentations as A
import cv2, matplotlib.pyplot as plt, numpy as np, torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dataset import build_train_transforms, build_valid_transforms, read_pair_manifest, seed_everything, split_dataset
from losses import unpack_outputs
from losses_single_module import SingleModuleSegLoss
from metrics_v9 import evaluate_v9_batch
from models import MiTB3SingleModuleNet
from augmentations import build_domain_transforms, build_preprocess_fn
from utils import ensure_dir, resolve_device, save_checkpoint, save_history

CORE_METRICS = ["accuracy", "precision", "recall", "specificity", "iou", "dice"]
EXTRA_METRICS = ["small_dice", "tiny_dice", "boundary_precision", "boundary_recall", "boundary_f1"]
SURFACE_METRICS = ["hd95", "assd"]
OUTPUT_ROOT = CURRENT_DIR / "outputs_low_label"
MODULE_CHOICES = [
    "baseline", "c2_residual", "c3_residual", "c4_residual", "boundary_aux", "deep_supervision", "small_sampler", "low_contrast_residual",
    "rim_completion_aux", "low_contrast_edge_aux", "tiny_region_recall_aux", "annotation_transition_aux", "feature_calibration", "feature_calibration_boundary_aux"
]


def should_enable_progress_bar():
    try:
        if stat.S_ISREG(os.fstat(sys.stdout.fileno()).st_mode):
            return False
    except (AttributeError, OSError):
        pass
    return sys.stdout.isatty() or sys.stderr.isatty() or os.environ.get("PYCHARM_HOSTED") == "1"


INTERACTIVE_LOG = should_enable_progress_bar()


class MetricMeter:
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, value, n=1):
        if value is None or n <= 0 or not np.isfinite(float(value)):
            return
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


class LabeledSegDataset(Dataset):
    def __init__(self, images, masks, transform, preprocess):
        self.images, self.masks, self.transform, self.preprocess = images, masks, transform, preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(str(self.images[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {self.images[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[index]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {self.masks[index]}")
        sample = self.transform(image=image, mask=np.expand_dims((mask > 127).astype(np.float32), axis=-1))
        image = self.preprocess(sample["image"]).astype(np.float32)
        return torch.from_numpy(np.transpose(image, (2, 0, 1))).float(), torch.from_numpy(np.transpose(sample["mask"].astype(np.float32), (2, 0, 1))).float()


class UnlabeledConsistencyDataset(Dataset):
    def __init__(self, images, weak_transform, strong_transform, preprocess):
        self.images, self.weak_transform, self.strong_transform, self.preprocess = images, weak_transform, strong_transform, preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(str(self.images[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {self.images[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        weak = self.preprocess(self.weak_transform(image=image)["image"]).astype(np.float32)
        strong = self.preprocess(self.strong_transform(image=image)["image"]).astype(np.float32)
        return torch.from_numpy(np.transpose(weak, (2, 0, 1))).float(), torch.from_numpy(np.transpose(strong, (2, 0, 1))).float()


def parse_args():
    parser = argparse.ArgumentParser(description="Low-label/semi-supervised DFUC training for SMP MiT-B3 baseline or C3Residual.")
    parser.add_argument("--module", type=str, default="baseline", choices=MODULE_CHOICES, help="Model variant.")
    parser.add_argument("--semi_supervised", type=str, default="on", choices=["on", "off"], help="Use unlabeled consistency training.")
    parser.add_argument("--aug_mode", type=str, default="domain", choices=["basic", "domain"], help="Supervised augmentation mode.")
    parser.add_argument("--data_path", type=str, default="./external_data_from_desktop/DFUC2022", help="DFUC dataset root path.")
    parser.add_argument("--label_fraction", type=float, default=0.1, help="Fraction of DFUC train masks used as labeled data.")
    parser.add_argument("--label_count", type=int, default=0, help="Exact labeled sample count. Overrides label_fraction when >0.")
    parser.add_argument("--include_labeled_as_unlabeled", type=str, default="off", choices=["on", "off"], help="Also use labeled images in unlabeled consistency pool.")
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Labeled batch size.")
    parser.add_argument("--unlabeled_batch_size", type=int, default=8, help="Unlabeled batch size.")
    parser.add_argument("--img_size", type=int, default=256, help="Image size.")
    parser.add_argument("--encoder_name", type=str, default="mit_b3", help="SMP encoder name.")
    parser.add_argument("--encoder_weights", type=str, default="imagenet", help="Encoder pretrained weights.")
    parser.add_argument("--refine_channels", type=int, default=32, help="C3Residual head width.")
    parser.add_argument("--residual_scale", type=float, default=0.25, help="C3Residual logit scale.")
    parser.add_argument("--encoder_lr", type=float, default=2e-5, help="Encoder learning rate.")
    parser.add_argument("--decoder_lr", type=float, default=1e-4, help="Decoder/module learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--aux_weight", type=float, default=0.2, help="Auxiliary loss weight.")
    parser.add_argument("--boundary_weight", type=float, default=0.08, help="Boundary loss weight.")
    parser.add_argument("--main_boundary_weight", type=float, default=0.0, help="Optional boundary loss weight applied directly to the main segmentation logits.")
    parser.add_argument("--unsup_weight", type=float, default=0.35, help="Max consistency/pseudo-label loss weight.")
    parser.add_argument("--unsup_warmup_epochs", type=int, default=12, help="Ramp-up epochs for unsupervised loss.")
    parser.add_argument("--pseudo_threshold", type=float, default=0.85, help="Teacher confidence threshold.")
    parser.add_argument("--ema_decay", type=float, default=0.995, help="Mean-teacher EMA decay.")
    parser.add_argument("--small_area", type=float, default=0.01, help="Small wound threshold.")
    parser.add_argument("--tiny_area", type=float, default=0.002, help="Tiny wound threshold.")
    parser.add_argument("--boundary_radius", type=int, default=2, help="Boundary metric radius.")
    parser.add_argument("--loss_boundary_radius", type=int, default=1, help="Morphological boundary-band radius used by boundary supervision.")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps. Keep 1 for the canonical 256px protocol.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="DFUC train split ratio.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=2026, help="Legacy combined random seed.")
    parser.add_argument("--split_seed", type=int, default=None, help="Dataset split seed. Defaults to --seed for legacy runs.")
    parser.add_argument("--label_seed", type=int, default=None, help="Labeled-subset seed. Defaults to --seed + 2027 for legacy runs.")
    parser.add_argument("--train_seed", type=int, default=None, help="Initialization and training seed. Defaults to --seed.")
    parser.add_argument("--protocol_dir", type=str, default="", help="Frozen protocol directory containing train/validation and labeled/unlabeled CSV manifests.")
    parser.add_argument("--steps_per_epoch", type=int, default=0, help="Fixed labeled optimizer-step budget per epoch. Zero uses one pass over the labeled loader.")
    parser.add_argument("--amp", type=str, default="off", choices=["auto", "on", "off"], help="Enable mixed precision.")
    parser.add_argument("--save_dir", type=str, default="", help="Checkpoint directory. Auto-filled when empty.")
    parser.add_argument("--plot_dir", type=str, default="", help="Plot directory. Auto-filled when empty.")
    parser.add_argument("--history_path", type=str, default="", help="History json path. Auto-filled when empty.")
    parser.add_argument("--early_stop_patience", type=int, default=24, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    parser.add_argument("--max_train_batches", type=int, default=0, help="Debug only: limit train batches.")
    parser.add_argument("--max_valid_batches", type=int, default=0, help="Debug only: limit valid batches.")
    return parser.parse_args()


def sanitize_name(text):
    return text.replace("/", "-").replace("\\", "-").replace(" ", "_").replace(".", "_").lower()


def label_tag(args):
    return f"label{args.label_count}" if args.label_count > 0 else f"label{int(round(args.label_fraction * 100))}pct"


def normalize_seed_args(args):
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    args.label_seed = args.seed + 2027 if args.label_seed is None else args.label_seed
    args.train_seed = args.seed if args.train_seed is None else args.train_seed
    args.seed = args.train_seed
    return args


def resolve_output_paths(args):
    run_name = f"{sanitize_name(args.module)}_{'semi' if args.semi_supervised == 'on' else 'sup'}_{sanitize_name(args.aug_mode)}"
    root = OUTPUT_ROOT / f"seed{args.seed}" / label_tag(args) / run_name
    args.save_dir = args.save_dir or str(root / "checkpoints")
    args.plot_dir = args.plot_dir or str(root / "plots")
    args.history_path = args.history_path or str(root / "history.json")
    return args


def build_unlabeled_weak_transform(img_size):
    return A.Compose([A.Resize(img_size, img_size)])


def build_unlabeled_strong_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.OneOf([A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=1.0), A.RandomBrightnessContrast(brightness_limit=0.30, contrast_limit=0.30, p=1.0), A.RandomGamma(gamma_limit=(65, 145), p=1.0), A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=28, val_shift_limit=22, p=1.0), A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0)], p=0.95),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7), p=1.0), A.MotionBlur(blur_limit=7, p=1.0), A.GaussNoise(var_limit=(8.0, 45.0), p=1.0), A.ImageCompression(quality_lower=40, quality_upper=95, p=1.0)], p=0.40),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.20),
    ])


def split_labeled_unlabeled(train_images, train_masks, args):
    indices = list(range(len(train_images)))
    random.Random(args.label_seed).shuffle(indices)
    labeled_count = args.label_count if args.label_count > 0 else int(round(len(indices) * args.label_fraction))
    labeled_count = max(1, min(len(indices), labeled_count))
    labeled_ids, unlabeled_ids = set(indices[:labeled_count]), indices[labeled_count:]
    if args.include_labeled_as_unlabeled == "on":
        unlabeled_ids = indices
    labeled_images = [train_images[i] for i in indices if i in labeled_ids]
    labeled_masks = [train_masks[i] for i in indices if i in labeled_ids]
    unlabeled_images = [train_images[i] for i in unlabeled_ids]
    return labeled_images, labeled_masks, unlabeled_images


def protocol_fraction_tag(args):
    if args.label_count > 0:
        return f"count{args.label_count}"
    return f"{int(round(args.label_fraction * 100)):02d}pct"


def load_frozen_protocol(args):
    protocol_dir = Path(args.protocol_dir)
    protocol_metadata_path = protocol_dir / "protocol.json"
    if protocol_metadata_path.exists():
        args.protocol_metadata = json.loads(protocol_metadata_path.read_text(encoding="utf-8"))
    train_images, train_masks = read_pair_manifest(protocol_dir / "train.csv", args.data_path)
    valid_images, valid_masks = read_pair_manifest(protocol_dir / "validation.csv", args.data_path)
    fraction_tag = protocol_fraction_tag(args)
    labeled_images, labeled_masks = read_pair_manifest(protocol_dir / f"labeled_{fraction_tag}.csv", args.data_path)
    unlabeled_images, _ = read_pair_manifest(protocol_dir / f"unlabeled_{fraction_tag}.csv", args.data_path)
    if args.include_labeled_as_unlabeled == "on":
        unlabeled_images = train_images
    return train_images, train_masks, valid_images, valid_masks, labeled_images, labeled_masks, unlabeled_images


def build_dataloaders(args):
    if args.protocol_dir:
        train_images, train_masks, valid_images, valid_masks, labeled_images, labeled_masks, unlabeled_images = load_frozen_protocol(args)
    else:
        train_images, train_masks, valid_images, valid_masks = split_dataset(args.data_path, args.train_ratio, args.split_seed)
        labeled_images, labeled_masks, unlabeled_images = split_labeled_unlabeled(train_images, train_masks, args)
    preprocess = build_preprocess_fn(args.encoder_name, args.encoder_weights)
    supervised_transform = build_domain_transforms(args.img_size) if args.aug_mode == "domain" else build_train_transforms(args.img_size)
    labeled_dataset = LabeledSegDataset(labeled_images, labeled_masks, supervised_transform, preprocess)
    valid_dataset = LabeledSegDataset(valid_images, valid_masks, build_valid_transforms(args.img_size), preprocess)
    unlabeled_dataset = UnlabeledConsistencyDataset(unlabeled_images, build_unlabeled_weak_transform(args.img_size), build_unlabeled_strong_transform(args.img_size), preprocess) if args.semi_supervised == "on" and unlabeled_images else None
    labeled_generator = torch.Generator().manual_seed(args.train_seed)
    unlabeled_generator = torch.Generator().manual_seed(args.train_seed + 1)
    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), generator=labeled_generator)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.unlabeled_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), generator=unlabeled_generator) if unlabeled_dataset is not None else None
    return labeled_loader, unlabeled_loader, valid_loader, labeled_dataset, unlabeled_dataset, valid_dataset


def use_amp_training(device, amp_mode):
    return device.type == "cuda" if amp_mode == "auto" else device.type == "cuda" and amp_mode == "on"


def build_optimizer(model, args):
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(param) for param in encoder_params}
    other_params = [param for param in model.parameters() if id(param) not in encoder_ids]
    return optim.AdamW([{"params": encoder_params, "lr": args.encoder_lr}, {"params": other_params, "lr": args.decoder_lr}], weight_decay=args.weight_decay)


def set_requires_grad(model, enabled):
    for param in model.parameters():
        param.requires_grad = enabled


def update_ema_model(teacher, student, decay):
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            teacher_buffer.copy_(student_buffer)


def masked_pseudo_loss(student_logits, teacher_probs, threshold):
    pseudo = (teacher_probs >= 0.5).float()
    confidence = torch.maximum(teacher_probs, 1.0 - teacher_probs)
    conf_mask = (confidence >= threshold).float()
    valid = conf_mask.sum().clamp_min(1.0)
    bce = (F.binary_cross_entropy_with_logits(student_logits, pseudo, reduction="none") * conf_mask).sum() / valid
    student_probs = torch.sigmoid(student_logits) * conf_mask
    pseudo = pseudo * conf_mask
    intersection = (student_probs.flatten(1) * pseudo.flatten(1)).sum(dim=1)
    denominator = student_probs.flatten(1).sum(dim=1) + pseudo.flatten(1).sum(dim=1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice, conf_mask.mean().detach()


def unsup_weight_for_epoch(args, epoch):
    if args.semi_supervised != "on" or args.unsup_weight <= 0:
        return 0.0
    if args.unsup_warmup_epochs <= 0:
        return args.unsup_weight
    return args.unsup_weight * min(1.0, float(epoch) / float(args.unsup_warmup_epochs))


def format_metrics(metrics, names):
    return " | ".join(f"{name}: {metrics.get(name, 0.0):.4f}" for name in names)


def log_line(message=""):
    print(message, flush=True)


def update_meters(meters, metrics, counts):
    for name, value in metrics.items():
        if name not in meters:
            meters[name] = MetricMeter()
        meters[name].update(value, counts.get(name, 1))


def repeat_loader(loader):
    while True:
        yield from loader


def run_train_epoch(model, teacher, labeled_loader, unlabeled_loader, criterion, optimizer, device, args, epoch, scaler, amp_enabled):
    model.train()
    if teacher is not None:
        teacher.eval()
    loss_meter, sup_meter, unsup_meter, conf_meter = MetricMeter(), MetricMeter(), MetricMeter(), MetricMeter()
    metric_meters = {name: MetricMeter() for name in CORE_METRICS + EXTRA_METRICS}
    epoch_batches = args.steps_per_epoch if args.steps_per_epoch > 0 else len(labeled_loader)
    max_batches = args.max_train_batches
    total_batches = epoch_batches if max_batches <= 0 else min(epoch_batches, max_batches)
    labeled_source = repeat_loader(labeled_loader) if args.steps_per_epoch > 0 else iter(labeled_loader)
    unlabeled_iter = repeat_loader(unlabeled_loader) if unlabeled_loader is not None else None
    iterator = tqdm(itertools.islice(labeled_source, total_batches), total=total_batches, desc="Train", dynamic_ncols=True, leave=INTERACTIVE_LOG, disable=not INTERACTIVE_LOG)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, masks) in enumerate(iterator):
        if max_batches > 0 and batch_index >= max_batches:
            break
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            sup_loss = criterion(outputs, masks)
            unsup_loss, confidence_ratio = torch.zeros((), device=device), torch.zeros((), device=device)
            current_unsup_weight = unsup_weight_for_epoch(args, epoch)
            if unlabeled_iter is not None and current_unsup_weight > 0:
                weak_images, strong_images = next(unlabeled_iter)
                weak_images, strong_images = weak_images.to(device, non_blocking=True), strong_images.to(device, non_blocking=True)
                with torch.no_grad():
                    teacher_logits, _ = unpack_outputs(teacher(weak_images))
                    teacher_probs = torch.sigmoid(teacher_logits.float())
                student_logits, _ = unpack_outputs(model(strong_images))
                unsup_loss, confidence_ratio = masked_pseudo_loss(student_logits.float(), teacher_probs, args.pseudo_threshold)
            loss = sup_loss + current_unsup_weight * unsup_loss
        scaler.scale(loss / args.grad_accum_steps).backward()
        should_step = (batch_index + 1) % args.grad_accum_steps == 0 or batch_index + 1 == total_batches
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if teacher is not None:
                update_ema_model(teacher, model, args.ema_decay)
        logits, _ = unpack_outputs(outputs)
        loss_meter.update(loss.item(), images.size(0))
        sup_meter.update(sup_loss.item(), images.size(0))
        unsup_meter.update(unsup_loss.item(), images.size(0))
        conf_meter.update(confidence_ratio.item(), images.size(0))
        metrics, counts = evaluate_v9_batch(torch.sigmoid(logits).detach(), masks, small_area=args.small_area, tiny_area=args.tiny_area, boundary_radius=args.boundary_radius, surface_metrics=False)
        update_meters(metric_meters, metrics, counts)
        if INTERACTIVE_LOG:
            iterator.set_postfix(loss=f"{loss_meter.avg:.4f}", sup=f"{sup_meter.avg:.4f}", unsup=f"{unsup_meter.avg:.4f}", dice=f"{metric_meters['dice'].avg:.4f}", conf=f"{conf_meter.avg:.3f}")
    metrics = {name: meter.avg for name, meter in metric_meters.items()}
    metrics.update({"sup_loss": sup_meter.avg, "unsup_loss": unsup_meter.avg, "pseudo_confidence_ratio": conf_meter.avg, "unsup_weight": unsup_weight_for_epoch(args, epoch)})
    return loss_meter.avg, metrics


def run_valid_epoch(model, dataloader, criterion, device, args, amp_enabled):
    model.eval()
    loss_meter = MetricMeter()
    metric_meters = {name: MetricMeter() for name in CORE_METRICS + EXTRA_METRICS + SURFACE_METRICS}
    max_batches = args.max_valid_batches
    total_batches = len(dataloader) if max_batches <= 0 else min(len(dataloader), max_batches)
    iterator = tqdm(dataloader, total=total_batches, desc="Valid", dynamic_ncols=True, leave=INTERACTIVE_LOG, disable=not INTERACTIVE_LOG)
    with torch.no_grad():
        for batch_index, (images, masks) in enumerate(iterator):
            if max_batches > 0 and batch_index >= max_batches:
                break
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, masks)
            logits, _ = unpack_outputs(outputs)
            loss_meter.update(loss.item(), images.size(0))
            metrics, counts = evaluate_v9_batch(torch.sigmoid(logits).detach(), masks, small_area=args.small_area, tiny_area=args.tiny_area, boundary_radius=args.boundary_radius, surface_metrics=True)
            update_meters(metric_meters, metrics, counts)
            if INTERACTIVE_LOG:
                iterator.set_postfix(loss=f"{loss_meter.avg:.4f}", dice=f"{metric_meters['dice'].avg:.4f}", small=f"{metric_meters['small_dice'].avg:.4f}", bf1=f"{metric_meters['boundary_f1'].avg:.4f}")
    return loss_meter.avg, {name: meter.avg for name, meter in metric_meters.items()}


def save_plots(history, save_dir):
    ensure_dir(save_dir)
    epochs = [item["epoch"] for item in history]
    plot_items = [("loss", [item["train_loss"] for item in history], [item["valid_loss"] for item in history])]
    metric_names = sorted(set().union(*[set(item["train_metrics"].keys()) | set(item["valid_metrics"].keys()) for item in history]))
    for name in metric_names:
        plot_items.append((name, [item["train_metrics"].get(name, np.nan) for item in history], [item["valid_metrics"].get(name, np.nan) for item in history]))
    for name, train_values, valid_values in plot_items:
        plt.figure(figsize=(9, 5.5))
        if np.isfinite(np.asarray(train_values, dtype=np.float32)).any():
            plt.plot(epochs, train_values, label=f"Train {name}")
        if np.isfinite(np.asarray(valid_values, dtype=np.float32)).any():
            plt.plot(epochs, valid_values, label=f"Valid {name}")
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.title(f"{name} Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}_curve.png"))
        plt.close()


def save_epoch_state(args, model, teacher, optimizer, epoch, valid_loss, valid_metrics, history, best_trackers):
    state = {"epoch": epoch, "model_state_dict": model.state_dict(), "teacher_state_dict": teacher.state_dict() if teacher is not None else None, "optimizer_state_dict": optimizer.state_dict(), "valid_loss": valid_loss, "valid_metrics": valid_metrics, "args": vars(args)}
    tracker = best_trackers["dice"]
    improved = valid_metrics["dice"] > tracker["value"]
    if improved:
        tracker["value"], tracker["epoch"], tracker["valid_loss"], tracker["valid_metrics"] = valid_metrics["dice"], epoch, valid_loss, dict(valid_metrics)
        save_checkpoint(state, os.path.join(args.save_dir, "best_dice.pth"))
    save_history({"history": history, "best": {"dice": tracker["value"], "epoch": tracker["epoch"], "valid_loss": tracker["valid_loss"], "valid_metrics": tracker["valid_metrics"]}, "args": vars(args)}, args.history_path)
    save_plots(history, args.plot_dir)
    return improved


def print_header(args, device, model, labeled_dataset, unlabeled_dataset, valid_dataset):
    total_params = sum(p.numel() for p in model.parameters())
    log_line("\n===== Low-Label Training Configuration =====")
    log_line(f"Module/Semi/Aug: {args.module}/{args.semi_supervised}/{args.aug_mode}")
    log_line(f"Encoder: {args.encoder_name} | Weights: {args.encoder_weights}")
    log_line(f"Device: {device}")
    log_line(f"Labeled/Unlabeled/Valid Samples: {len(labeled_dataset)}/{0 if unlabeled_dataset is None else len(unlabeled_dataset)}/{len(valid_dataset)}")
    log_line(f"Label Setting: fraction={args.label_fraction}, count={args.label_count if args.label_count > 0 else len(labeled_dataset)}")
    log_line(f"Seeds: split={args.split_seed}, label={args.label_seed}, train={args.train_seed}")
    log_line(f"Protocol Dir: {args.protocol_dir or 'legacy generated split'}")
    log_line(f"Image/Batch/Epochs: {args.img_size}/{args.batch_size}+{args.unlabeled_batch_size}/{args.epochs}")
    log_line(f"Optimizer steps per epoch: {args.steps_per_epoch if args.steps_per_epoch > 0 else len(labeled_dataset) // args.batch_size + int(len(labeled_dataset) % args.batch_size > 0)}")
    log_line(f"Gradient accumulation: {args.grad_accum_steps} | Effective labeled batch: {args.batch_size * args.grad_accum_steps}")
    log_line(f"LRs: encoder={args.encoder_lr}, decoder_module={args.decoder_lr}")
    log_line(f"Unsupervised: weight={args.unsup_weight}, warmup={args.unsup_warmup_epochs}, threshold={args.pseudo_threshold}, ema={args.ema_decay}")
    log_line(f"Boundary Loss: auxiliary={args.boundary_weight}, main_logits={getattr(args, 'main_boundary_weight', 0.0)}, radius={args.loss_boundary_radius}")
    log_line(f"AMP: {use_amp_training(device, args.amp)} | Parameters: {total_params:,}")
    log_line(f"Checkpoint Dir: {args.save_dir}")
    log_line(f"Plot Dir: {args.plot_dir}")
    log_line(f"History Path: {args.history_path}")
    log_line("============================================\n")


def main(args):
    args = normalize_seed_args(args)
    args.metric_aggregation = "image_macro"
    args = resolve_output_paths(args)
    seed_everything(args.train_seed)
    device = resolve_device(args.device)
    amp_enabled = use_amp_training(device, args.amp)
    labeled_loader, unlabeled_loader, valid_loader, labeled_dataset, unlabeled_dataset, valid_dataset = build_dataloaders(args)
    model = MiTB3SingleModuleNet(args.module, args.encoder_name, args.encoder_weights, args.refine_channels, args.residual_scale).to(device)
    teacher = deepcopy(model).to(device) if args.semi_supervised == "on" and unlabeled_loader is not None else None
    if teacher is not None:
        set_requires_grad(teacher, False)
    criterion = SingleModuleSegLoss(args.aux_weight, args.boundary_weight, args.tiny_area, getattr(args, "main_boundary_weight", 0.0), args.loss_boundary_radius)
    optimizer = build_optimizer(model, args)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ensure_dir(args.save_dir)
    ensure_dir(args.plot_dir)
    ensure_dir(str(Path(args.history_path).parent))
    print_header(args, device, model, labeled_dataset, unlabeled_dataset, valid_dataset)
    history, early_stop_counter = [], 0
    best_trackers = {"dice": {"value": -1.0, "epoch": 0, "valid_loss": None, "valid_metrics": {}}}
    stop_reason = "max_epochs"
    for epoch in range(1, args.epochs + 1):
        log_line(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_metrics = run_train_epoch(model, teacher, labeled_loader, unlabeled_loader, criterion, optimizer, device, args, epoch, scaler, amp_enabled)
        valid_loss, valid_metrics = run_valid_epoch(model, valid_loader, criterion, device, args, amp_enabled)
        scheduler.step(valid_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "train_metrics": train_metrics, "valid_metrics": valid_metrics})
        log_line(f"Train Loss: {train_loss:.4f} | {format_metrics(train_metrics, CORE_METRICS + EXTRA_METRICS + ['sup_loss', 'unsup_loss', 'pseudo_confidence_ratio', 'unsup_weight'])}")
        log_line(f"Valid Loss: {valid_loss:.4f} | {format_metrics(valid_metrics, CORE_METRICS + EXTRA_METRICS + SURFACE_METRICS)}")
        improved = save_epoch_state(args, model, teacher, optimizer, epoch, valid_loss, valid_metrics, history, best_trackers)
        early_stop_counter = 0 if improved else early_stop_counter + 1
        if early_stop_counter >= args.early_stop_patience:
            log_line(f"Early stopping at epoch {epoch}.")
            stop_reason = "early_stopping"
            break
    tracker = best_trackers["dice"]
    completion_path = Path(args.history_path).with_name("training_complete.json")
    completion_path.write_text(
        json.dumps(
            {
                "completed": True,
                "stop_reason": stop_reason,
                "last_epoch": history[-1]["epoch"] if history else 0,
                "best_epoch": tracker["epoch"],
                "best_macro_dice": tracker["value"],
                "checkpoint": str(Path(args.save_dir) / "best_dice.pth"),
                "protocol_dir": args.protocol_dir,
                "split_seed": args.split_seed,
                "label_seed": args.label_seed,
                "train_seed": args.train_seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log_line(f"Training finished. best dice: {tracker['value']:.4f}, epoch: {tracker['epoch']}")


if __name__ == "__main__":
    main(parse_args())

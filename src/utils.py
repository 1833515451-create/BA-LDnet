import json
import os

import matplotlib.pyplot as plt
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_device(requested="auto"):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")

    if torch.cuda.is_available():
        try:
            arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
            if arch_list:
                supported = []
                for arch in arch_list:
                    if arch.startswith("sm_"):
                        try:
                            supported.append(int(arch.split("_")[1]))
                        except ValueError:
                            pass
                if supported:
                    major, minor = torch.cuda.get_device_capability(0)
                    current_arch = major * 10 + minor
                    if current_arch > max(supported):
                        return torch.device("cpu")
            torch.zeros(1).to("cuda")
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def save_checkpoint(state, save_path):
    ensure_dir(os.path.dirname(save_path))
    torch.save(state, save_path)


def save_history(history, save_path):
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_training_plots(history, save_dir):
    ensure_dir(save_dir)
    epochs = [item["epoch"] for item in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [item["train_loss"] for item in history], label="Train Loss")
    plt.plot(epochs, [item["valid_loss"] for item in history], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    metric_names = [key for key in history[0]["train_metrics"].keys()]
    for metric_name in metric_names:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, [item["train_metrics"][metric_name] for item in history], label=f"Train {metric_name}")
        plt.plot(epochs, [item["valid_metrics"][metric_name] for item in history], label=f"Valid {metric_name}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric_name}_curve.png"))
        plt.close()

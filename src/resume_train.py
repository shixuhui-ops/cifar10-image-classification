import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import create_dataloaders
from models import build_model, count_parameters
from train import run_epoch, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume CIFAR-10 training from a checkpoint."
    )

    parser.add_argument(
        "--resume",
        required=True,
        help="last_model.pt 文件路径，或包含该文件的目录。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认使用 checkpoint 所在目录。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="训练目标总轮数，不是新增轮数。",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="早停耐心值；默认读取 checkpoint 配置。",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="仅用于冒烟测试。",
    )

    return parser.parse_args()


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def get_best_from_history(history):
    if not history:
        return -1.0, 0, 0

    best_index = max(
        range(len(history)),
        key=lambda index: history[index]["val_accuracy"],
    )
    best_accuracy = history[best_index]["val_accuracy"]
    best_epoch = history[best_index]["epoch"]

    stale_epochs = len(history) - best_index - 1

    return best_accuracy, best_epoch, stale_epochs


def main():
    args = parse_args()

    resume_path = Path(args.resume)
    if resume_path.is_dir():
        resume_path = resume_path / "last_model.pt"

    if not resume_path.exists():
        raise FileNotFoundError(
            f"找不到 checkpoint: {resume_path}"
        )

    checkpoint = torch.load(
        resume_path,
        map_location="cpu",
        weights_only=False,
    )

    saved_config = checkpoint["config"]

    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else resume_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(saved_config.get("seed", 42))
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")
    print(f"resume checkpoint: {resume_path}")

    augmentation_policy = saved_config.get(
        "augmentation_policy"
    )

    if augmentation_policy is None:
        augmentation_policy = (
            "none"
            if saved_config.get("no_augmentation", False)
            else "all"
        )

    loaders, class_names = create_dataloaders(
        data_dir=saved_config.get("data_dir", "data"),
        batch_size=int(saved_config.get("batch_size", 128)),
        seed=seed,
        num_workers=int(
            saved_config.get("num_workers", 0)
        ),
        use_augmentation=augmentation_policy != "none",
        augmentation_policy=augmentation_policy,
    )

    model_name = checkpoint["model_name"]

    model = build_model(
        name=model_name,
        num_classes=len(class_names),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])

    label_smoothing = float(
        saved_config.get("label_smoothing", 0.1)
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
    )

    learning_rate = float(
        saved_config.get("lr", 3e-4)
    )
    weight_decay = float(
        saved_config.get("weight_decay", 5e-4)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )
    move_optimizer_state_to_device(optimizer, device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(saved_config.get("epochs", args.epochs)),
    )
    scheduler.load_state_dict(
        checkpoint["scheduler_state"]
    )

    history = list(checkpoint.get("history", []))

    best_val_accuracy, best_epoch, stale_epochs = (
        get_best_from_history(history)
    )

    start_epoch = int(checkpoint["epoch"]) + 1

    patience = args.patience
    if patience is None:
        patience = int(
            saved_config.get("patience", 8)
        )

    config = dict(saved_config)
    config["resume_checkpoint"] = str(resume_path)
    config["resumed"] = True
    config["target_epochs"] = args.epochs
    config["augmentation_policy"] = augmentation_policy

    save_json(output_dir / "config.json", config)

    print(
        f"starting epoch: {start_epoch}; "
        f"target epochs: {args.epochs}; "
        f"best epoch: {best_epoch}; "
        f"best validation accuracy: {best_val_accuracy:.4f}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            limit_batches=args.limit_batches,
        )

        val_loss, val_accuracy = run_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            optimizer=None,
            limit_batches=args.limit_batches,
        )

        scheduler.step()

        improved = val_accuracy > best_val_accuracy

        if improved:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "elapsed_seconds": time.time() - start_time,
            }
        )

        save_json(output_dir / "history.json", history)

        current_checkpoint = {
            "epoch": epoch,
            "model_name": model_name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "epochs_without_improvement": stale_epochs,
            "history": history,
            "config": config,
        }

        torch.save(
            current_checkpoint,
            output_dir / "last_model.pt",
        )

        if improved:
            torch.save(
                current_checkpoint,
                output_dir / "best_model.pt",
            )

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.4f} | "
            f"best epoch {best_epoch}"
        )

        if stale_epochs >= patience:
            print(
                f"early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}"
            )
            break

    best_path = output_dir / "best_model.pt"

    best_checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        best_checkpoint["model_state"]
    )

    test_loss, test_accuracy = run_epoch(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        optimizer=None,
        limit_batches=args.limit_batches,
    )

    test_metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "augmentation_policy": augmentation_policy,
        "resumed": True,
        "resume_checkpoint": str(resume_path),
    }

    save_json(
        output_dir / "test_metrics.json",
        test_metrics,
    )

    print(
        f"test loss: {test_loss:.4f} | "
        f"test accuracy: {test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
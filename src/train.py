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


def set_seed(seed: int):
    """固定随机种子，保证实验尽可能可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    limit_batches=0,
):
    """执行一个训练或验证轮次。"""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(loader):
        if limit_batches > 0 and batch_index >= limit_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("当前 DataLoader 没有产生任何样本。")

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-10 image classification model."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet_se",
        choices=["baseline", "baseline_wide", "resnet", "resnet_se"],
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/resnet_se",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
    )
    parser.add_argument(
        "--augmentation-policy",
        type=str,
        choices=["none", "crop", "flip", "color", "all"],
        default=None,
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    augmentation_policy = args.augmentation_policy

    if augmentation_policy is None:
        augmentation_policy = (
            "none" if args.no_augmentation else "all"
        )

    loaders, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        use_augmentation=augmentation_policy != "none",
        augmentation_policy=augmentation_policy,
    )

    model = build_model(
        name=args.model,
        num_classes=len(class_names),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    config = vars(args).copy()
    config["device"] = str(device)
    config["num_parameters"] = count_parameters(model)
    config["class_names"] = class_names
    config["augmentation_policy"] = augmentation_policy

    save_json(output_dir / "config.json", config)

    history = []
    best_val_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
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

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        epoch_result = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "elapsed_seconds": elapsed,
        }

        history.append(epoch_result)
        save_json(output_dir / "history.json", history)

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"lr {current_lr:.6f} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.4f} | "
            f"time {elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_name": args.model,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "history": history,
            "config": config,
        }

        torch.save(
            checkpoint,
            output_dir / "last_model.pt",
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint["best_val_accuracy"] = best_val_accuracy

            torch.save(
                checkpoint,
                output_dir / "best_model.pt",
            )
            print("  saved new best model")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}"
            )
            break

    best_checkpoint_path = output_dir / "best_model.pt"

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(best_checkpoint["model_state"])

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
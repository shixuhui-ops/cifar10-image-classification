import argparse
import time
from pathlib import Path

import torch
from torch import nn

from data import create_dataloaders
from models import TransferResNet18, count_parameters
from train import run_epoch, save_json, set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ImageNet-pretrained ResNet-18 on CIFAR-10."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default="outputs/resnet18_transfer",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--limit-batches", type=int, default=0)
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs 和 patience 必须为正整数。")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    loaders, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        use_augmentation=True,
    )

    # 从已下载的 ImageNet 权重初始化，再微调。
    model = TransferResNet18(
        num_classes=len(class_names),
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.layer4.parameters(),
                "lr": args.backbone_lr,
            },
            {
                "params": model.backbone.fc.parameters(),
                "lr": args.head_lr,
            },
        ],
        weight_decay=5e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    config = vars(args).copy()
    config.update({
        "model": "resnet18_transfer",
        "device": str(device),
        "num_parameters": count_parameters(model),
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "class_names": class_names,
        "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "input_size": 96,
        "trainable_layers": ["layer4", "fc"],
        "label_smoothing": 0.1,
        "weight_decay": 5e-4,
        "val_ratio": 0.1,
        "use_augmentation": True,
        "is_smoke_test": args.limit_batches > 0,
    })
    save_json(output_dir / "config.json", config)

    print(f"Total parameters: {config['total_parameters']:,}")
    print(f"Trainable parameters: {config['num_parameters']:,}")

    history = []
    best_val_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        backbone_lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]

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
            limit_batches=args.limit_batches,
        )

        elapsed = time.perf_counter() - start

        history.append({
            "epoch": epoch,
            "learning_rate": head_lr,
            "backbone_learning_rate": backbone_lr,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "elapsed_seconds": elapsed,
        })
        save_json(output_dir / "history.json", history)

        improved = val_accuracy > best_val_accuracy
        if improved:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model_name": "resnet18_transfer",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "history": history,
            "config": config,
        }

        torch.save(checkpoint, output_dir / "last_model.pt")
        if improved:
            torch.save(checkpoint, output_dir / "best_model.pt")

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"backbone lr {backbone_lr:.6f} | "
            f"head lr {head_lr:.6f} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.4f} | "
            f"time {elapsed:.1f}s"
        )

        if improved:
            print("  saved new best model")

        if stale_epochs >= args.patience:
            print(f"Early stopping; best epoch: {best_epoch}")
            break

    checkpoint = torch.load(
        output_dir / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_accuracy = run_epoch(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        limit_batches=args.limit_batches,
    )

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "is_smoke_test": args.limit_batches > 0,
    }
    save_json(output_dir / "test_metrics.json", metrics)

    print(
        f"test loss: {test_loss:.4f} | "
        f"test accuracy: {test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
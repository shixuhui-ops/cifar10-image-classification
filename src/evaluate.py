import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    create_dataloaders,
)
from models import build_model


def save_json(path, content):
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, indent=2, ensure_ascii=False)


def plot_history(run_dir, figure_dir):
    with (run_dir / "history.json").open(
        encoding="utf-8"
    ) as file:
        history = json.load(file)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for split, label in (("train", "Train"), ("val", "Validation")):
        axes[0].plot(
            epochs,
            [row[f"{split}_loss"] for row in history],
            label=label,
        )
        axes[1].plot(
            epochs,
            [100 * row[f"{split}_accuracy"] for row in history],
            label=label,
        )

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()

    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy (%)")
    fig.suptitle(run_dir.name)
    fig.tight_layout()
    fig.savefig(figure_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def plot_confusion(matrix, class_names, figure_dir):
    # 按真实类别逐行归一化，每行之和为 1。
    normalized = matrix / np.maximum(
        matrix.sum(axis=1, keepdims=True), 1
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for axis, values, title, is_normalized in (
        (axes[0], matrix, "Counts", False),
        (axes[1], normalized, "Row-normalized", True),
    ):
        image = axis.imshow(
            values,
            cmap="Blues",
            vmin=0,
            vmax=1 if is_normalized else None,
        )
        axis.set_xticks(range(len(class_names)))
        axis.set_yticks(range(len(class_names)))
        axis.set_xticklabels(
            class_names, rotation=45, ha="right"
        )
        axis.set_yticklabels(class_names)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(title)

        threshold = values.max() / 2
        for row in range(len(class_names)):
            for col in range(len(class_names)):
                value = values[row, col]
                text = (
                    f"{value:.2f}"
                    if is_normalized
                    else str(int(value))
                )
                axis.text(
                    col,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > threshold else "black",
                )

        fig.colorbar(image, ax=axis, fraction=0.046)

    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def plot_errors(dataset, targets, predictions, confidence,
                class_names, figure_dir):
    wrong_indices = np.flatnonzero(targets != predictions)

    if len(wrong_indices) == 0:
        return

    # 展示模型最有信心但预测错误的 16 张图像。
    order = np.argsort(-confidence[wrong_indices])
    selected = wrong_indices[order[:16]]

    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    fig, axes = plt.subplots(4, 4, figsize=(11, 11))

    for axis in axes.flat:
        axis.axis("off")

    for axis, index in zip(axes.flat, selected):
        image, _ = dataset[int(index)]
        image = (image * std + mean).clamp(0, 1)
        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_title(
            f"True: {class_names[targets[index]]}\n"
            f"Pred: {class_names[predictions[index]]} "
            f"({confidence[index]:.1%})",
            fontsize=9,
        )

    fig.suptitle("Highest-confidence misclassifications")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figure_dir / "error_samples.png", dpi=200)
    plt.close(fig)


def evaluate_run(run_dir, loader, class_names, device):
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = build_model(
        checkpoint["model_name"],
        num_classes=len(class_names),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    targets_list = []
    predictions_list = []
    confidence_list = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=run_dir.name):
            logits = model(images.to(device))
            probabilities = logits.softmax(dim=1)
            confidence, predictions = probabilities.max(dim=1)

            targets_list.append(labels.numpy())
            predictions_list.append(predictions.cpu().numpy())
            confidence_list.append(confidence.cpu().numpy())

    targets = np.concatenate(targets_list)
    predictions = np.concatenate(predictions_list)
    confidence = np.concatenate(confidence_list)

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(class_names))),
    )

    save_json(run_dir / "classification_report.json", report)
    np.savez_compressed(
        run_dir / "predictions.npz",
        targets=targets,
        predictions=predictions,
        confidence=confidence,
    )
    np.savetxt(
        run_dir / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
        header=",".join(class_names),
        comments="",
    )

    with (run_dir / "per_class_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "class",
            "precision",
            "recall_per_class_accuracy",
            "f1_score",
            "support",
        ])
        for name in class_names:
            metrics = report[name]
            writer.writerow([
                name,
                metrics["precision"],
                metrics["recall"],
                metrics["f1-score"],
                metrics["support"],
            ])

    plot_history(run_dir, figure_dir)
    plot_confusion(matrix, class_names, figure_dir)
    plot_errors(
        loader.dataset,
        targets,
        predictions,
        confidence,
        class_names,
        figure_dir,
    )

    result = {
        "experiment": run_dir.name,
        "model": checkpoint["model_name"],
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameters": checkpoint["config"]["num_parameters"],
        "best_epoch": checkpoint["epoch"],
        "best_val_accuracy": checkpoint["best_val_accuracy"],
        "test_accuracy": accuracy_score(targets, predictions),
        "macro_f1": report["macro avg"]["f1-score"],
        "test_samples": len(targets),
    }

    save_json(run_dir / "evaluation_summary.json", result)

    print(
        f"\n{result['model']}: "
        f"test accuracy={result['test_accuracy']:.2%}, "
        f"macro F1={result['macro_f1']:.4f}"
    )
    print(f"Figures saved to: {figure_dir}\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["outputs/baseline", "outputs/resnet_se"],
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--comparison-path",
        default="outputs/model_comparison.csv",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    loaders, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_augmentation=False,
    )

    results = []
    for run in args.runs:
        results.append(
            evaluate_run(
                Path(run),
                loaders["test"],
                class_names,
                device,
            )
        )

    # 同时保存模型对比表，供报告使用。
    comparison_path = Path(args.comparison_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()
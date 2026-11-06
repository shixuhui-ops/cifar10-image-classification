import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import CIFAR10

from data import build_transforms
from gradcam import compute_cam
from models import build_model


OUTPUT_DIR = Path("reports/figures")


def read_json(path):
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def save_figure(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_training(run_dir):
    history = read_json(run_dir / "history.json")
    metrics = read_json(run_dir / "evaluation_summary.json")
    epochs = [item["epoch"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1))

    for split, label, color in (
        ("train", "Train", "#2563eb"),
        ("val", "Validation", "#ea580c"),
    ):
        axes[0].plot(
            epochs,
            [item[f"{split}_loss"] for item in history],
            label=label,
            color=color,
        )
        axes[1].plot(
            epochs,
            [100 * item[f"{split}_accuracy"] for item in history],
            label=label,
            color=color,
        )

    for axis in axes:
        axis.axvline(
            metrics["best_epoch"],
            color="#64748b",
            linestyle=":",
            label="Selected epoch",
        )
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    axes[0].set_ylabel("Smoothed cross-entropy")
    axes[1].set_ylabel("Accuracy (%)")
    fig.tight_layout()
    save_figure(fig, "main_training_curves.png")


def plot_confusion(run_dir, class_names):
    matrix = np.loadtxt(
        run_dir / "confusion_matrix.csv",
        delimiter=",",
        skiprows=1,
        dtype=int,
    )

    if matrix.shape != (10, 10):
        raise ValueError("混淆矩阵必须为 10×10。")

    normalized = (
        100 * matrix / matrix.sum(axis=1, keepdims=True)
    )

    fig, axis = plt.subplots(figsize=(6.5, 5.0))
    image = axis.imshow(
        normalized,
        cmap="Blues",
        vmin=0,
        vmax=100,
    )

    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    axis.set_xticklabels(
        class_names, rotation=45, ha="right", fontsize=8
    )
    axis.set_yticklabels(class_names, fontsize=8)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Row-normalized confusion matrix (%)", fontsize=10)

    for row in range(10):
        for column in range(10):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value >= 50 else "#172033",
            )

    colorbar = fig.colorbar(image, ax=axis, fraction=0.046)
    colorbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    save_figure(fig, "main_confusion_matrix.png")


def plot_errors(dataset, predictions, confidence, targets):
    wrong = np.flatnonzero(predictions != targets)
    selected = wrong[np.argsort(-confidence[wrong])[:4]]

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))

    for axis in axes:
        axis.axis("off")

    for axis, index in zip(axes, selected):
        index = int(index)
        axis.imshow(dataset.data[index], interpolation="nearest")
        axis.set_title(
            f"Test #{index}\n"
            f"True: {dataset.classes[targets[index]]}\n"
            f"Pred: {dataset.classes[predictions[index]]}\n"
            f"Confidence: {confidence[index]:.1%}",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, "error_examples.png")
    return [int(index) for index in selected]


def plot_gradcam(run_dir, dataset, saved_predictions):
    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint["model_name"] not in {"resnet", "resnet_se"}:
        raise ValueError("当前紧凑 Grad-CAM 绘图支持项目中的残差模型。")

    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 一个正确样本和一个典型错误样本。
    # 完整八样本版本仍保留在原实验目录。
    selected = [3, 2161]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2))

    for row, index in enumerate(selected):
        image, target = dataset[index]

        with torch.enable_grad():
            inputs = image.unsqueeze(0)
            features = model.stem(inputs)
            features = model.layer1(features)
            features = model.layer2(features)
            features = model.layer3(features)
            logits = model.classifier(features)

            probabilities = logits.softmax(dim=1)
            prediction = int(probabilities.argmax(dim=1).item())
            confidence = float(probabilities[0, prediction].item())

            if prediction != int(saved_predictions[index]):
                raise RuntimeError(
                    f"样本 {index} 的当前预测与保存结果不一致。"
                )

            predicted_cam = compute_cam(
                features,
                logits,
                prediction,
                retain_graph=prediction != target,
            )
            true_cam = (
                predicted_cam
                if prediction == target
                else compute_cam(features, logits, target)
            )

        original = dataset.data[index]

        axes[row, 0].imshow(original, interpolation="nearest")
        axes[row, 0].set_title(
            f"Test #{index} | True: {dataset.classes[target]}",
            fontsize=9,
        )

        for column, cam, title in (
            (
                1,
                predicted_cam,
                f"Predicted: {dataset.classes[prediction]} "
                f"({confidence:.1%})",
            ),
            (
                2,
                true_cam,
                f"Target: {dataset.classes[target]}",
            ),
        ):
            axes[row, column].imshow(
                original, interpolation="nearest"
            )
            axes[row, column].imshow(
                cam,
                cmap="jet",
                alpha=0.45,
                vmin=0,
                vmax=1,
                interpolation="bilinear",
            )
            axes[row, column].set_title(title, fontsize=9)

        for axis in axes[row]:
            axis.axis("off")

    fig.tight_layout()
    save_figure(fig, "gradcam_examples.png")
    return selected


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    summary = read_json(Path("reports/results_summary.json"))
    experiment = summary["selected_experiment"]
    run_dir = Path("outputs") / experiment
    config = read_json(run_dir / "config.json")

    _, eval_transform = build_transforms(use_augmentation=False)
    dataset = CIFAR10(
        root=config["data_dir"],
        train=False,
        download=False,
        transform=eval_transform,
    )

    with np.load(run_dir / "predictions.npz") as saved:
        targets = saved["targets"]
        predictions = saved["predictions"]
        confidence = saved["confidence"]

    if not np.array_equal(targets, np.asarray(dataset.targets)):
        raise RuntimeError("预测文件与测试集标签顺序不一致。")

    plot_training(run_dir)
    plot_confusion(run_dir, dataset.classes)
    error_indices = plot_errors(
        dataset, predictions, confidence, targets
    )
    cam_indices = plot_gradcam(
        run_dir, dataset, predictions
    )

    manifest = {
        "experiment": experiment,
        "error_sample_indices": error_indices,
        "gradcam_sample_indices": cam_indices,
        "notes": [
            "Metrics and curves use saved original evaluation records.",
            "Grad-CAM is regenerated from the saved CPU-model checkpoint.",
            "Error examples are selected by highest wrong-prediction confidence.",
            "Grad-CAM maps are normalized separately.",
        ],
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Report figures ready. Main experiment: {experiment}")


if __name__ == "__main__":
    main()
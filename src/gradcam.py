import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

from data import build_transforms, CIFAR10_MEAN, CIFAR10_STD
from models import build_model


def compute_cam(features, logits, class_index, retain_graph=False):
    """计算指定类别相对于最后一组卷积特征的 Grad-CAM。"""
    gradients = torch.autograd.grad(
        outputs=logits[0, class_index],
        inputs=features,
        retain_graph=retain_graph,
    )[0]

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=(32, 32),
        mode="bilinear",
        align_corners=False,
    )

    cam = cam[0, 0].detach().cpu().numpy()
    maximum = float(cam.max())
    if maximum > 0:
        cam = cam / maximum

    return cam


def select_samples(targets, predictions, confidence):
    """选择四类正确样本和四个高置信度错误样本。"""
    selected = []

    # 每类选取第一个正确样本，最多展示四类。
    for class_index in range(10):
        candidates = np.flatnonzero(
            (targets == class_index) & (predictions == targets)
        )
        if len(candidates):
            selected.append(int(candidates[0]))
        if len(selected) == 4:
            break

    wrong = np.flatnonzero(targets != predictions)
    wrong = wrong[np.argsort(-confidence[wrong])]
    selected.extend(int(index) for index in wrong[:4])

    return selected


def main():
    run_dir = Path("outputs/resnet_no_se")
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, eval_transform = build_transforms(use_augmentation=False)
    dataset = CIFAR10(
        root=checkpoint["config"]["data_dir"],
        train=False,
        download=False,
        transform=eval_transform,
    )
    class_names = dataset.classes

    with np.load(run_dir / "predictions.npz") as saved:
        targets = saved["targets"]
        predictions = saved["predictions"]
        confidence = saved["confidence"]

    # 确认预测文件与当前数据集顺序一致。
    if not np.array_equal(targets, np.asarray(dataset.targets)):
        raise RuntimeError("预测文件与测试集标签不一致，请重新运行 evaluate.py。")

    selected = select_samples(targets, predictions, confidence)
    if not selected:
        raise RuntimeError("没有可展示的样本。")

    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    fig, axes = plt.subplots(
        len(selected),
        3,
        figsize=(10, 2.6 * len(selected)),
        squeeze=False,
    )
    records = []

    for row, index in enumerate(selected):
        image, target = dataset[index]
        inputs = image.unsqueeze(0)

        # 显式取得最后一个残差阶段的输出，避免反向钩子
        # 与模型中的原位 ReLU 操作发生冲突。
        x = model.stem(inputs)
        x = model.layer1(x)
        x = model.layer2(x)
        features = model.layer3(x)
        logits = model.classifier(features)

        probabilities = logits.softmax(dim=1)
        prediction = int(probabilities.argmax(dim=1).item())
        probability = float(probabilities[0, prediction].item())

        if prediction != int(predictions[index]):
            raise RuntimeError(
                "当前模型与预测文件不一致，请重新运行 evaluate.py。"
            )

        predicted_cam = compute_cam(
            features,
            logits,
            prediction,
            retain_graph=prediction != target,
        )

        if prediction == target:
            true_cam = predicted_cam
        else:
            true_cam = compute_cam(features, logits, target)

        original = (
            (image * std + mean)
            .clamp(0, 1)
            .permute(1, 2, 0)
            .numpy()
        )

        correct = prediction == target
        status = "Correct" if correct else "Incorrect"

        axes[row, 0].imshow(original, interpolation="nearest")
        axes[row, 0].set_title(
            f"{status} | Test #{index}\n"
            f"True: {class_names[target]}"
        )

        for column, cam, title in (
            (
                1,
                predicted_cam,
                f"Predicted: {class_names[prediction]}\n"
                f"Confidence: {probability:.1%}",
            ),
            (
                2,
                true_cam,
                f"Target class: {class_names[target]}",
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
            axes[row, column].set_title(title)

        for axis in axes[row]:
            axis.axis("off")

        records.append({
            "test_index": index,
            "true_class": class_names[target],
            "predicted_class": class_names[prediction],
            "confidence": probability,
            "correct": correct,
        })

        print(
            f"Test #{index}: "
            f"true={class_names[target]}, "
            f"pred={class_names[prediction]}, "
            f"confidence={probability:.2%}"
        )

    fig.suptitle(
        "Grad-CAM: original / predicted-class map / true-class map\n"
        "Warm colors indicate stronger positive contribution",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    figure_path = figure_dir / "gradcam.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    with (run_dir / "gradcam_samples.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved: {figure_path}")
    print(f"Saved: {run_dir / 'gradcam_samples.csv'}")


if __name__ == "__main__":
    main()
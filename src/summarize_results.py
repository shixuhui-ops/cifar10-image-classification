import csv
import json
from pathlib import Path

import numpy as np


EXPERIMENTS = [
    "baseline",
    "baseline_matched",
    "resnet_se",
    "resnet_se_noaug",
    "resnet_se_no_smoothing",
    "resnet_no_se",
    "resnet18_transfer",
    "gpu_resnet_seed42",
]


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main():
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name in EXPERIMENTS:
        run_dir = Path("outputs") / name
        config = read_json(run_dir / "config.json")
        history = read_json(run_dir / "history.json")
        metrics = read_json(run_dir / "evaluation_summary.json")
        original = read_json(run_dir / "test_metrics.json")

        matrix = np.loadtxt(
            run_dir / "confusion_matrix.csv",
            delimiter=",",
            skiprows=1,
            dtype=int,
        )

        if matrix.shape != (10, 10) or matrix.sum() != 10000:
            raise ValueError(f"{name}: 混淆矩阵尺寸或样本数异常。")

        matrix_accuracy = float(matrix.trace() / matrix.sum())
        for accuracy in (
            metrics["test_accuracy"],
            original["test_accuracy"],
        ):
            if not np.isclose(
                matrix_accuracy, accuracy, rtol=0, atol=1e-10
            ):
                raise ValueError(f"{name}: 测试准确率记录不一致。")

        rows.append({
            "experiment": name,
            "device": config["device"],
            "seed": config["seed"],
            "max_epochs": config["epochs"],
            "epochs_run": len(history),
            "best_epoch": metrics["best_epoch"],
            "batch_size": config["batch_size"],
            "augmentation": not config.get("no_augmentation", False),
            "label_smoothing": config["label_smoothing"],
            "total_parameters": metrics["total_parameters"],
            "trainable_parameters": metrics["trainable_parameters"],
            "best_val_accuracy": metrics["best_val_accuracy"],
            "test_accuracy": metrics["test_accuracy"],
            "macro_f1": metrics["macro_f1"],
            # 不含数据下载、最终测试和评估绘图时间。
            "training_validation_seconds": round(
                sum(item["elapsed_seconds"] for item in history), 2
            ),
        })

    csv_path = report_dir / "experiment_summary.csv"
    with csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 主分析使用 CPU 正式实验，初步实验及 GPU 复现单独保留。
    formal_cpu_rows = [
        row for row in rows
        if row["experiment"] not in {"baseline", "gpu_resnet_seed42"}
        and row["device"] == "cpu"
    ]
    selected = max(
        formal_cpu_rows,
        key=lambda row: row["best_val_accuracy"],
    )

    selected_dir = Path("outputs") / selected["experiment"]
    class_names = read_json(
        selected_dir / "config.json"
    )["class_names"]
    matrix = np.loadtxt(
        selected_dir / "confusion_matrix.csv",
        delimiter=",",
        skiprows=1,
        dtype=int,
    )

    per_class = []
    for index, name in enumerate(class_names):
        support = int(matrix[index].sum())
        per_class.append({
            "class": name,
            "correct": int(matrix[index, index]),
            "support": support,
            "accuracy": float(matrix[index, index] / support),
        })

    confusions = sorted(
        [
            {
                "true_class": class_names[i],
                "predicted_class": class_names[j],
                "count": int(matrix[i, j]),
            }
            for i in range(10)
            for j in range(10)
            if i != j
        ],
        key=lambda item: -item["count"],
    )

    summary = {
        "experiments": rows,
        "selection_rule": (
            "Highest validation accuracy among formal CPU experiments; "
            "pilot and GPU reproduction are reported separately."
        ),
        "selected_experiment": selected["experiment"],
        "selected_model_per_class": per_class,
        "selected_model_top_confusions": confusions[:8],
        "limitations": [
            "Single-seed experiments; no statistical significance claim.",
            "CPU and GPU runs use different PyTorch versions.",
            "Pretrained model uses external ImageNet data, 96x96 inputs, "
            "and a different training budget.",
            "Test losses with different label smoothing are not comparable.",
            "Reported timings are descriptive, not controlled benchmarks.",
        ],
    }

    json_path = report_dir / "results_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("All eight experiments passed consistency checks.")
    print(f"Selected by validation: {selected['experiment']}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
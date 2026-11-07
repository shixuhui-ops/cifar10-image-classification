import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


REPORT_DIR = Path("reports")


def read_csv(path):
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())

    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
            "ci95_half_width": 0.0,
        }

    std = float(values.std(ddof=1))
    critical = float(stats.t.ppf(0.975, n - 1))
    half_width = critical * std / np.sqrt(n)

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
        "ci95_half_width": float(half_width),
    }


def paired_comparison(name, rows_a, rows_b):
    values_a = {
        int(row["seed"]): float(row["test_accuracy"])
        for row in rows_a
    }
    values_b = {
        int(row["seed"]): float(row["test_accuracy"])
        for row in rows_b
    }

    common_seeds = sorted(
        set(values_a).intersection(values_b)
    )

    paired_a = np.asarray(
        [values_a[seed] for seed in common_seeds],
        dtype=float,
    )
    paired_b = np.asarray(
        [values_b[seed] for seed in common_seeds],
        dtype=float,
    )
    differences = paired_a - paired_b

    result = confidence_interval(differences)
    result.update(
        {
            "comparison": name,
            "seeds": common_seeds,
            "mean_difference": result["mean"],
            "difference_ci95_low": result["ci95_low"],
            "difference_ci95_high": result["ci95_high"],
        }
    )

    if len(common_seeds) >= 2:
        test = stats.ttest_rel(
            paired_a,
            paired_b,
        )
        result["t_statistic"] = float(test.statistic)
        result["p_value"] = float(test.pvalue)
    else:
        result["t_statistic"] = None
        result["p_value"] = None

    result.pop("mean", None)
    result.pop("std", None)
    result.pop("ci95_half_width", None)

    return result


def main():
    multiseed_rows = read_csv(
        REPORT_DIR / "multiseed_runs.csv"
    )
    augmentation_rows = read_csv(
        REPORT_DIR / "augmentation_multiseed_runs.csv"
    )

    groups = {
        "resnet_no_se": [
            row
            for row in multiseed_rows
            if row["model"] == "resnet_no_se"
        ],
        "resnet_se": [
            row
            for row in multiseed_rows
            if row["model"] == "resnet_se"
            and float(row["label_smoothing"]) == 0.1
        ],
        "resnet_se_no_smoothing": [
            row
            for row in multiseed_rows
            if row["model"] == "resnet_se"
            and float(row["label_smoothing"]) == 0.0
        ],
    }

    augmentation_groups = {
        policy: [
            row
            for row in augmentation_rows
            if row["augmentation_policy"] == policy
        ]
        for policy in [
            "all",
            "crop",
            "flip",
            "color",
            "none",
        ]
    }

    group_summary = {}

    for name, rows in groups.items():
        values = [
            float(row["test_accuracy"])
            for row in rows
        ]
        group_summary[name] = confidence_interval(values)

    for name, rows in augmentation_groups.items():
        values = [
            float(row["test_accuracy"])
            for row in rows
        ]
        group_summary[f"augmentation_{name}"] = (
            confidence_interval(values)
        )

    comparisons = [
        paired_comparison(
            "resnet_no_se - resnet_se",
            groups["resnet_no_se"],
            groups["resnet_se"],
        ),
        paired_comparison(
            "resnet_se_smoothing - resnet_se_no_smoothing",
            groups["resnet_se"],
            groups["resnet_se_no_smoothing"],
        ),
    ]

    for policy in ["crop", "flip", "color"]:
        comparisons.append(
            paired_comparison(
                f"augmentation_{policy} - augmentation_none",
                augmentation_groups[policy],
                augmentation_groups["none"],
            )
        )

    output = {
        "group_summary": group_summary,
        "paired_comparisons": comparisons,
        "method": {
            "confidence_level": 0.95,
            "test": "paired t-test",
            "standard_deviation": "sample standard deviation",
        },
        "limitations": [
            "Each comparison uses only three seeds.",
            "The paired t-test has low statistical power at n=3.",
            "Results should be described as exploratory evidence.",
            "The all-augmentation condition currently has only seed=42.",
        ],
    }

    output_path = REPORT_DIR / "statistical_analysis.json"
    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {output_path}")

    for name, values in group_summary.items():
        print(
            f"{name}: "
            f"mean={values['mean']:.4f}, "
            f"std={values['std']:.4f}, "
            f"95% CI="
            f"[{values['ci95_low']:.4f}, "
            f"{values['ci95_high']:.4f}]"
        )

    for comparison in comparisons:
        print(
            f"{comparison['comparison']}: "
            f"mean difference="
            f"{comparison['mean_difference']:.4f}, "
            f"95% CI="
            f"[{comparison['difference_ci95_low']:.4f}, "
            f"{comparison['difference_ci95_high']:.4f}], "
            f"p={comparison['p_value']}"
        )


if __name__ == "__main__":
    main()
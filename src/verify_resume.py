"""用小规模真实训练验证断点续训的一致性。"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

import resume_train
import train


def assert_same(left, right, name):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), f"{name} 不一致"
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), f"{name} 字段不一致"
        for key in left:
            assert_same(left[key], right[key], f"{name}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), f"{name} 长度不一致"
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, f"{name}[{index}]")
    else:
        assert left == right, f"{name}: {left!r} != {right!r}"


def load_checkpoint(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


def main():
    Path("outputs").mkdir(exist_ok=True)
    root = Path(
        tempfile.mkdtemp(
            prefix="rng_equivalence_smoke_",
            dir="outputs",
        )
    )
    continuous = root / "continuous"
    resumed = root / "resumed"
    resumed.mkdir()

    print(f"检查结果目录：{root.resolve()}", flush=True)

    original_save = torch.save

    def save_with_snapshot(checkpoint, path, *args, **kwargs):
        original_save(checkpoint, path, *args, **kwargs)

        # 保存第 1 轮快照，模拟在两轮训练中途发生中断。
        filename = Path(path).name
        if checkpoint["epoch"] == 1 and filename in {
            "last_model.pt",
            "best_model.pt",
        }:
            original_save(
                checkpoint,
                resumed / filename,
                *args,
                **kwargs,
            )

    training_args = [
        "train.py",
        "--model", "baseline",
        "--epochs", "2",
        "--batch-size", "8",
        "--limit-batches", "2",
        "--seed", "42",
        "--output-dir", str(continuous),
    ]

    with patch.object(sys, "argv", training_args):
        with patch.object(torch, "save", save_with_snapshot):
            train.main()

    resume_args = [
        "resume_train.py",
        "--resume", str(resumed / "last_model.pt"),
        "--epochs", "2",
        "--limit-batches", "2",
    ]

    with patch.object(sys, "argv", resume_args):
        resume_train.main()

    expected = load_checkpoint(continuous / "last_model.pt")
    actual = load_checkpoint(resumed / "last_model.pt")

    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
    ):
        assert_same(expected[key], actual[key], key)
        print(f"PASS: {key}")

    # 运行耗时自然不同，比较其余训练历史字段。
    def comparable_history(checkpoint):
        return [
            {
                key: value
                for key, value in row.items()
                if key != "elapsed_seconds"
            }
            for row in checkpoint["history"]
        ]

    assert_same(
        comparable_history(expected),
        comparable_history(actual),
        "history",
    )
    print("PASS: history")
    print("断点续训一致性检查通过")


if __name__ == "__main__":
    main()
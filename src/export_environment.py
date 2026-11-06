import argparse
import json
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path

import torch
import torchvision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=["cpu", "gpu"], required=True)
    args = parser.parse_args()

    packages = {}
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version

    information = {
        "environment_name": args.name,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "packages": dict(sorted(packages.items())),
    }

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    environment_path = report_dir / f"environment_{args.name}.json"
    environment_path.write_text(
        json.dumps(information, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 使用版本号，而不是本机 wheel 的绝对路径，便于其他电脑安装。
    excluded = {"pip", "setuptools", "wheel"}
    requirements = [
        f"{name}=={version}"
        for name, version in sorted(
            packages.items(), key=lambda item: item[0].lower()
        )
        if name.lower() not in excluded
    ]

    requirements_path = Path(f"requirements-{args.name}.txt")
    requirements_path.write_text(
        "\n".join(requirements) + "\n",
        encoding="utf-8",
    )

    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Saved: {environment_path}")
    print(f"Saved: {requirements_path}")


if __name__ == "__main__":
    main()
import json
from pathlib import Path

import torch
from torch import nn

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import build_model, count_parameters


def count_flops(model, input_shape=(1, 3, 32, 32)):
    total_flops = 0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal total_flops

        batch_size = output.shape[0]
        output_height = output.shape[2]
        output_width = output.shape[3]

        kernel_height, kernel_width = module.kernel_size
        input_channels = module.in_channels
        output_channels = module.out_channels
        groups = module.groups

        operations = (
            batch_size
            * output_height
            * output_width
            * output_channels
            * (input_channels // groups)
            * kernel_height
            * kernel_width
            * 2
        )

        total_flops += operations

    def linear_hook(module, inputs, output):
        nonlocal total_flops

        batch_size = output.shape[0]
        operations = (
            batch_size
            * module.in_features
            * module.out_features
            * 2
        )

        total_flops += operations

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(
                module.register_forward_hook(conv_hook)
            )
        elif isinstance(module, nn.Linear):
            handles.append(
                module.register_forward_hook(linear_hook)
            )

    model.eval()

    with torch.inference_mode():
        sample = torch.randn(*input_shape)
        model(sample)

    for handle in handles:
        handle.remove()

    return int(total_flops)


def main():
    model_names = [
        "baseline",
        "baseline_wide",
        "resnet",
        "resnet_se",
        "baseline_flops_matched",
    ]

    results = {}

    for name in model_names:
        model = build_model(name)
        parameters = count_parameters(model)
        flops = count_flops(model)

        results[name] = {
            "trainable_parameters": parameters,
            "flops": flops,
            "gflops": flops / 1e9,
            "input_shape": [1, 3, 32, 32],
            "counting_rule": (
                "Conv2d and Linear multiply-add operations; "
                "one multiply-add counted as two FLOPs."
            ),
        }

        print(
            f"{name}: "
            f"parameters={parameters:,}, "
            f"FLOPs={flops:,}, "
            f"GFLOPs={flops / 1e9:.4f}"
        )

    output_path = Path("reports") / "flops_summary.json"
    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
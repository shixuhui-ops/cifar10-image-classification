"""保存和恢复 Python、NumPy、PyTorch 的随机数状态。"""

import random

import numpy as np
import torch


def capture_rng_state():
    numpy_state = np.random.get_state()

    return {
        "python": random.getstate(),
        # 转为普通列表，兼容 torch.load(weights_only=True)。
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(state):
    random.setstate(state["python"])

    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )

    # 检查点可能通过 map_location 被加载到其他设备。
    torch.set_rng_state(state["torch"].cpu())

    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["cuda"]]
        )
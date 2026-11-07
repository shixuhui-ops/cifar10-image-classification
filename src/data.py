from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def build_transforms(
    policy: str = "all",
    use_augmentation: bool | None = None,
):
    """构建训练集和验证/测试集的数据变换。"""

    if use_augmentation is not None:
        policy = "all" if use_augmentation else "none"

    valid_policies = {"none", "crop", "flip", "color", "all"}
    if policy not in valid_policies:
        raise ValueError(
            f"未知增强策略: {policy}，可选值为 {sorted(valid_policies)}"
        )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_operations = []

    if policy in {"crop", "all"}:
        train_operations.append(
            transforms.RandomCrop(32, padding=4)
        )

    if policy in {"flip", "all"}:
        train_operations.append(
            transforms.RandomHorizontalFlip()
        )

    if policy in {"color", "all"}:
        train_operations.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            )
        )

    train_operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_transform = transforms.Compose(train_operations)

    return train_transform, eval_transform


def create_dataloaders(
    data_dir: str = "data",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    use_augmentation: bool = True,
    augmentation_policy: str | None = None,
):
    """
    下载 CIFAR-10，并划分训练集、验证集和测试集。

    返回:
        loaders: 包含 train、val、test 三个 DataLoader 的字典
        class_names: CIFAR-10 类别名称
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio 必须在 0 和 1 之间。")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if augmentation_policy is None:
        augmentation_policy = (
            "all" if use_augmentation else "none"
        )

    train_transform, eval_transform = build_transforms(
        augmentation_policy
    )

    augmented_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform,
    )

    evaluation_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=False,
        transform=eval_transform,
    )

    total_size = len(augmented_dataset)
    val_size = int(total_size * val_ratio)

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(
        total_size,
        generator=generator,
    ).tolist()

    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]

    train_dataset = Subset(augmented_dataset, train_indices)
    val_dataset = Subset(evaluation_dataset, val_indices)

    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=eval_transform,
    )

    pin_memory = torch.cuda.is_available()

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    return loaders, CLASS_NAMES
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ConvBNAct(nn.Sequential):
    """卷积、BatchNorm 和激活函数组合。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class BaselineCNN(nn.Module):
    """基础卷积神经网络，用于基准对比。"""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNAct(3, 64),
            ConvBNAct(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            ConvBNAct(64, 128),
            ConvBNAct(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            ConvBNAct(128, 256),
            ConvBNAct(256, 256),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class WideBaselineCNN(nn.Module):
    """参数量接近残差网络的加宽 CNN 对照模型。"""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNAct(3, 96),
            ConvBNAct(96, 96),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            ConvBNAct(96, 192),
            ConvBNAct(192, 192),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            ConvBNAct(192, 400),
            ConvBNAct(400, 400),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(200, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。"""

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        weights = self.pool(x).view(batch_size, channels)
        weights = self.fc(weights).view(batch_size, channels, 1, 1)
        return x * weights


class ResidualSEBlock(nn.Module):
    """带 SE 注意力的残差卷积块。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = SEBlock(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)

        out = out + identity
        return self.activation(out)


class ImprovedResNetSE(nn.Module):
    """适用于 CIFAR-10 的轻量级残差注意力网络。"""

    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = ConvBNAct(3, 64)

        self.layer1 = nn.Sequential(
            ResidualSEBlock(64, 64),
            ResidualSEBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            ResidualSEBlock(64, 128, stride=2),
            ResidualSEBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            ResidualSEBlock(128, 256, stride=2),
            ResidualSEBlock(256, 256),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)

class TransferResNet18(nn.Module):
    """ImageNet 预训练 ResNet-18，微调 layer4 和分类层。"""

    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            num_classes,
        )

        # 冻结前面的特征提取层。
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.layer4.parameters():
            parameter.requires_grad = True

        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

        # 输入沿用现有数据模块的 CIFAR-10 标准化。
        # 在模型内部还原像素，再转换为 ImageNet 标准化。
        self.register_buffer(
            "cifar_mean",
            torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "cifar_std",
            torch.tensor((0.2023, 0.1994, 0.2010)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )

    def train(self, mode=True):
        super().train(mode)

        # 冻结部分的 BatchNorm 均值和方差也保持固定。
        self.backbone.eval()
        self.backbone.layer4.train(mode)
        self.backbone.fc.train(mode)
        return self

    def forward(self, x):
        x = x * self.cifar_std + self.cifar_mean
        x = F.interpolate(
            x,
            size=(96, 96),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        x = (x - self.imagenet_mean) / self.imagenet_std
        return self.backbone(x)

def build_model(name="resnet_se", num_classes=10):
    """创建基础 CNN、残差网络或 SE 残差网络。"""
    model_name = name.lower()

    if model_name == "resnet18_transfer":
        # 加载检查点时不需要再次初始化 ImageNet 权重。
        return TransferResNet18(
            num_classes=num_classes,
            pretrained=False,
        )

    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)

    if model_name == "baseline_wide":
        return WideBaselineCNN(num_classes=num_classes)

    if model_name == "resnet":
        model = ImprovedResNetSE(num_classes=num_classes)

        # 只移除通道注意力，其余结构与 SE 模型一致。
        for module in model.modules():
            if isinstance(module, ResidualSEBlock):
                module.attention = nn.Identity()

        return model

    if model_name in {"resnet_se", "improved"}:
        return ImprovedResNetSE(num_classes=num_classes)

    raise ValueError(
        f"未知模型: {name}。"
        "可选模型为 baseline、resnet 或 resnet_se。"
    )

def count_parameters(model):
    """统计可训练参数量。"""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    for model_name in ("baseline", "resnet_se"):
        model = build_model(model_name)
        sample = torch.randn(2, 3, 32, 32)
        output = model(sample)
        print(
            f"{model_name}: "
            f"output_shape={tuple(output.shape)}, "
            f"parameters={count_parameters(model):,}"
        )
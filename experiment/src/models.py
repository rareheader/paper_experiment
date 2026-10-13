"""按论文正文给出的尺寸搭建的模型。

MNIST / CIFAR-10 的卷积网络按正文尺寸反推(见阅读笔记 14.3):
- MNIST:两层 5x5 卷积 + 320x50、50x10 全连接。无填充、两次 2x2 池化,28x28 -> 4x4,
  第二层输出通道 20;第一层通道原文没写死,取同类结构常见配法 10。
- CIFAR-10:同样的两层 5x5 卷积,通道 3->6->16,池化后 400x120、120x84、84x10。
- CIFAR-100:ResNet-50,按 32x32 的通行改法把第一层换成 3x3 步幅 1 并去掉开头最大池化
  (这个改法不是原文句子,原文没写 CIFAR 茎怎么改)。

警告:ResNet-50 只建议在 GPU 上跑,本机 CPU 上 120 轮会变成以天计的空转。
"""
import torch.nn as nn
import torchvision


class MNISTNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(320, 50), nn.ReLU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class CIFAR10Net(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(400, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def resnet50_cifar(num_classes: int) -> nn.Module:
    m = torchvision.models.resnet50(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def build_model(dataset: str, arch: str | None = None) -> nn.Module:
    if arch == "resnet50":
        ncls = {"mnist": 10, "cifar10": 10, "cifar100": 100}[dataset]
        if dataset == "mnist":
            raise ValueError("ResNet-50 只用于 CIFAR 系列数据集")
        return resnet50_cifar(ncls)
    return {"mnist": MNISTNet, "cifar10": CIFAR10Net}[dataset]()


def param_shapes(dataset: str, arch: str | None = None) -> list[tuple]:
    """逐层余弦用的"层"定义为每个参数张量(权重与偏置),与笔记第 9 节一致。"""
    return [tuple(p.shape) for p in build_model(dataset, arch).parameters()]

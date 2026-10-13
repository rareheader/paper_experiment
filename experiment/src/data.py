"""数据加载(整集预转张量)、Dirichlet Non-IID 划分与标签打乱攻击。

为 CPU 训练效率,数据不走 PIL/DataLoader:原始数据一次性转成 uint8 张量
(N,C,H,W),训练时按索引取批次、在批次内做归一化,数据开销接近零。

Dirichlet 划分与论文一致:每个类别抽一份 Dir(beta) 比例分给 N 个用户,
beta 越小越不均衡,beta=100 近似 IID(笔记 14.2)。
标签打乱是数据级攻击:恶意用户对标签空间做一个固定随机置换后再正常训练(笔记 14.4)。
"""
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

NUM_CLASSES = {"mnist": 10, "cifar10": 10, "cifar100": 100}

# 与 torchvision 常用 Normalize 一致(通道均值/标准差)
NORMALIZE = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def load_tensors(dataset: str, train: bool):
    """返回 (X uint8 N×C×H×W, y int64)。优先读 .pt 缓存;否则走 torchvision 并写缓存。

    CIFAR-10 的 .pt 缓存由 prepare_cifar10_from_parquet.py 生成——本机到
    torchvision 默认源太慢,见该脚本的模块注释。
    """
    split = "train" if train else "test"
    cache = DATA_ROOT / f"{dataset}_{split}.pt"
    if cache.exists():
        d = torch.load(cache)
        return d["X"], d["y"]
    if dataset == "mnist":
        ds = datasets.MNIST(DATA_ROOT, train=train, download=True)
        X = ds.data.unsqueeze(1).contiguous()
        y = ds.targets.clone()
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(DATA_ROOT, train=train, download=True)
        X = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor(ds.targets, dtype=torch.long)
    elif dataset == "cifar100":
        ds = datasets.CIFAR100(DATA_ROOT, train=train, download=True)
        X = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor(ds.targets, dtype=torch.long)
    else:
        raise ValueError(dataset)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "y": y}, cache)
    return X, y


def normalize_batch(x_uint8: torch.Tensor, dataset: str) -> torch.Tensor:
    mean, std = NORMALIZE[dataset]
    mt = torch.tensor(mean).view(-1, 1, 1)
    st = torch.tensor(std).view(-1, 1, 1)
    return (x_uint8.float() / 255.0 - mt) / st


def dirichlet_partition(labels, n_users: int, beta: float, rng: np.random.Generator):
    """每个类别独立抽一份 Dir(beta) 比例分给 n_users 个用户。"""
    labels = np.asarray(labels)
    ncls = int(labels.max()) + 1
    user_idx: list[list[int]] = [[] for _ in range(n_users)]
    for c in range(ncls):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        p = rng.dirichlet(np.repeat(beta, n_users))
        cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
        for u, part in enumerate(np.split(idx_c, cuts)):
            user_idx[u].extend(part.tolist())
    out = []
    for u in range(n_users):
        rng.shuffle(user_idx[u])
        out.append(np.asarray(user_idx[u], dtype=np.int64))
    return out

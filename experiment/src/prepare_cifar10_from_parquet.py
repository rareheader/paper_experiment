"""从 HuggingFace parquet 构建 CIFAR-10 张量缓存(data/cifar10_train.pt / cifar10_test.pt)。

为什么存在这个脚本:本机到 www.cs.toronto.edu(torchvision 默认源)的直连速度
只有 ~25 kB/s,自带下载要 2 小时以上;hf-mirror.com 的 parquet 镜像约 460 kB/s。
脚本把 parquet 里的 PNG 图像解码成 uint8 张量 (N,3,32,32),存成 .pt 缓存,
data.load_tensors 会优先读缓存,从而完全绕开 torchvision 的原始包下载。

用法:
    # 先手动下载两个 parquet(约 145 MB)到 data/_hf/:
    #   https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet
    #   https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet
    python src/prepare_cifar10_from_parquet.py
"""
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "data" / "_hf"


def convert(split: str):
    table = pq.read_table(HF / f"cifar10_{split}.parquet")
    img_col = "img" if "img" in table.column_names else "image"
    label_col = "label" if "label" in table.column_names else "labels"
    imgs = table.column(img_col).to_pylist()
    labels = table.column(label_col).to_pylist()
    n = len(imgs)
    X = torch.empty(n, 3, 32, 32, dtype=torch.uint8)
    for i, rec in enumerate(imgs):
        arr = np.asarray(Image.open(io.BytesIO(rec["bytes"])), dtype=np.uint8)  # HWC
        X[i] = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        if i % 10000 == 0:
            print(f"{split} {i}/{n}", flush=True)
    y = torch.tensor(labels, dtype=torch.long)
    out = ROOT / "data" / f"cifar10_{split}.pt"
    torch.save({"X": X, "y": y}, out)
    print("saved", out, tuple(X.shape), tuple(y.shape), "labels:", sorted(set(labels))[:12])


if __name__ == "__main__":
    convert("train")
    convert("test")

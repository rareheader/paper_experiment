"""DLG 梯度重建对比:明文梯度 vs 加法掩码后的梯度(对应论文图 9 的结论)。

DLG(Deep Leakage from Gradients, Zhu et al., NeurIPS 2019):攻击者拿到某用户
一步训练的梯度后,优化一张虚拟图像,使其在模型上的梯度逼近观测梯度,从而重建样本。

重建网络与流程严格对齐官方实现(mit-han-lab/dlg):
- 三层 5x5 卷积 + Sigmoid、无池化(stride 2 下采样),权重 uniform(-0.5, 0.5);
- 虚拟图像与虚拟标签(softmax 参数化)一起用默认 LBFGS 优化 300 轮;
- 匹配损失 = 逐层 ((gx - gy)**2).sum() 之和。

注意:重建实验不用论文的 ReLU+MaxPool 训练网络——max-pool 的稀疏反向传播会让
"图像 -> 参数梯度"映射出现简并(不同图像产生相同参数梯度),官方 DLG 演示同样
避开了池化层。MNIST 训练实验仍用 models.py 里按论文还原的网络。

- Case A(明文):攻击者拿到 g_i          -> 预期重建出数字
- Case B(加法掩码):攻击者拿到 [g]_1 = g_i + PRG(sk1),掩码未知且量级 ~U(-10,10)
  -> 期望匹配损失停在量级远大的平台上,重建结果保持噪声
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_tensors  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "results"


class DLGDemoNet(nn.Module):
    """官方 DLG 演示用的 LeNet 家族结构(MNIST 单通道版,无池化)。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1),
            act(),
        )
        self.fc = nn.Linear(588, num_classes)

    def forward(self, x):
        out = self.body(x)
        out = out.flatten(1)
        return self.fc(out)


def dlg_weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


def gradient_of(model, x, y):
    model.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def infer_label(observed):
    """iDLG(Zhao et al., 2020)式标签推断:最后一层偏置的梯度 = softmax - onehot,
    其中真实标签分量恒为负、其余恒为正,取最小分量位置即真实标签。"""
    bias_grad = observed["grads"][-1].flatten()
    return int(torch.argmin(bias_grad))


def dlg_reconstruct(model, observed, steps=300, seed=0, restarts=3):
    """LBFGS 求解虚拟图像(官方 DLG 流程 + iDLG 标签推断 + 多重启)。

    标签先解析推断再固定,只优化虚拟图像:DLG 原版的"图像+标签联合 LBFGS"
    在 MNIST 上容易陷进错误盆地(实测标签联合优化停在 match~1e3,图像不收敛),
    iDLG 指出标签本可以从梯度里解析得到,固定后图像求解稳定得多。
    对掩码情形这还是更苛刻的攻击设定:攻击者已经知道真实标签,仍然重建不出图像。

    LBFGS 偶发掉进坏盆地(实测 3 个种子里约 1 个),做多重启取匹配损失最小者。
    """
    best = None
    for r in range(restarts):
        torch.manual_seed(seed + r)
        dummy_x = torch.randn_like(observed["x"]).requires_grad_(True)
        label = infer_label(observed)
        ncls = observed["grads"][-1].shape[0]
        dummy_y = F.one_hot(torch.tensor([label]), ncls).float()
        optimizer = torch.optim.LBFGS([dummy_x])
        match = None

        for _ in range(steps):
            def closure():
                optimizer.zero_grad()
                dummy_loss = F.cross_entropy(model(dummy_x), dummy_y)
                dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
                nonlocal_match = sum(((gx - gy) ** 2).sum() for gx, gy in
                                     zip(dummy_grads, observed["grads"]))
                nonlocal_match.backward()
                return nonlocal_match

            optimizer.step(closure)

        with torch.enable_grad():
            def final_closure():
                optimizer.zero_grad()
                dummy_loss = F.cross_entropy(model(dummy_x), dummy_y)
                dummy_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
                nonlocal_match = sum(((gx - gy) ** 2).sum() for gx, gy in
                                     zip(dummy_grads, observed["grads"]))
                nonlocal_match.backward()
                return nonlocal_match
            match = float(final_closure().detach())

        img = dummy_x.detach().squeeze().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
        if best is None or match < best[1]:
            best = (img, match, label)
    return best


def main(steps=300):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    X, y = load_tensors("mnist", train=True)
    i = int(rng.integers(len(y)))
    # 官方 DLG 用未归一化的 [0,1] 输入;归一化会把 Sigmoid 推进饱和区,重建变慢
    x = X[i].float().div(255.0).unsqueeze(0)
    yt = torch.tensor([int(y[i])])

    model = DLGDemoNet()
    model.apply(dlg_weights_init)
    model.eval()

    true_grads = gradient_of(model, x, yt)
    # 加法掩码:PRG(sk1) 的输出近似 U(-10, 10)(笔记 3.2),量级远大于梯度本身
    masks = [torch.from_numpy(rng.uniform(-10, 10, size=p.shape).astype(np.float32))
             for p in model.parameters()]
    masked_grads = [g + m for g, m in zip(true_grads, masks)]

    img_plain, match_plain, label_plain = dlg_reconstruct(
        model, {"x": x, "grads": true_grads}, steps=steps)
    img_masked, match_masked, label_masked = dlg_reconstruct(
        model, {"x": x, "grads": masked_grads}, steps=steps)

    orig = x.squeeze().numpy()
    orig_n = (orig - orig.min()) / (orig.max() - orig.min() + 1e-12)
    mse_plain = float(np.mean((img_plain - orig_n) ** 2))
    mse_masked = float(np.mean((img_masked - orig_n) ** 2))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    panels = [
        (orig_n, f"原始图像 (label={yt.item()})"),
        (img_plain, f"Case A 明文梯度重建 (推断标签 {label_plain})\n匹配损失 {match_plain:.2e}"),
        (img_masked, f"Case B 加法掩码后重建 (推断标签 {label_masked})\n匹配损失 {match_masked:.2e}"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("DLG 梯度重建:明文 vs SFLBR 加法掩码(论文图 9 的定性复现)", fontsize=12)
    fig.tight_layout()
    out_png = OUT_DIR / "dlg_comparison.png"
    fig.savefig(out_png, dpi=150)
    print(f"DLG RESULT steps={steps} true_label={yt.item()} label_plain={label_plain} "
          f"label_masked={label_masked} match_plain={match_plain:.4e} match_masked={match_masked:.4e} "
          f"mse_plain={mse_plain:.4f} mse_masked={mse_masked:.4f} saved={out_png}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    main(args.steps)

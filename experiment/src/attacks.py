"""梯度级拜占庭攻击的实现(数据级攻击 label_shuffle 在 data.py)。

- alie:恶意用户按诚实更新的逐坐标均值 mu 加 z 倍标准差,发送 mu + z*sigma,
  z 取论文实验用的 1.5。这是"全知攻击者"的最坏情况设定,按论文原文描述实现。
- agr:恶意更新取中位数锚的反方向单位向量,缩放到典型梯度范数,再乘系数 lam
  (笔记 14.4:反方向单位向量 x 优化出来的系数 lam;这里 lam 默认 1.0,可用 --agr-lam 调)。

两个函数都直接修改传入的 grads 列表(浮点扁平向量)。
"""
import numpy as np


def apply_alie(grads: list[np.ndarray], mal_idx: list[int], z: float = 1.5):
    honest = np.stack([g for i, g in enumerate(grads) if i not in set(mal_idx)])
    mu = honest.mean(axis=0)
    sigma = honest.std(axis=0)
    adv = mu + z * sigma
    for i in mal_idx:
        grads[i] = adv
    return grads


def apply_agr(grads: list[np.ndarray], mal_idx: list[int], lam: float = 1.0, eps: float = 1e-12):
    anchor = np.median(np.stack(grads), axis=0)  # 攻击者对"诚实方向"的代理
    scale = float(np.mean([np.linalg.norm(g) for g in grads]))
    unit = anchor / max(float(np.linalg.norm(anchor)), eps)
    for i in mal_idx:
        grads[i] = -lam * scale * unit
    return grads

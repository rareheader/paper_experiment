"""聚合规则:FedAvg / Krum / 坐标中值 / 截尾均值 / SFLBR。

SFLBR 与笔记第 8、9、10 节公式一致:
    med        = 每个坐标取中位数
    mu_ij      = 第 j 层上 g_i 与 med 的余弦(层 = 每个参数张量)
    score(mu)  = (mu+1)^2   若 -1 <= mu <= 0
               = mu + 1     若 0 < mu <= 1
    rho_i      = sum_j score(mu_ij) / sum_k sum_j score(mu_kj)
    g_agg      = sum_i rho_i * g_i

密码学部分(双掩码、随机缩放 R/r)不影响精度结果:加法掩码在加权求和时整段抵消,
乘法掩码和缩放不改变余弦与中位数等式。笔记第 11 节已经用三用户闭环自检验证过
(最大误差 1.1e-16,numeric_result.txt),所以精度复现在明文上实现聚合规则即可。
"""
import numpy as np


def score_func(mu: float) -> float:
    return (mu + 1.0) ** 2 if mu <= 0 else mu + 1.0


def _layer_slices(shapes) -> list[tuple[int, int]]:
    sizes = [int(np.prod(s)) for s in shapes]
    bounds = np.cumsum([0] + sizes)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(sizes))]


def fedavg(grads) -> np.ndarray:
    return np.stack(grads).mean(axis=0)


def coord_median(grads) -> np.ndarray:
    return np.median(np.stack(grads), axis=0)


def trimmed_mean(grads, f: int) -> np.ndarray:
    n = len(grads)
    f = max(0, min(f, (n - 1) // 2))
    G = np.sort(np.stack(grads), axis=0)
    return G[f:n - f].mean(axis=0)


def krum(grads, f: int) -> np.ndarray:
    """标准单点 Krum:选到其余 n-f-2 个最近邻距离和最小的那条梯度。"""
    G = np.stack(grads)
    n = len(G)
    d = np.linalg.norm(G[:, None, :] - G[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    keep = max(n - f - 2, 1)
    scores = np.sort(d, axis=1)[:, :keep].sum(axis=1)
    return G[int(np.argmin(scores))]


def sflbr(grads, shapes, return_rho: bool = False):
    G = np.stack(grads)
    n = len(G)
    med = np.median(G, axis=0)
    slices = _layer_slices(shapes)
    scores = np.zeros(n)
    mus = np.zeros((n, len(slices)))
    for li, (a, b) in enumerate(slices):
        m = med[a:b]
        mn = float(np.linalg.norm(m))
        for i in range(n):
            v = G[i, a:b]
            vn = float(np.linalg.norm(v))
            if mn < 1e-12 and vn < 1e-12:
                mu = 1.0
            elif mn < 1e-12 or vn < 1e-12:
                mu = 0.0
            else:
                mu = float(v @ m / (vn * mn))
            mus[i, li] = mu
        scores += np.array([score_func(mu) for mu in mus[:, li]])
    rho = scores / scores.sum()
    agg = (rho[:, None] * G).sum(axis=0)
    if return_rho:
        return agg, rho, mus
    return agg


def aggregate(name: str, grads, shapes, mal_frac: float):
    """统一入口。返回 (聚合梯度, 诊断 dict)。"""
    n = len(grads)
    f = int(round(mal_frac * n))
    if name == "fedavg":
        return fedavg(grads), {}
    if name == "median":
        return coord_median(grads), {}
    if name == "tmean":
        return trimmed_mean(grads, f), {}
    if name == "krum":
        return krum(grads, f), {}
    if name == "sflbr":
        agg, rho, _ = sflbr(grads, shapes, return_rho=True)
        return agg, {"rho": rho.tolist()}
    raise ValueError(f"unknown aggregation: {name}")

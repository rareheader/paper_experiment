"""联邦训练主循环(明文域;密码学部分不影响精度,见 aggregation.py 模块注释)。

本地损失与论文 Fig. 3 / 笔记第 6 节一致:
    loss = CE(y, f(w_i; x)) + alpha * ||w_i - w||_2
注意是 L2 距离本身,不是 FedProx 的距离平方;alpha 按 MNIST/CIFAR-10/CIFAR-100
取 0.01/0.005/0.001。锚是本轮下发的全局模型(不建计算图)。

全局更新与算法 1 一致:
    w <- w + eta * sum_i rho_i * g_i,   g_i = w_i - w,   eta 默认 1.0

学习率:初始 0.1,在总轮数的 1/3 和 2/3 处各乘 0.1(论文是 120 轮在第 40/80 轮衰减,
等比例换算到缩短的轮数表;传 --decay-rounds 40 80 可回到论文原值)。
动量:原文写 1e-4,比常见 0.9 小三个数量级,近似无动量,这里取 0 并在 README 注明。
"""
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import attacks as atk
from aggregation import aggregate
from data import NUM_CLASSES, NORMALIZE, dirichlet_partition, load_tensors, normalize_batch
from models import build_model, param_shapes

ALPHA_AUTO = {"mnist": 0.01, "cifar10": 0.005, "cifar100": 0.001}


@dataclasses.dataclass
class FLConfig:
    dataset: str = "mnist"
    arch: str | None = None          # "resnet50" 走 GPU 大模型路径
    agg: str = "sflbr"               # sflbr | fedavg | median | tmean | krum
    attack: str = "none"             # none | label_shuffle | alie | agr
    mal_frac: float = 0.0
    alie_z: float = 1.5
    agr_lam: float = 1.0
    beta: float = 0.5                # Dirichlet 参数;100 近似 IID
    users: int = 10
    sample: int = 10                 # 每轮抽多少用户;=users 即全员参加
    rounds: int = 20
    epochs: int = 3
    batch: int = 32
    lr: float = 0.1
    eta: float = 1.0
    alpha: float | None = None       # None 时按数据集取 0.01/0.005/0.001
    weight_decay: float = 0.0
    decay_rounds: tuple | None = None  # None 时按 rounds 的 1/3、2/3 等比换算
    eval_every: int = 2
    seed: int = 0
    threads: int = 0                 # 0 = 用 torch 默认
    outdir: Path = Path("results")
    tag: str = ""


def flatten_state(model) -> np.ndarray:
    return torch.cat([p.detach().flatten() for p in model.parameters()]).cpu().numpy().astype(np.float64)


def load_flat(model, flat: np.ndarray):
    with torch.no_grad():
        off = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(flat[off:off + n].astype(np.float32)).view_as(p))
            off += n


def _norm_tensors(dataset: str):
    mean, std = NORMALIZE[dataset]
    return (torch.tensor(mean).view(-1, 1, 1), torch.tensor(std).view(-1, 1, 1))


def local_train(model, global_model, X, y, indices, label_perm, batch,
                alpha: float, lr: float, epochs: int, weight_decay: float, rng, mt, st):
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.0)
    model.train()
    n = len(indices)
    for _ in range(epochs):
        order = indices[rng.permutation(n)]
        for s in range(0, n, batch):
            ix = order[s:s + batch]
            xb = (X[ix].float() / 255.0 - mt) / st
            yb = y[ix]
            if label_perm is not None:
                yb = label_perm[yb]
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb)
            if alpha > 0:
                sq = None
                for p, a in zip(model.parameters(), global_model.parameters()):
                    d = p - a
                    sq = d.pow(2).sum() if sq is None else sq + d.pow(2).sum()
                loss = ce + alpha * (sq + 1e-12).sqrt()
            else:
                loss = ce
            loss.backward()
            opt.step()


def evaluate(model, X, y, mt, st, batch: int = 512) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for s in range(0, len(y), batch):
            xb = (X[s:s + batch].float() / 255.0 - mt) / st
            correct += int((model(xb).argmax(1) == y[s:s + batch]).sum())
    model.train()
    return correct / len(y)


def run_fl(cfg: FLConfig) -> dict:
    assert cfg.attack != "label_shuffle" or cfg.mal_frac > 0, "label_shuffle 需要恶意用户"
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    alpha = ALPHA_AUTO[cfg.dataset] if cfg.alpha is None else cfg.alpha
    decay = cfg.decay_rounds or (max(1, round(cfg.rounds / 3)), max(2, round(2 * cfg.rounds / 3)))

    Xtr, ytr = load_tensors(cfg.dataset, train=True)
    Xte, yte = load_tensors(cfg.dataset, train=False)
    mt, st = _norm_tensors(cfg.dataset)
    parts = dirichlet_partition(ytr.numpy(), cfg.users, cfg.beta, rng)
    n_mal = int(round(cfg.mal_frac * cfg.users))
    mal_users = set(rng.choice(cfg.users, size=n_mal, replace=False).tolist()) if n_mal else set()
    # 数据级攻击:每个恶意用户对标签空间做固定置换
    label_perms: dict[int, torch.Tensor] = {}
    if cfg.attack == "label_shuffle":
        for u in mal_users:
            label_perms[u] = torch.from_numpy(rng.permutation(NUM_CLASSES[cfg.dataset])).long()
    shapes = param_shapes(cfg.dataset, cfg.arch)

    gmodel = build_model(cfg.dataset, cfg.arch)
    rows: list[dict] = []
    best_acc = 0.0
    t_start = time.time()

    for t in range(cfg.rounds):
        lr = cfg.lr * (0.1 if t >= decay[0] else 1.0) * (0.1 if t >= decay[1] else 1.0)
        chosen = sorted(rng.choice(cfg.users, size=min(cfg.sample, cfg.users), replace=False).tolist()) \
            if cfg.sample < cfg.users else list(range(cfg.users))
        base_flat = flatten_state(gmodel)

        t0 = time.time()
        local_flat: dict[int, np.ndarray] = {}
        for u in chosen:
            lmodel = build_model(cfg.dataset, cfg.arch)
            lmodel.load_state_dict(gmodel.state_dict())
            local_train(lmodel, gmodel, Xtr, ytr, parts[u], label_perms.get(u), cfg.batch,
                        alpha, lr, cfg.epochs, cfg.weight_decay, rng, mt, st)
            local_flat[u] = flatten_state(lmodel)
            del lmodel
        grads = [local_flat[u] - base_flat for u in chosen]

        mal_chosen = [i for i, u in enumerate(chosen) if u in mal_users]
        if cfg.attack == "alie" and mal_chosen:
            grads = atk.apply_alie(grads, mal_chosen, cfg.alie_z)
        elif cfg.attack == "agr" and mal_chosen:
            grads = atk.apply_agr(grads, mal_chosen, cfg.agr_lam)

        agg, diag = aggregate(cfg.agg, grads, shapes, cfg.mal_frac)
        load_flat(gmodel, base_flat + cfg.eta * agg)
        train_sec = time.time() - t0

        do_eval = (t % cfg.eval_every == 0) or (t == cfg.rounds - 1)
        acc = evaluate(gmodel, Xte, yte, mt, st) if do_eval else None
        if acc is not None:
            best_acc = max(best_acc, acc)

        row = {"round": t + 1, "lr": lr, "acc": acc, "train_sec": round(train_sec, 2)}
        if "rho" in diag:
            rho = np.asarray(diag["rho"])
            honest = [float(rho[i]) for i in range(len(chosen)) if i not in mal_chosen]
            row["rho_honest_mean"] = round(float(np.mean(honest)), 6) if honest else None
            row["rho_mal_mean"] = round(float(np.mean(rho[mal_chosen])), 8) if mal_chosen else None
            row["rho_mal_max"] = round(float(np.max(rho[mal_chosen])), 8) if mal_chosen else None
        rows.append(row)
        print(f"[{cfg.tag or cfg.agg}] round {t+1}/{cfg.rounds} lr={lr:g} "
              f"acc={acc if acc is not None else '-'} train={train_sec:.1f}s "
              + (f"rho_mal_mean={row.get('rho_mal_mean')}" if "rho_mal_mean" in row else ""),
              flush=True)

    wall = time.time() - t_start
    final_acc = next(r["acc"] for r in reversed(rows) if r["acc"] is not None)

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    slug = f"{cfg.dataset}_{cfg.agg}_{cfg.attack}_mr{cfg.mal_frac}_beta{cfg.beta}_r{cfg.rounds}_s{cfg.seed}"
    if cfg.tag:
        slug += f"_{cfg.tag}"
    import csv
    with open(cfg.outdir / f"{slug}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = dict(cfg=dataclasses.asdict(cfg) | {"decay_rounds": list(decay), "outdir": str(cfg.outdir)},
                alpha_used=alpha, mal_users=sorted(mal_users), final_acc=final_acc,
                best_acc=best_acc, wall_sec=round(wall, 1))
    with open(cfg.outdir / f"{slug}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"RESULT slug={slug} final_acc={final_acc:.4f} best_acc={best_acc:.4f} wall_sec={wall:.0f}", flush=True)
    return meta

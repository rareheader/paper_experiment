"""单次实验的命令行入口。

示例:
    # 论文全规模(GPU):CIFAR-10、120 轮、3 个本地 epoch、第 40/80 轮衰减
    python src/run_experiment.py --dataset cifar10 --agg sflbr --attack label_shuffle \
        --mal-frac 0.3 --beta 0.5 --rounds 120 --epochs 3 --decay-rounds 40 80

    # 本机 CPU 缩小规模(默认值即缩放配置,见 README)
    python src/run_experiment.py --dataset mnist --agg sflbr --attack alie --mal-frac 0.5

--agg 可选:sflbr / fedavg / median / tmean / krum
--attack 可选:none / label_shuffle / alie / agr
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import FLConfig, run_fl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="SFLBR 复现:单次联邦训练实验")
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10", "cifar100"])
    ap.add_argument("--arch", default=None, choices=[None, "resnet50"])
    ap.add_argument("--agg", default="sflbr", choices=["sflbr", "fedavg", "median", "tmean", "krum"])
    ap.add_argument("--attack", default="none", choices=["none", "label_shuffle", "alie", "agr"])
    ap.add_argument("--mal-frac", type=float, default=0.0)
    ap.add_argument("--alie-z", type=float, default=1.5)
    ap.add_argument("--agr-lam", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--decay-rounds", type=int, nargs=2, default=None)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch 线程数;0 = 默认")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = FLConfig(
        dataset=args.dataset, arch=args.arch, agg=args.agg, attack=args.attack,
        mal_frac=args.mal_frac, alie_z=args.alie_z, agr_lam=args.agr_lam, beta=args.beta,
        users=args.users, sample=args.sample, rounds=args.rounds, epochs=args.epochs,
        batch=args.batch, lr=args.lr, eta=args.eta, alpha=args.alpha,
        weight_decay=args.weight_decay,
        decay_rounds=tuple(args.decay_rounds) if args.decay_rounds else None,
        eval_every=args.eval_every, seed=args.seed, threads=args.threads,
        outdir=Path(args.outdir), tag=args.tag,
    )
    run_fl(cfg)


if __name__ == "__main__":
    main()

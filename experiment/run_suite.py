"""一键跑完本机 CPU 缩放版实验套件(与 README 里记录的结果完全对应)。

用法:
    python run_suite.py            # 跑完整套件(约 1.5~2 小时,依机器而定)
    python run_suite.py --only mnist   # 只跑 MNIST 部分
    python run_suite.py --only cifar
    python run_suite.py --only dlg

套件规模是论文实验的缩小版(轮数缩短、只跑小卷积网络),原因与偏差清单见 README。
论文全规模(GPU)配置的运行命令见 README 第 5 节。
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# 公共参数:10 个用户全员参加、本地 3 个 epoch、批 32、SGD lr=0.1(按轮数等比衰减)
COMMON = ["--users", "10", "--sample", "10", "--epochs", "3", "--batch", "32",
          "--lr", "0.1", "--beta", "0.5", "--threads", "2"]

RUNS = [
    # ---------- MNIST:干净场景(β=0.5 Non-IID)----------
    dict(cmd=["--dataset", "mnist", "--agg", "sflbr",  "--attack", "none", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "fedavg", "--attack", "none", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "sflbr",  "--attack", "none", "--rounds", "15", "--beta", "100"]),
    # ---------- MNIST:标签打乱,恶意比例 30% ----------
    dict(cmd=["--dataset", "mnist", "--agg", "sflbr",  "--attack", "label_shuffle", "--mal-frac", "0.3", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "fedavg", "--attack", "label_shuffle", "--mal-frac", "0.3", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "median", "--attack", "label_shuffle", "--mal-frac", "0.3", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "tmean",  "--attack", "label_shuffle", "--mal-frac", "0.3", "--rounds", "15"]),
    # ---------- MNIST:ALIE,恶意比例 30%(中位数保证域内)与 50%(理论边界)----------
    dict(cmd=["--dataset", "mnist", "--agg", "sflbr",  "--attack", "alie", "--mal-frac", "0.3", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "fedavg", "--attack", "alie", "--mal-frac", "0.3", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "median", "--attack", "alie", "--mal-frac", "0.3", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "tmean",  "--attack", "alie", "--mal-frac", "0.3", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "sflbr",  "--attack", "alie", "--mal-frac", "0.5", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "fedavg", "--attack", "alie", "--mal-frac", "0.5", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "median", "--attack", "alie", "--mal-frac", "0.5", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "tmean",  "--attack", "alie", "--mal-frac", "0.5", "--rounds", "15"]),
    # ---------- MNIST:AGR,恶意比例 30% ----------
    dict(cmd=["--dataset", "mnist", "--agg", "sflbr",  "--attack", "agr", "--mal-frac", "0.3", "--rounds", "15"]),
    dict(cmd=["--dataset", "mnist", "--agg", "fedavg", "--attack", "agr", "--mal-frac", "0.3", "--rounds", "15"]),
    # ---------- CIFAR-10:标签打乱 30% / ALIE 30% 与 50% ----------
    dict(cmd=["--dataset", "cifar10", "--agg", "sflbr",  "--attack", "label_shuffle", "--mal-frac", "0.3", "--rounds", "12"]),
    dict(cmd=["--dataset", "cifar10", "--agg", "fedavg", "--attack", "label_shuffle", "--mal-frac", "0.3", "--rounds", "12"]),
    dict(cmd=["--dataset", "cifar10", "--agg", "median", "--attack", "label_shuffle", "--mal-frac", "0.3", "--rounds", "12"]),
    dict(cmd=["--dataset", "cifar10", "--agg", "sflbr",  "--attack", "alie", "--mal-frac", "0.3", "--rounds", "12"]),
    dict(cmd=["--dataset", "cifar10", "--agg", "fedavg", "--attack", "alie", "--mal-frac", "0.3", "--rounds", "12"]),
    dict(cmd=["--dataset", "cifar10", "--agg", "sflbr",  "--attack", "alie", "--mal-frac", "0.5", "--rounds", "12"]),
    dict(cmd=["--dataset", "cifar10", "--agg", "fedavg", "--attack", "alie", "--mal-frac", "0.5", "--rounds", "12"]),
]


def run_part(which: str):
    log_path = ROOT / "results" / "suite_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        if which in ("mnist", "all"):
            for i, r in enumerate(RUNS):
                if not r["cmd"][1].startswith("mnist"):
                    continue
                line = f"===== [{i+1}/{len(RUNS)}] {' '.join(r['cmd'])} ====="
                print(line, flush=True)
                log.write(line + "\n")
                subprocess.run([PY, str(ROOT / "src" / "run_experiment.py"), *COMMON, *r["cmd"]],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if which in ("cifar", "all"):
            for i, r in enumerate(RUNS):
                if not r["cmd"][1].startswith("cifar"):
                    continue
                line = f"===== [{i+1}/{len(RUNS)}] {' '.join(r['cmd'])} ====="
                print(line, flush=True)
                log.write(line + "\n")
                subprocess.run([PY, str(ROOT / "src" / "run_experiment.py"), *COMMON, *r["cmd"]],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if which in ("dlg", "all"):
            line = "===== DLG 梯度重建对比 ====="
            print(line, flush=True)
            log.write(line + "\n")
            subprocess.run([PY, str(ROOT / "src" / "dlg.py")],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "mnist", "cifar", "dlg"])
    args = ap.parse_args()
    t0 = time.time()
    run_part(args.only)
    print(f"SUITE DONE in {(time.time()-t0)/60:.1f} min", flush=True)

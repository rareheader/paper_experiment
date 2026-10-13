"""汇总 results/ 下的实验输出:生成 summary.csv 和准确率曲线/柱状图。

用法:python src/make_report.py
输出:
    results/summary.csv                 每次运行的最终/最佳准确率
    results/plot_<dataset>_<attack>.png 各攻击场景下的准确率-轮数曲线
    results/plot_final_acc.png          各场景最终准确率柱状图
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SLUG_RE = re.compile(
    r"^(?P<dataset>mnist|cifar10|cifar100)_(?P<agg>sflbr|fedavg|median|tmean|krum)_"
    r"(?P<attack>none|label_shuffle|alie|agr)_"
    r"mr(?P<mr>[\d.]+)_beta(?P<beta>[\d.]+)_r(?P<rounds>\d+)_s(?P<seed>\d+)(?:_(?P<tag>\w+))?$"
)

AGG_LABEL = {"sflbr": "SFLBR", "fedavg": "FedAvg", "median": "坐标中值",
             "tmean": "截尾均值", "krum": "Krum"}
ATTACK_LABEL = {"none": "无攻击", "label_shuffle": "标签打乱", "alie": "ALIE", "agr": "AGR"}


def load_runs():
    runs = []
    for meta_path in sorted(RESULTS.glob("*.meta.json")):
        m = SLUG_RE.match(meta_path.name[:-10])
        if not m:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        csv_path = meta_path.with_suffix("").with_suffix(".csv")
        rounds, accs = [], []
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["acc"]:
                    rounds.append(int(row["round"]))
                    accs.append(float(row["acc"]))
        runs.append(dict(slug=meta_path.name[:-10], group=m.groupdict(), meta=meta,
                         rounds=rounds, accs=accs))
    return runs


def main():
    runs = load_runs()
    if not runs:
        print("results/ 下没有可汇总的运行结果")
        return

    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "agg", "attack", "mal_frac", "beta", "rounds", "seed",
                    "final_acc", "best_acc", "wall_sec", "slug"])
        for r in sorted(runs, key=lambda x: (x["group"]["dataset"], x["group"]["attack"],
                                             float(x["group"]["mr"]), x["group"]["agg"])):
            g = r["group"]
            w.writerow([g["dataset"], g["agg"], g["attack"], g["mr"], g["beta"],
                        g["rounds"], g["seed"], f"{r['meta']['final_acc']:.4f}",
                        f"{r['meta']['best_acc']:.4f}", r["meta"]["wall_sec"], r["slug"]])

    # 按场景画曲线
    scenes = sorted({(g["dataset"], g["attack"], float(g["mr"]), float(g["beta"]))
                     for r in runs for g in [r["group"]]})
    for dataset, attack, mr, beta in scenes:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        subset = [r for r in runs
                  if r["group"]["dataset"] == dataset and r["group"]["attack"] == attack
                  and float(r["group"]["mr"]) == mr and float(r["group"]["beta"]) == beta]
        for r in sorted(subset, key=lambda x: x["group"]["agg"]):
            g = r["group"]
            ax.plot(r["rounds"], [a * 100 for a in r["accs"]], marker="o", ms=3,
                    label=f"{AGG_LABEL.get(g['agg'], g['agg'])}")
        ax.set_xlabel("通信轮数")
        ax.set_ylabel("全局模型测试准确率 (%)")
        title = f"{dataset.upper()}  {ATTACK_LABEL.get(attack, attack)}  恶意比例 {mr:.0%}  β={beta:g}"
        ax.set_title(title + "(本机 CPU 缩小复现,15 轮内)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = RESULTS / f"plot_{dataset}_{attack}_mr{mr:g}_beta{beta:g}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("曲线图:", out.name)

    # 最终准确率柱状图(MNIST + CIFAR 分面)
    scenes_ma = sorted({(g["dataset"], g["attack"], float(g["mr"]), float(g["beta"]))
                        for r in runs for g in [r["group"]]})
    n = len(scenes_ma)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
        for ax, (dataset, attack, mr, beta) in zip(axes[0], scenes_ma):
            subset = [r for r in runs
                      if r["group"]["dataset"] == dataset and r["group"]["attack"] == attack
                      and float(r["group"]["mr"]) == mr and float(r["group"]["beta"]) == beta]
            subset = sorted(subset, key=lambda x: x["meta"]["final_acc"])
            names = [AGG_LABEL.get(r["group"]["agg"], r["group"]["agg"]) for r in subset]
            vals = [r["meta"]["final_acc"] * 100 for r in subset]
            colors = ["#d62728" if r["group"]["agg"] == "sflbr" else "#7f8fa6" for r in subset]
            ax.barh(names, vals, color=colors)
            for i, v in enumerate(vals):
                ax.text(v + 0.3, i, f"{v:.2f}", va="center", fontsize=9)
            ax.set_xlim(0, 105)
            ax.set_title(f"{dataset.upper()} {ATTACK_LABEL.get(attack, attack)} mr={mr:g} β={beta:g}",
                         fontsize=10)
            ax.set_xlabel("最终测试准确率 (%)")
        fig.tight_layout()
        out = RESULTS / "plot_final_acc.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("柱状图:", out.name)

    print("汇总:", RESULTS / "summary.csv")


if __name__ == "__main__":
    main()

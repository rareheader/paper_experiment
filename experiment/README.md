# SFLBR 复现实验说明

论文:Jinbo Wang, Ruijin Wang 等, **A Byzantine-Robust Secure Federated Learning Scheme in Heterogeneous Data**(SFLBR), IEEE TIFS, vol. 21, pp. 7287–7300, 2026. DOI: 10.1109/TIFS.2026.3714122

本文件夹是论文实验部分的本机复现:**代码完整、实验在本机 CPU 上按比例缩小跑完**,结果在 `results/`。论文全规模(GPU)的运行命令也一并给出,可直接搬去有卡的机器执行。

---

## 1. 本机环境配置(实测)

以下数据由 `src/collect_env.py` 自动采集(可随时重跑生成 `environment_report.txt`):

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11(内核 build 10.0.26200) |
| CPU | 13th Gen Intel Core i7-13620H(10 核 16 线程,x86_64) |
| 内存 | 15.7 GB |
| GPU | **无 NVIDIA GPU**(无 nvidia-smi,torch.cuda.is_available = False) |
| Python | 3.11.9(64 位) |
| PyTorch | 2.10.0+cpu(CPU 版,无 CUDA) |
| torchvision | 0.25.0+cpu |
| numpy | 2.3.5 |
| matplotlib | 3.10.8 |
| pyarrow | 26.0.0(仅 CIFAR-10 数据转换用) |
| torch 线程数 | 实验统一 `--threads 2`(微测 1/2/4/8/10 线程后选定,2 线程对小卷积核最快 ≈3.1 ms/批) |

**GPU 说明(重要)**:论文全部实验跑在 NVIDIA GeForce RTX 3090 上。本机没有 N 卡,所以:

- 训练实验按比例缩小后在 CPU 上**真实跑完**(见第 4 节),不是只给步骤;
- CIFAR-100 + ResNet-50 的 120 轮全规模实验**没有跑**——CPU 上估算需要以天计,且与论文数字不可比;
- 全规模命令见第 5 节,拿到有 GPU 的机器即可原样执行。

## 2. 复现范围与原理说明

**精度复现不需要实现密码学。** SFLBR 的精度由三件事决定:坐标中位数锚、逐层余弦信任分、本地主动正则;双掩码加密只是让这三件事能在密文上算,掩码在聚合中精确抵消(阅读笔记第 11 节已用三用户闭环自检验证,最大误差 1.1e-16,见 `../复现/numeric_result.txt`)。因此训练实验在明文域实现聚合规则,精度结果与密文域一致。

**各模块与论文/笔记的对应:**

| 代码文件 | 内容 | 对应 |
|---|---|---|
| `src/models.py` | MNISTNet / CIFAR10Net(按正文尺寸反推)、ResNet-50 CIFAR 变体 | 笔记 14.3 |
| `src/data.py` | MNIST/CIFAR 加载、Dirichlet Non-IID 划分、标签打乱 | 论文实验设置、笔记 14.2 |
| `src/attacks.py` | ALIE(μ+zσ, z=1.5)、AGR(中位数锚反方向×λ) | 论文实验 D 节、笔记 14.4 |
| `src/aggregation.py` | FedAvg / Krum / 坐标中值 / 截尾均值 / SFLBR(scoreFunc 分段信任分) | 论文式(1)(2)、笔记 8–10 节 |
| `src/train.py` | 本地损失 CE + α·‖w_i−w‖₂(**L2 距离本身,非平方**)、全局更新 w←w+η·Σρᵢgᵢ | 论文 Fig.3 / 算法 1、笔记 6、10 节 |
| `src/dlg.py` | DLG 梯度重建:明文 vs 加法掩码 | 论文图 9 |
| `src/make_report.py` | 汇总 CSV + 曲线/柱状图 | — |

## 3. 快速开始

```bash
cd experiment
python src/collect_env.py          # 1. 生成环境报告
python run_suite.py --only all     # 2. 跑完整套件(本机约 2~2.5 小时)
python src/make_report.py          # 3. 生成 summary.csv 和图
python src/dlg.py                  # 已含在套件里,可单独重跑(约 8 分钟)
```

单次实验(所有超参可调):

```bash
python src/run_experiment.py --dataset mnist --agg sflbr --attack alie \
    --mal-frac 0.5 --beta 0.5 --rounds 15 --users 10 --sample 10 \
    --epochs 3 --batch 32 --lr 0.1 --threads 2
```

`--agg` 可选 `sflbr / fedavg / median / tmean / krum`;`--attack` 可选 `none / label_shuffle / alie / agr`。

## 4. 本机实际跑的实验与结果

共 **24 次训练运行 + 1 次 DLG 重建**,每次运行一个逐轮 CSV(`results/*.csv`)加元数据(`*.meta.json`);汇总在 `results/summary.csv`,曲线图和柱状图在 `results/plot_*.png`,完整日志在 `results/suite_log.txt`。

**最终测试准确率(全局模型,单种子 seed=0):**

| 场景 | SFLBR | FedAvg | 坐标中值 | 截尾均值 |
|---|---|---|---|---|
| MNIST 干净 β=0.5 | **98.87** | 98.87 | — | — |
| MNIST 干净 IID β=100 | **99.05** | — | — | — |
| MNIST 标签打乱 30% | 98.12 | 95.91 | 98.41 | 98.48 |
| MNIST AGR 30% | **97.72** | 96.32 | — | — |
| MNIST ALIE 30% | 91.11 | 97.32 | 59.92 | 10.32 |
| MNIST ALIE 50%(理论边界) | 9.80 | 9.82 | 9.80 | 9.80 |
| CIFAR-10 标签打乱 30% | **50.56** | 49.38 | 50.06 | — |
| CIFAR-10 ALIE 30% | 41.76 | 48.49 | — | — |
| CIFAR-10 ALIE 50%(理论边界) | 10.00 | 10.00 | — | — |

**DLG 梯度重建(论文图 9 的定性复现,`results/dlg_comparison.png`):**

| 观测对象 | 梯度匹配损失 | 重建结果 |
|---|---|---|
| 明文梯度 g | 3.88e-08 | **逐像素精确还原数字**(MSE = 0.0000) |
| SFLBR 加法掩码后 [g]₁ = g + PRG(sk₁) | 4.38e+05 | 纯噪声,与原图无关 |

攻击者即使已经通过 iDLG 从偏置梯度解析推断出标签(更强设定),掩码后的匹配损失仍高出 13 个数量级,无法重建。

**七条关键发现(对应论文的定性结论):**

1. **干净场景正则不伤精度**——SFLBR 与 FedAvg 在 β=0.5 干净场景完全打平(98.87%),IID 下 99.05%。对应论文"恶意比例为 0 时没有把干净训练做差"。
2. **标签打乱 30%:SFLBR 比 FedAvg 高 2.2 个点**(98.12% vs 95.91%),中值/截尾均值同样稳。方向与论文表 II 一致(差距量级也接近论文 MNIST 10%~30% 档的 2.48 点)。
3. **AGR 30%:SFLBR 97.72% > FedAvg 96.32%**——反向梯度被信任分压掉,与论文"恶意权重被压到近零"的机制一致(训练日志里 SFLBR 的 rho_mal_mean ≈ 0.06,远低于均匀份额 0.10)。
4. **ALIE 30% 的细微差别(要诚实记录)**:SFLBR(91.11%)大幅好于鲁棒基线中值(59.92%)和截尾均值(10.32%),但**无防御的 FedAvg(97.32%)反而最高**——ALIE 向量与诚实共识同向,3 份 ALIE 在均值里被 7 份诚实稀释;而 SFLBR 的中位数锚在训练前期被 ALIE 拉偏(见曲线第 5 轮的深谷),信任分恢复后准确率回升。论文表 III 在 CIFAR-10 上 ALIE 10% 时 SFLBR 也只有 52.29%,同样显示 ALIE 对它伤害不小;论文的优势是相对鲁棒基线而言。
5. **ALIE 50%:全部方法一起崩到随机水平**(MNIST 与 CIFAR-10 都如此)。50% 正是中位数保证的边界:5 份相同的 ALIE 向量恰好占据中位数边界,把锚拉向攻击方向(日志可见 rho_mal_mean 0.116 > rho_honest_mean 0.084)。论文表 III 报告 CIFAR-10@50% 有 36.12% 的幸存,我们的缩小复现没有复现出这个余量——可能与论文攻击统计量的模糊表述("加在恶意梯度的均值上")、120 轮 ResNet-50 的规模差异有关,这也与笔记第 13 节对 50% 边界的怀疑一致。
6. **截尾均值在 ALIE 下不可靠**:μ+1.5σ 在多数坐标上不是极端值,裁不掉(tmean ALIE 30% 崩到 10.3%)——印证"比较类统计量对同向攻击天然脆弱"。
7. **加密确实挡住 DLG**:明文梯度 300 步内逐像素精确重建;加法掩码后,即使攻击者知道标签,匹配损失停在 4.4e5(差 13 个数量级)。对应论文图 9。

图:`results/plot_*.png`(各攻击场景准确率-轮数曲线、最终准确率柱状图);DLG 对比图 `results/dlg_comparison.png`。

## 5. 论文全规模复现(GPU 机器)

论文设定:120 轮,每轮抽 10 用户,本地 3 epoch,批 32,SGD 初始 lr 0.1 在第 40/80 轮各乘 0.1,权重衰减按原文,CIFAR-100 用 ResNet-50,数据划分 Dirichlet β ∈ {100, 0.5, 0.05},恶意比例 MR ∈ {0, 0.1, ..., 0.5}。对应命令:

```bash
# CIFAR-10 + 标签打乱 + MR=0.3(对应论文表 II 的一格)
python src/run_experiment.py --dataset cifar10 --agg sflbr --attack label_shuffle \
    --mal-frac 0.3 --beta 0.5 --rounds 120 --epochs 3 --batch 32 \
    --decay-rounds 40 80 --weight-decay 1e-5

# CIFAR-100 + ResNet-50 + ALIE + MR=0.5
python src/run_experiment.py --dataset cifar100 --arch resnet50 --agg sflbr --attack alie \
    --mal-frac 0.5 --beta 0.05 --rounds 120 --epochs 3 --decay-rounds 40 80

# 换聚合基线只需换 --agg:fedavg / median / tmean / krum
# 换攻击只需换 --attack:label_shuffle / alie / agr
```

注意事项(与笔记 14 节一致):

- ResNet-50 必须在 GPU 上跑(本机 CPU 跑会以天计);
- MNIST/CIFAR-10 的第一层卷积通道数原文没写死,本仓库按 320/400 的全连接输入反推取了常见配法;
- 验收对着正文的**差值**而不是绝对值:CIFAR-10 β=0.05 标签打乱 50% 时 SFLBR 应比 FedAvg 高约 10 个点;AGR 50% 时所有方法一起掉到随机猜测附近;
- 每格建议至少 3 个种子(`--seed 0/1/2`)取均值。

## 6. 与论文实验的偏差清单(本机缩小复现)

| 项 | 论文 | 本机 | 原因 |
|---|---|---|---|
| 硬件 | RTX 3090 | i7-13620H CPU(2 线程) | 本机无 N 卡 |
| 轮数 | 120 | MNIST 15 / CIFAR-10 12 | CPU 时间预算;论文的 lr 衰减点(1/3、2/3)等比换算 |
| 数据集 | MNIST、CIFAR-10、CIFAR-100 | MNIST、CIFAR-10 | CIFAR-100 + ResNet-50 只适合 GPU |
| 基线 | FedAvg、Krum、Median、T-mean、ShieldFL、PEFL、RFLPA | FedAvg、Krum、Median、T-mean | ShieldFL/PEFL/RFLPA 的贡献在隐私协议而非聚合规则,明文域只剩聚合规则可比;Krum 已实现,未进默认套件 |
| 攻击 | 标签打乱、ALIE、AGR、DLG | 全部四种 | — |
| 用户数 | 主实验人数未写明(每轮抽 10);D 节 50 抽 10 | 10 人全员参加 | 按笔记 14.2 的读法取最小可解释配置 |
| 密码学 | 双掩码 + 双服务器协议 | 明文域聚合(掩码代数上精确抵消) | 精度不受影响,协议代数另见笔记第 11 节 |
| 动量 | 原文写 1e-4 | 0(≈ 无动量,同一数量级) | 原文数值异常小,复现按笔记 6 节追问的处理 |
| 种子 | 未说明 | 单种子 seed=0 | CPU 预算;结论看多场景一致性而非单点 |

## 7. 文件清单

```
experiment/
├── README.md                          本文件
├── environment_report.txt             自动采集的环境快照
├── run_suite.py                       一键复现套件
├── data/                              MNIST(torchvision 下载)、CIFAR-10(.pt 缓存)
│   └── _hf/                           CIFAR-10 的 parquet 镜像源文件(可删)
├── src/
│   ├── models.py  data.py  attacks.py  aggregation.py  train.py
│   ├── run_experiment.py              单次实验 CLI
│   ├── prepare_cifar10_from_parquet.py CIFAR-10 数据源替代(默认源太慢)
│   ├── dlg.py                         DLG 重建对比
│   ├── make_report.py                 汇总出表出图
│   └── collect_env.py                 环境快照
└── results/                           每次运行的 CSV/meta、summary.csv、图、suite_log.txt
```

# FedTracker-Pro: 增强型联邦学习模型保护框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-178%20passing-brightgreen.svg)](https://github.com/yourusername/FedTracker-Pro)

## 📋 项目概述

FedTracker-Pro 是一个融合最新技术的增强型联邦学习模型保护框架，旨在解决联邦学习模型面临的知识产权保护挑战：

- **自适应保护**: 基于客户端容忍度动态调整嵌入强度 (FedAWM)
- **密码学安全**: 非对称数字签名实现确定性验证 (Fed-PK-Judge)
- **多层防御**: 统计 + 密码学 + 行为的三层验证体系
- **鲁棒增强**: 遗忘增强定位 + 频谱冗余提升攻击抵抗能力

## 🎯 核心创新

| 创新点 | 描述 | 技术来源 |
|--------|------|----------|
| 自适应指纹分配 | 基于客户端容忍度动态调整嵌入强度 | FedAWM |
| 密码学验证层 | 非对称数字签名实现确定性验证 | Fed-PK-Judge |
| 遗忘增强定位 | 利用机器遗忘识别稳定参数子空间 | Unlearning-Guided |
| 三层验证体系 | 统计 + 密码学 + 行为验证的纵深防御 | 本研究融合 |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+（推荐，用于 GPU 加速）

### 安装

```bash
git clone https://github.com/yourusername/FedTracker-Pro.git
cd FedTracker-Pro

# 方式一：使用安装脚本（推荐，含开发依赖）
bash scripts/setup.sh

# 方式二：手动安装
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 验证安装

```bash
# 语法检查
make lint

# 运行全部单元测试（175 个测试）
pytest tests/unit/ -v

# 运行集成测试（3 个测试）
pytest tests/integration/ -v

# 运行全部测试
pytest tests/ -v

# 覆盖率报告
pytest tests/ --cov=src --cov=experiments --cov-report=term
```

---

## 🧪 实验运行指南

### 配置文件说明

项目使用 YAML 配置文件控制所有实验参数。配置通过 `Config` 类加载，支持联邦训练、防御模块、攻击评估等全部参数。

| 配置文件 | 路径 | 用途 |
|----------|------|------|
| 完整默认配置 | `configs/default.yaml` | 全参数默认值，包含联邦学习 / 水印 / 指纹 / 密码学 / 遗忘增强等所有字段 |
| 日志配置 | `configs/logging.yaml` | Python logging dictConfig |
| CIFAR-10 + ResNet-18 | `experiments/configs/cifar10_resnet18.yaml` | CIFAR-10 基线实验 |
| CIFAR-100 + VGG-16 | `experiments/configs/cifar100_vgg16.yaml` | CIFAR-100 消融实验 |
| 实验默认配置 | `experiments/configs/default.yaml` | 实验运行器配置 |

#### 配置文件关键字段

```yaml
# 联邦学习参数
federated:
  num_clients: 50          # 客户端总数
  client_fraction: 0.1     # 每轮采样比例
  global_rounds: 200       # 全局通信轮数
  local_epochs: 5          # 客户端本地训练 epoch 数
  local_batch_size: 64     # 本地 batch size
  local_lr: 0.01           # 本地学习率
  global_lr: 1.0           # 全局学习率（FedAvg 聚合系数）

# 数据参数
data:
  dataset: "cifar10"       # 数据集：cifar10 / cifar100 / mnist
  iid: false               # 是否 IID 划分
  alpha: 0.5               # Dirichlet 非 IID 参数（越小越异构）
  num_shards: 200          # Shard 划分碎片数

# 模型参数
model:
  name: "resnet18"         # 模型：resnet18 / resnet34 / vgg11 / vgg16 / mobilenetv2
  num_classes: 10

# 防御模块开关（均可独立启用 / 禁用）
watermark:   { enabled: true, ... }    # CL 水印
fingerprint: { enabled: true, ... }    # 参数指纹
adaptive_allocation: { enabled: true, ... }  # 自适应分配
crypto:      { enabled: true, ... }    # 密码学验证
unlearning:  { enabled: true, ... }    # 遗忘增强

# 验证阈值
verification:
  level1_threshold: 0.75   # 指纹相似度阈值（第一层）
  level3_threshold: 0.9    # 水印验证阈值（第三层）
```

### 实验 1：基线对比实验

**目的**：对比 FedTracker-Pro 全量防御与无保护基线，在 3 种基础攻击（微调 / 剪枝 / 量化）下的所有权存活率。

```python
"""运行基线对比实验。"""
import torch
from src.core.config import Config
from src.core.fed_tracker_pro import FedTrackerPro
from src.models import ResNet18
from experiments.exp_baseline import build_default_attacks
from experiments.utils import save_results, create_experiment_dir

# 1. 加载配置
config = Config("configs/default.yaml")

# 2. 初始化框架（自动创建 50 个客户端、聚合器、防御模块）
model = ResNet18(num_classes=config.model.num_classes)
framework = FedTrackerPro(config)
framework.initialize(model)

# 3. 联邦训练
framework.train(num_rounds=config.federated.global_rounds)

# 4. 构建基线攻击集合
device = "cuda" if torch.cuda.is_available() else "cpu"
attacks = build_default_attacks(device=device)
# attacks 包含: FineTuningAttack, PruningAttack, QuantizationAttack

# 5. 评估攻击鲁棒性
test_loader = framework.data_manager.get_test_loader()
results = framework.evaluate_attack_robustness(attacks, test_loader)
print(results)
# 示例输出: {'fine_tuning': 1.0, 'pruning': 1.0, 'quantization': 1.0}

# 6. 保存结果
exp_dir = create_experiment_dir()
save_results(results, exp_dir, "baseline_results.json")
```

**快捷脚本**：

```bash
# 使用默认配置
bash scripts/train.sh

# 指定配置文件
bash scripts/train.sh experiments/configs/cifar10_resnet18.yaml

# 或使用 Makefile
make run-baseline
```

### 实验 2：消融实验

**目的**：逐一禁用防御模块，评估各组件对最终所有权验证的贡献。分为 6 组：baseline / watermark_only / fingerprint_only / adaptive / crypto / full。

```python
"""运行消融实验。"""
from src.core.config import Config
from src.core.fed_tracker_pro import FedTrackerPro
from src.models import ResNet18
from experiments.exp_ablation import get_ablation_groups
from experiments.exp_robustness import build_robustness_attacks
from experiments.utils import save_results, create_experiment_dir

groups = get_ablation_groups()
# groups = {
#     "baseline":         {watermark: F, fingerprint: F, adaptive: F, crypto: F, unlearning: F},
#     "watermark_only":   {watermark: T, fingerprint: F, adaptive: F, crypto: F, unlearning: F},
#     "fingerprint_only": {watermark: F, fingerprint: T, adaptive: F, crypto: F, unlearning: F},
#     "adaptive":         {watermark: T, fingerprint: T, adaptive: T, crypto: F, unlearning: F},
#     "crypto":           {watermark: T, fingerprint: T, adaptive: T, crypto: T, unlearning: F},
#     "full":             {watermark: T, fingerprint: T, adaptive: T, crypto: T, unlearning: T},
# }

all_results = {}
for group_name, flags in groups.items():
    # 根据消融组动态修改配置
    config = Config("configs/default.yaml")
    config.watermark.enabled = flags["watermark"]
    config.fingerprint.enabled = flags["fingerprint"]
    config.adaptive.enabled = flags["adaptive"]
    config.crypto.enabled = flags["crypto"]
    config.unlearning.enabled = flags["unlearning"]

    # 初始化并训练
    model = ResNet18(num_classes=config.model.num_classes)
    framework = FedTrackerPro(config)
    framework.initialize(model)
    framework.train(num_rounds=config.federated.global_rounds)

    # 全部 6 种攻击评估
    attacks = build_robustness_attacks(device=framework.device)
    test_loader = framework.data_manager.get_test_loader()
    robustness = framework.evaluate_attack_robustness(attacks, test_loader)
    all_results[group_name] = robustness

# 保存完整结果
exp_dir = create_experiment_dir()
save_results(all_results, exp_dir, "ablation_results.json")
```

**快捷脚本**：

```bash
bash scripts/evaluate.sh
# 或
make run-ablation
```

### 实验 3：攻击鲁棒性评估

**目的**：在全部 6 种攻击下评估 FedTracker-Pro 的所有权验证存活率。

```python
"""运行全攻击鲁棒性评估。"""
import torch
from src.core.config import Config
from src.core.fed_tracker_pro import FedTrackerPro
from src.models import ResNet18
from experiments.exp_robustness import build_robustness_attacks
from experiments.utils import save_results, create_experiment_dir

config = Config("configs/default.yaml")
model = ResNet18(num_classes=config.model.num_classes)
framework = FedTrackerPro(config)
framework.initialize(model)
framework.train(num_rounds=config.federated.global_rounds)

# 6 种攻击：微调、剪枝、量化、覆盖、模糊、模型提取
attacks = build_robustness_attacks(device=framework.device)
test_loader = framework.data_manager.get_test_loader()
results = framework.evaluate_attack_robustness(attacks, test_loader)
print(results)
# 示例输出:
# {'fine_tuning': 1.0, 'pruning': 1.0, 'quantization': 1.0,
#  'overwriting': 1.0, 'ambiguity': 1.0, 'model_extraction': 1.0}

exp_dir = create_experiment_dir()
save_results(results, exp_dir, "robustness_results.json")
```

### 实验 4：可扩展性实验

**目的**：测试框架在不同客户端规模（10 ~ 100）下的性能表现。

```python
"""运行可扩展性实验。"""
from src.core.config import Config
from src.core.fed_tracker_pro import FedTrackerPro
from src.models import ResNet18
from experiments.exp_scalability import generate_client_scenarios
from experiments.utils import save_results, create_experiment_dir

scenarios = generate_client_scenarios(min_clients=10, max_clients=100, step=10)
# scenarios = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

all_results = {}
for num_clients in scenarios:
    config = Config("configs/default.yaml")
    config.federated.num_clients = num_clients

    model = ResNet18(num_classes=config.model.num_classes)
    framework = FedTrackerPro(config)
    framework.initialize(model)
    framework.train(num_rounds=config.federated.global_rounds)

    test_loader = framework.data_manager.get_test_loader()
    metrics = framework.server.evaluate(test_loader)
    all_results[f"clients_{num_clients}"] = {
        "accuracy": metrics.get("accuracy", 0.0),
        "loss": metrics.get("loss", float("inf")),
    }

exp_dir = create_experiment_dir()
save_results(all_results, exp_dir, "scalability_results.json")
```

### 实验 5：所有权验证

**目的**：在训练完成后，对可疑模型执行三层所有权验证。

```python
"""运行三层所有权验证。"""
import torch
from src.core.config import Config
from src.core.fed_tracker_pro import FedTrackerPro
from src.models import ResNet18

config = Config("configs/default.yaml")
model = ResNet18(num_classes=config.model.num_classes)
framework = FedTrackerPro(config)
framework.initialize(model)
framework.train(num_rounds=config.federated.global_rounds)

# 对全局模型进行验证
suspicious_model = framework.global_model
is_owner, client_id, confidence = framework.verify_ownership(suspicious_model)
print(f"所有权: {is_owner}, 客户端ID: {client_id}, 置信度: {confidence:.4f}")
# 三层验证逻辑:
#   Level 1 — 指纹相似度 >= level1_threshold (0.75)
#   Level 2 — 密码学签名验证通过
#   Level 3 — 水印验证准确率 >= level3_threshold (0.9)
# 任意层失败即返回 False

# 指定候选客户端范围
is_owner, client_id, confidence = framework.verify_ownership(
    suspicious_model, candidate_clients=[0, 1, 2]
)
```

### 跨数据集实验

替换配置文件即可切换数据集和模型：

```bash
# CIFAR-10 + ResNet-18（基线）
python -c "
from src.core.config import Config
from src.core.fed_tracker_pro import FedTrackerPro
from src.models import ResNet18

config = Config('experiments/configs/cifar10_resnet18.yaml')
model = ResNet18(num_classes=10)
framework = FedTrackerPro(config)
framework.initialize(model)
framework.train()
"

# CIFAR-100 + VGG-16（消融）
python -c "
from src.core.config import Config
from src.core.fed_tracker_pro import FedTrackerPro
from src.models import VGG16

config = Config('experiments/configs/cifar100_vgg16.yaml')
model = VGG16(num_classes=100)
framework = FedTrackerPro(config)
framework.initialize(model)
framework.train()
"
```

### 可用模型

| 模型            | 工厂函数                                        | 参数量   | 适用场景              |
| ------------- | ------------------------------------------- | ----- | ----------------- |
| ResNet-18     | `ResNet18(num_classes)`                     | ~11M  | 基线实验（CIFAR-10）    |
| ResNet-34     | `ResNet34(num_classes)`                     | ~21M  | 大模型验证             |
| VGG-11        | `VGG11(num_classes)`                        | ~133M | 传统架构对比            |
| VGG-16        | `VGG16(num_classes)`                        | ~138M | 传统架构对比（CIFAR-100） |
| MobileNetV2   | `MobileNetV2(num_classes)`                  | ~3.5M | 移动端/轻量级场景         |
| DiffusionUNet | `DiffusionUNet(in_channels, base_channels)` | ~2M   | 扩散去噪子模块（需偶数空间维度）  |

---

## 📁 项目结构

```
fedtracker-pro/
├── configs/
│   ├── default.yaml                # 全参数默认配置
│   └── logging.yaml                 # 日志配置
├── docs/
│   ├── api.md                       # API 文档
│   └── tutorial.md                  # 教程
├── experiments/
│   ├── configs/
│   │   ├── cifar10_resnet18.yaml    # CIFAR-10 实验配置
│   │   ├── cifar100_vgg16.yaml      # CIFAR-100 实验配置
│   │   └── default.yaml             # 实验默认配置
│   ├── exp_baseline.py              # 基线对比实验（3 种攻击）
│   ├── exp_ablation.py              # 消融实验（6 组配置）
│   ├── exp_robustness.py            # 鲁棒性实验（6 种攻击）
│   ├── exp_scalability.py           # 可扩展性实验（10~100 客户端）
│   └── utils.py                     # 实验工具（保存结果、创建目录、聚合指标）
├── scripts/
│   ├── setup.sh                     # 环境安装脚本
│   ├── train.sh                     # 基线训练快捷脚本
│   ├── evaluate.sh                  # 消融评估快捷脚本
│   └── visualize.sh                 # 可视化占位脚本
├── src/
│   ├── aggregation/
│   │   ├── base_aggregator.py       # 聚合器抽象基类
│   │   ├── fed_avg.py               # FedAvg 聚合算法
│   │   └── fed_prox.py              # FedProx 聚合算法
│   ├── attacks/
│   │   ├── base_attack.py           # 攻击抽象基类
│   │   ├── fine_tuning.py           # 微调攻击
│   │   ├── pruning.py               # 剪枝攻击（幅值 / 随机）
│   │   ├── quantization.py          # 量化攻击
│   │   ├── overwriting.py           # 覆盖攻击
│   │   ├── ambiguity.py             # 模糊攻击
│   │   └── model_extraction.py      # 模型提取攻击
│   ├── core/
│   │   ├── config.py                # 分模块配置系统（YAML 读写）
│   │   ├── base_client.py           # 标准客户端实现
│   │   ├── base_server.py           # 服务器基类（评估 / 检查点）
│   │   └── fed_tracker_pro.py       # 主框架类
│   ├── datasets/
│   │   ├── federated_dataset.py     # 联邦数据管理器（IID / Dirichlet / Shard）
│   │   ├── cifar.py                 # CIFAR-10/100 数据适配器
│   │   └── mnist.py                 # MNIST 数据适配器
│   ├── defense/
│   │   ├── adaptive_allocation.py   # 自适应指纹分配（FedAWM）
│   │   ├── crypto_verification.py   # 密码学验证（ECDSA 签名）
│   │   ├── multi_layer_verify.py    # 三层验证编排
│   │   ├── unlearning_guided.py     # 遗忘增强定位
│   │   ├── fingerprint/
│   │   │   ├── base_fingerprint.py  # 指纹抽象基类
│   │   │   └── param_fingerprint.py # 参数指纹（LSB 嵌入）
│   │   └── watermark/
│   │       ├── base_watermark.py     # 水印抽象基类
│   │       └── cl_watermark.py      # CL 水印（GEM 持续学习）
│   ├── models/
│   │   ├── resnet.py                # ResNet-18 / 34
│   │   ├── vgg.py                   # VGG-11 / 16
│   │   ├── mobilenet.py             # MobileNetV2
│   │   └── diffusion.py             # DiffusionUNet（扩散去噪骨干）
│   └── utils/
│       ├── logger.py                # 日志管理（文件 + 控制台）
│       ├── metric_utils.py          # 评估指标（准确率 / 损失 / FSS）
│       ├── data_utils.py            # 数据划分（IID / Dirichlet / Shard）
│       ├── crypto_utils.py          # ECDSA 签名 / 哈希 / LSB 嵌入
│       └── visualization.py         # 可视化（训练曲线 / 分布 / 鲁棒性）
├── tests/
│   ├── unit/                        # 175 个单元测试
│   └── integration/                 # 3 个集成测试
├── Makefile                         # make install / test / lint / run-baseline
├── setup.py                         # pip install -e .
├── requirements.txt                 # 运行时依赖
├── requirements-dev.txt             # 开发依赖
└── README.md
```

---

## ✅ 开发状态

| 阶段 | 模块 | 状态 | 测试 |
|------|------|------|------|
| 阶段 1 | 基础工具 (`logger`, `metric_utils`, `data_utils`, `crypto_utils`, `visualization`) | ✅ 完成 | 66 tests, 95% |
| 阶段 2 | 配置系统 (`config`) + 聚合器 (`fed_avg`, `fed_prox`) | ✅ 完成 | — |
| 阶段 3 | 数据集 (`federated_dataset`, `cifar`, `mnist`) + 模型 (`resnet`, `vgg`, `mobilenet`, `diffusion`) | ✅ 完成 | — |
| 阶段 4 | 防御模块 (`cl_watermark`, `param_fingerprint`, `adaptive_allocation`, `crypto_verification`, `unlearning_guided`, `multi_layer_verify`) | ✅ 完成 | — |
| 阶段 5 | 攻击模块 (`fine_tuning`, `pruning`, `quantization`, `overwriting`, `ambiguity`, `model_extraction`) | ✅ 完成 | — |
| 阶段 6 | 攻击鲁棒性测试增强 | ✅ 完成 | — |
| 阶段 7 | 主框架 (`FedTrackerPro`) + 客户端 / 服务器 | ✅ 完成 | — |
| 阶段 8 | 实验脚本 + 配置 + `setup.py` + `Makefile` | ✅ 完成 | — |
| 阶段 9 | 工程化（`requirements-dev` / 脚本 / 文档 / 集成测试 / diffusion 模型） | ✅ 完成 | 178 total |
| 阶段 10 | 端到端实验与论文结果复现 | ⏳ 待运行 | — |

**总测试**: 178 passing（175 unit + 3 integration）

---

## 📊 技术架构

### 三层防御架构

```
FedTracker-Pro Framework
├── 主动防御层
│   ├── 自适应水印分配 (FedAWM / CL Watermark)
│   ├── 参数指纹嵌入 (Parametric Fingerprint / LSB)
│   └── 遗忘增强定位 (Unlearning-Guided)
├── 验证检测层
│   ├── Level 1 — 统计验证 (FSS 指纹相似度 ≥ 0.75)
│   ├── Level 2 — 密码学验证 (ECDSA 数字签名)
│   └── Level 3 — 行为验证 (CL 水印准确率 ≥ 0.9)
└── 溯源追踪层
    ├── 参数指纹溯源 (client embedding 唯一性)
    ├── 自适应分配追踪 (tolerance-based α weighting)
    └── 密码学不可抵赖 (signature embedding in weights)
```

### 攻击防御对应表

| 攻击类型 | 实现类 | 防御机制 | 预期存活率 |
|----------|--------|----------|------------|
| 微调攻击 | `FineTuningAttack` | 遗忘增强 + CL 水印 | >85% |
| 剪枝攻击 | `PruningAttack` | 大权重嵌入 + 频谱冗余 | >80% |
| 量化攻击 | `QuantizationAttack` | 量化感知嵌入 + 纠错码 | >85% |
| 覆盖攻击 | `OverwritingAttack` | 多层水印 + 密码学唯一性 | >90% |
| 模糊攻击 | `AmbiguityAttack` | 密码学不可伪造性 + 时间戳链 | >95% |
| 模型提取 | `ModelExtractionAttack` | 参数指纹 + 自适应分配 | >80% |

---

## 🛠️ 常用命令

```bash
# 安装
make install                          # pip install 依赖

# 测试
make test                             # pytest tests/unit/ -v
make test-cov                         # 覆盖率报告

# 代码检查
make lint                             # 语法编译检查

# 基线实验
make run-baseline                     # python experiments/exp_baseline.py
make run-ablation                     # python experiments/exp_ablation.py

# Shell 脚本
bash scripts/setup.sh                 # 安装环境
bash scripts/train.sh                  # 基线训练（默认 configs/default.yaml）
bash scripts/train.sh experiments/configs/cifar10_resnet18.yaml  # 指定配置
bash scripts/evaluate.sh              # 消融评估（默认 configs/default.yaml）
bash scripts/visualize.sh             # 可视化提示（需手动调用 visualization.py）

# 清理
make clean                            # 清理构建产物
```

---

## 📝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'feat: Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件

## 👥 作者

Research Team

## 📚 参考文献

1. **FedTracker**: Shao et al. "FedTracker: Furnishing Ownership Verification and Traceability for Federated Learning Model." IEEE TDSC, 2025.

2. **FedUIMF**: Cao et al. "FedUIMF: Unambiguous and Imperceptible Model Fingerprinting for Secure Federated Learning." IEEE TCE, 2025.

3. **FedDiff**: Li et al. "FedDiff: Diffusion Model Driven Federated Learning for Multi-Modal and Multi-Clients." IEEE TCSVT, 2024.

4. **FedAWM**: Sun et al. "FedAWM: Adaptive watermark allocation in non-IID federated learning." KBS, 2026.

5. **Fed-PK-Judge**: Kanakri & King. "Fed-PK-Judge: Provably Secure Intellectual-Property Protection for Federated Learning." IEEE IoT-J, 2025.

6. **Unlearning-Guided**: Pan et al. "Robust Watermarking for Federated Diffusion Models with Unlearning-Enhanced Redundancy." IEEE TDSC, 2025.
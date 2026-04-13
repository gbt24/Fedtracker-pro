# FedTracker-Pro: 增强型联邦学习模型保护框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-66%20passing-brightgreen.svg)](https://github.com/yourusername/FedTracker-Pro)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/yourusername/FedTracker-Pro)

## 📋 项目概述

FedTracker-Pro是一个融合最新技术的增强型联邦学习模型保护框架，旨在解决联邦学习模型面临的知识产权保护挑战：

- ✅ **自适应保护**: 根据客户端数据特性动态调整保护强度
- ✅ **密码学安全**: 引入可证明安全的验证机制
- ✅ **多层防御**: 构建统计+密码学+行为的三层验证体系
- ✅ **鲁棒增强**: 提升对各种攻击的抵抗能力

## 🎯 核心创新

| 创新点 | 描述 | 技术来源 |
|--------|------|----------|
| 自适应指纹分配 | 基于客户端容忍度动态调整嵌入强度 | FedAWM |
| 密码学验证层 | 非对称数字签名实现确定性验证 | Fed-PK-Judge |
| 遗忘增强定位 | 利用机器遗忘识别稳定参数子空间 | Unlearning-Guided |
| 三层验证体系 | 统计+密码学+行为验证的纵深防御 | 本研究融合 |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8 (可选，用于GPU加速)

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/FedTracker-Pro.git
cd FedTracker-Pro

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行测试

```bash
# 运行所有单元测试
pytest tests/unit/ -v

# 运行测试并检查覆盖率
pytest tests/unit/ --cov=src/utils --cov-report=html

# 运行特定模块的测试
pytest tests/unit/test_logger.py -v
```

### 使用示例

```python
from src.utils.logger import get_logger
from src.utils.metric_utils import MetricsTracker
from src.utils.data_utils import partition_data_iid

# 使用日志
logger = get_logger()
logger.info("FedTracker-Pro initialized")

# 使用评估指标
tracker = MetricsTracker()
tracker.update({'accuracy': 0.95, 'loss': 0.1}, round_num=1)

# 使用数据划分
from torch.utils.data import TensorDataset
dataset = TensorDataset(data, targets)
client_indices = partition_data_iid(dataset, num_clients=10)
```

## 📁 项目结构

```
fedtracker-pro/
├── src/
│   ├── core/                      # 核心框架（开发中）
│   ├── aggregation/               # 联邦聚合（开发中）
│   ├── defense/                   # 防御模块（开发中）
│   ├── attacks/                   # 攻击实现（开发中）
│   ├── models/                    # 模型定义（开发中）
│   ├── utils/                     # ✅ 工具函数（已完成）
│   └── datasets/                  # 数据集（开发中）
├── experiments/                   # 实验脚本（开发中）
├── tests/                        # ✅ 测试代码
├── configs/                       # 配置文件（开发中）
├── scripts/                       # 工具脚本（开发中）
├── docs/                          # 文档（开发中）
├── requirements.txt               # ✅ 依赖列表
└── README.md                    # 本文件
```

## ✅ 当前状态

### 阶段1: 基础工具模块 ✅ 已完成

**完成时间**: 2026年4月13日  
**测试覆盖**: 95% (66个测试全部通过)

#### 已完成模块

| 模块 | 测试数 | 覆盖率 | 功能 |
|------|--------|--------|------|
| logger.py | 11 | 98% | 日志管理系统，支持文件和控制台输出 |
| metric_utils.py | 18 | 98% | 评估指标（准确率、损失、指纹相似度） |
| data_utils.py | 14 | 100% | 数据处理（IID/非IID划分、Dirichlet、Shard） |
| crypto_utils.py | 15 | 90% | 密码学工具（ECDSA签名、哈希、LSB嵌入） |
| visualization.py | 8 | 95% | 可视化（训练曲线、数据分布、攻击鲁棒性） |

### 阶段2: 核心模块开发 🔄 进行中

#### 待开发模块

- [ ] 配置系统 (`src/core/config.py`)
- [ ] 联邦聚合器 (`src/aggregation/fed_avg.py`, `src/aggregation/fed_prox.py`)
- [ ] 水印模块 (`src/defense/watermark/cl_watermark.py`)
- [ ] 指纹模块 (`src/defense/fingerprint/param_fingerprint.py`)
- [ ] 自适应分配 (`src/defense/adaptive_allocation.py`)
- [ ] 密码学验证 (`src/defense/crypto_verification.py`)
- [ ] 遗忘增强 (`src/defense/unlearning_guided.py`)

## 📊 技术架构

### 三层防御架构

```
FedTracker-Pro Framework
├── 主动防御层
│   ├── 自适应水印分配 (FedAWM)
│   ├── 遗忘增强定位 (Unlearning-Guided)
│   └── 模型分割保护 (Split Learning)
├── 验证检测层
│   ├── 统计验证 (FSS评分)
│   ├── 密码学验证 (数字签名)
│   └── 行为验证 (对抗样本)
└── 溯源追踪层
    ├── 参数指纹 (Parametric)
    ├── 时间戳链 (Timestamp)
    └── 数字签名 (Signature)
```

## 🛡️ 攻击防御策略

| 攻击类型 | 防御机制 | 预期存活率 |
|----------|----------|------------|
| **模糊攻击** | 密码学不可伪造性+时间戳链 | >95% |
| **微调攻击** | 遗忘增强定位+CL水印 | >85% |
| **剪枝攻击** | 大权重嵌入+频谱冗余 | >80% |
| **量化攻击** | 量化感知嵌入+纠错码 | >85% |
| **覆盖攻击** | 多层水印+密码学唯一性 | >90% |
| **后门消除** | 对抗性触发器+参数指纹 | >80% |

## 📈 实验计划

### 数据集

| 数据集 | 样本数 | 类别数 | 用途 |
|--------|--------|--------|------|
| CIFAR-10 | 60K | 10 | 基础验证 |
| CIFAR-100 | 60K | 100 | 扩展验证 |
| Tiny-ImageNet | 100K | 200 | 大规模验证 |

### 模型架构

- ResNet-18 (11M参数) - 基础架构
- VGG-16 (138M参数) - 传统架构
- MobileNetV2 (3.5M) - 移动端

## 📝 贡献指南

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件

## 👥 作者

Research Team

## 📚 参考文献

1. **FedTracker**: Shao et al. "FedTracker: Furnishing Ownership Verification and Traceability for Federated Learning Model." IEEE TDSC, 2025.

2. **FedUIMF**: Cao et al. "FedUIMF: Unambiguous and Imperceptible Model Fingerprinting for Secure Federated Learning." IEEE TCE, 2025.

3. **FedDiff**: Li et al. "FedDiff: Diffusion Model Driven Federated Learning for Multi-Modal and Multi-Clients." IEEE TCSVT, 2024.

4. **FedAWM**: Sun et al. "FedAWM: Adaptive watermark allocation in non-IID federated learning." KBS, 2026.

5. **Fed-PK-Judge**: Kanakri & King. "Fed-PK-Judge: Provably Secure Intellectual-Property Protection for Federated Learning." IEEE IoT-J, 2025.

6. **Unlearning-Guided**: Pan et al. "Robust Watermarking for Federated Diffusion Models with Unlearning-Enhanced Redundancy." IEEE TDSC, 2025.

---

**项目状态**: 阶段1完成 | **测试覆盖**: 95% | **开发方法**: TDD

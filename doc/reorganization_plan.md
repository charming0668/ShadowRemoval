# Shadow_R 项目代码整理计划

## 1. 项目规模分析

### 1.1 代码行数统计

| 文件 | 行数 | 说明 |
|------|------|------|
| model_convnext.py | 540 | 模型定义（较大，需要拆分） |
| shadow_removal/train.py | 498 | 训练逻辑 |
| shadow_removal/metrics.py | 143 | 评估指标 |
| shadow_removal/losses.py | 130 | 损失函数 |
| predict.py | 125 | 预测脚本 |
| myFFCResblock0.py | 63 | FFC模块 |
| shadow_removal/dataset.py | 76 | 数据集 |
| **核心代码总计** | **~1575行** | 中小型项目 |

### 1.2 项目定位

这是一个**中小型深度学习项目**，特点是：
- 单一任务：阴影去除
- 核心代码量适中（~1500行）
- 模块间依赖关系简单
- 主要是训练、预测两个工作流

---

## 2. 设计原则

### 2.1 避免"过度工程化"

**问题**：为每个功能创建独立文件夹（如 `losses/`、`metrics/`）会导致：
1. 目录层级过深，导航成本增加
2. 简单文件需要额外的 `__init__.py`
3. import路径冗长
4. 对于小项目来说，维护成本高于收益

**原则**：
- 文件夹应该按**功能域**划分，而非按**代码类型**划分
- 一个文件夹内可以有多个相关文件，不必每个类型单独建文件夹
- 保持扁平结构，减少嵌套

### 2.2 参考业界实践

**PyTorch官方示例项目结构**（中小型项目）：
```
project/
├── models.py          # 所有模型
├── data.py            # 数据处理
├── train.py           # 训练入口
├── evaluate.py        # 评估
├── utils.py           # 工具函数
└── config.yaml        # 配置
```

**MMDetection/MMLab系列**（大型项目）才有复杂的目录结构。

---

## 3. 推荐的项目结构

### 3.1 精简版结构（推荐）

```
Shadow_R/
├── shadow_net/                 # 核心代码包（单一命名空间）
│   ├── __init__.py
│   ├── models.py               # fusion_net + 子模块
│   ├── losses.py               # 损失函数（130行，单文件足够）
│   ├── metrics.py              # 评估指标（143行，单文件足够）
│   ├── dataset.py              # 数据集加载
│   ├── trainer.py              # 训练器
│   ├── predictor.py            # 预测器
│   └── utils.py                # DDP、checkpoint等工具
│
├── scripts/                    # 命令行入口脚本
│   ├── train.py                # 训练入口
│   ├── predict.py              # 预测入口
│   └── train_ddp.sh            # DDP启动脚本
│
├── configs/                    # 配置文件
│   └── train.yaml
│
├── outputs/                    # 输出目录
│   └── checkpoints/
│
├── third_party/                # 第三方依赖（不动）
│   └── saicinpainting/
│
├── Dataset/                    # 数据集（不动）
├── test_dataset/               # 测试数据（不动）
├── doc/                        # 文档（不动）
│
├── requirements.txt
└── README.md
```

### 3.2 结构说明

| 目录/文件 | 说明 | 设计理由 |
|-----------|------|----------|
| `shadow_net/` | 核心代码包 | 统一命名空间，所有核心模块都在这里 |
| `models.py` | 所有模型定义 | 540行的model_convnext.py需要拆分，但放在一个文件或少量文件中 |
| `losses.py` | 损失函数 | 130行，单文件足够，不需要单独文件夹 |
| `metrics.py` | 评估指标 | 143行，单文件足够 |
| `trainer.py` | 训练逻辑 | 从train.py提取核心类 |
| `scripts/` | 入口脚本 | 与核心代码分离，方便命令行调用 |

### 3.3 import方式

```python
# 简洁的导入方式
from shadow_net.models import fusion_net, Discriminator
from shadow_net.losses import CombinedLoss
from shadow_net.metrics import calculate_psnr, calculate_ssim
from shadow_net.dataset import ShadowRemovalDataset
from shadow_net.trainer import Trainer
```

对比之前的设计：
```python
# 过度设计的导入方式（不推荐）
from models.fusion_net import fusion_net
from models.discriminator import Discriminator
from losses.combined_loss import CombinedLoss
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from data.dataset import ShadowRemovalDataset
```

---

## 4. models.py 的组织

由于 `model_convnext.py` 有540行，可以考虑两种方案：

### 方案A：单文件（推荐）

```python
# shadow_net/models.py

# === DWT相关 ===
def dwt_init(x): ...
class DWT(nn.Module): ...
class DWT_transform(nn.Module): ...

# === DWT-UNet ===
class dwt_ffc_UNet2(nn.Module): ...

# === 知识适应分支 ===
class ConvNeXt(nn.Module): ...
class knowledge_adaptation_convnext(nn.Module): ...

# === 融合网络（主模型）===
class fusion_net(nn.Module): ...

# === 判别器 ===
class Discriminator(nn.Module): ...
```

**优点**：所有模型在一个文件中，方便查看整体结构

### 方案B：拆分为少量文件

```
shadow_net/
├── models/
│   ├── __init__.py
│   ├── fusion_net.py       # 主模型 + DWT
│   └── discriminator.py    # 判别器（可选）
```

如果未来代码扩展，可以考虑这种拆分。

---

## 5. 删除的内容

| 删除项 | 原因 |
|--------|------|
| `Restormer/` | 不再使用Restormer |
| `model.py` | final_net包装类，直接用fusion_net |
| 根目录的 `test.py` | 功能重复 |
| `test_dataset.py` | 功能重复 |
| `shadow_removal/test.py` | 功能重复 |
| `shadow_removal/test_compare.py` | 测试对比脚本 |

---

## 6. 详细迁移计划

### 6.1 创建核心包

```bash
mkdir -p shadow_net
touch shadow_net/__init__.py
```

### 6.2 文件迁移

| 源文件 | 目标 | 操作 |
|--------|------|------|
| `model_convnext.py` | `shadow_net/models.py` | 移动，删除Restormer相关导入 |
| `myFFCResblock0.py` | `shadow_net/ffc.py` | 移动 |
| `shadow_removal/losses.py` | `shadow_net/losses.py` | 移动 |
| `shadow_removal/metrics.py` | `shadow_net/metrics.py` | 移动 |
| `shadow_removal/dataset.py` | `shadow_net/dataset.py` | 移动 |
| `shadow_removal/train.py` | `shadow_net/trainer.py` | 提取Trainer类 |
| `predict.py` | `shadow_net/predictor.py` | 提取predict函数 |

### 6.3 创建入口脚本

```python
# scripts/train.py
from shadow_net.trainer import Trainer
from shadow_net.models import fusion_net
# ...

def main():
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
```

### 6.4 更新导入路径

在 `shadow_net/models.py` 中更新FFC的导入：

```python
# 原来
from myFFCResblock0 import myFFCResblock
from saicinpainting.training.modules.ffc0 import FFCResnetBlock

# 修改后
from .ffc import myFFCResblock
from third_party.saicinpainting.training.modules.ffc0 import FFCResnetBlock
```

---

## 7. 最终结构对比

### 之前（过度设计）
```
Shadow_R/
├── models/
│   ├── fusion_net.py
│   ├── dwt_unet.py
│   ├── knowledge_adaptation.py
│   ├── discriminator.py
│   ├── layers.py
│   └── ffc/
│       └── ffc_resblock.py
├── losses/
│   ├── __init__.py
│   └── losses.py
├── metrics/
│   ├── __init__.py
│   └── metrics.py
├── data/
│   └── dataset.py
├── engines/
│   ├── trainer.py
│   └── predictor.py
├── utils/
│   └── distributed.py
...
```

### 现在（精简实用）
```
Shadow_R/
├── shadow_net/           # 一个包，所有核心代码
│   ├── models.py         # 所有模型
│   ├── losses.py         # 损失函数
│   ├── metrics.py        # 评估指标
│   ├── dataset.py        # 数据集
│   ├── trainer.py        # 训练器
│   ├── predictor.py      # 预测器
│   ├── ffc.py            # FFC模块
│   └── utils.py          # 工具函数
├── scripts/              # 入口脚本
├── configs/              # 配置文件
├── outputs/              # 输出
├── third_party/          # 第三方依赖
└── Dataset/              # 数据集
```

---

## 8. 预计工作量

| 阶段 | 时间 |
|------|------|
| 创建shadow_net包，迁移文件 | 20分钟 |
| 更新import路径 | 20分钟 |
| 创建scripts入口 | 15分钟 |
| 删除冗余文件 | 10分钟 |
| 验证功能 | 30分钟 |
| **总计** | **约1.5小时** |

---

## 9. 执行确认

请确认是否按照此精简方案执行。如有其他调整需求请告知。

核心改动：
1. 使用 `shadow_net/` 单一包管理所有核心代码
2. 不再为 losses/metrics/data 等单独建文件夹
3. 保持扁平结构，减少嵌套

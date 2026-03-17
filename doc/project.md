# ShadowRemoval 项目文档

## 1. 项目概述

基于深度学习的阴影去除项目，使用双分支网络架构（DWT+FFC U-Net + ConvNeXt）进行阴影去除。

## 2. 项目结构

```
ShadowRemoval/
├── Dataset/                    # 数据集
│   ├── train/
│   │   ├── shadow/            # 有阴影图像
│   │   └── no_shadow/         # 无阴影图像
│   └── val/
├── shadow_removal/             # 核心模块
│   ├── train.py               # 训练脚本
│   ├── test.py                # 测试脚本
│   ├── predict.py             # 预测脚本
│   ├── dataset.py             # 数据集
│   ├── losses.py              # 损失函数
│   └── metrics.py             # 评估指标
├── saicinpainting/             # FFC模块依赖
│   ├── training/modules/      # FFC核心模块
│   └── evaluation/losses/     # LPIPS指标
├── model_convnext.py           # 主模型定义
├── myFFCResblock0.py           # FFC残差块
├── config/
│   └── train_config.yaml       # 训练配置
├── checkpoints/                # 检查点保存目录
└── models/lpips_models/        # LPIPS预训练权重
```

## 3. 模型架构

### fusion_net
双分支阴影去除网络：
- **dwt_branch**: DWT+FFC U-Net，捕获局部和全局特征
- **knowledge_adaptation_branch**: ConvNeXt编码器 + 注意力解码器

### Discriminator
用于对抗训练的判别器（可选）

## 4. 使用方法

### 训练
```bash
# 单卡训练
python shadow_removal/train.py

# 多卡DDP训练
torchrun --nproc_per_node=4 shadow_removal/train.py
```

### 测试
```bash
python shadow_removal/test.py --checkpoint checkpoints/xxx/best_model.pth --config config/test_config.yaml
```

### 预测
```bash
python shadow_removal/predict.py --checkpoint checkpoints/xxx/best_model.pth --input your_images/
```

## 5. 损失函数

```
L_total = L1 + α·L_MS-SSIM + β·L_perceptual + γ·L_adversarial
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| α | 0.2 | MS-SSIM损失权重 |
| β | 0.01 | 感知损失权重 |
| γ | 0.0005 | 对抗损失权重 |

## 6. 评估指标

- **PSNR**: 峰值信噪比
- **SSIM**: 结构相似性
- **LPIPS**: 学习感知图像块相似性

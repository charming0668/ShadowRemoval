# ShadowRefiner 项目结构文档

## 项目概述

ShadowRefiner 是一个基于 PyTorch 实现的**无掩码阴影去除**深度学习项目。

- **论文**: ShadowRefiner: Towards Mask-free Shadow Removal via Fast Fourier Transformer
- **会议**: CVPRW 2024
- **荣誉**: NTIRE 2024 阴影去除挑战赛 - 感知赛道第一名、保真度赛道第二名

## 网络架构

项目采用**两阶段架构**：

1. **阴影去除阶段** (`fusion_net`):
   - DWT分支: 使用离散小波变换捕获多尺度特征
   - 知识适应分支: 基于ConvNeXt的语义特征提取
   - 融合层: 融合两个分支的输出

2. **图像增强阶段** (`Restormer`):
   - 基于傅里叶频域注意力的图像恢复网络
   - 细化第一阶段的结果

## 文档索引

### 核心模型文档
- [01_main_models.md](01_main_models.md) - 主模型模块（model.py, model_convnext.py, myFFCResblock0.py）
- [02_restormer.md](02_restormer.md) - Restormer 图像增强模块

### 训练模块文档
- [03_training_data.md](03_training_data.md) - 数据加载和处理（datasets.py, masks.py, aug.py）
- [04_training_losses.md](04_training_losses.md) - 损失函数（perceptual.py, adversarial.py, style_loss.py, feature_matching.py, distance_weighting.py, segmentation.py）
- [05_training_modules.md](05_training_modules.md) - 网络模块（ffc.py, pix2pixhd.py, multidilated_conv.py, depthwise_sep_conv.py, squeeze_excitation.py, spatial_transform.py, multiscale.py, fake_fakes.py, base.py）
- [06_training_trainers.md](06_training_trainers.md) - 训练器（base.py, default.py）
- [07_training_visualizers.md](07_training_visualizers.md) - 可视化（base.py, directory.py, colors.py, noop.py）

### 评估模块文档
- [08_evaluation.md](08_evaluation.md) - 评估模块（evaluator.py, data.py, base_loss.py, utils.py, refinement.py）

### 测试脚本
- [09_test_scripts.md](09_test_scripts.md) - 测试脚本和工具（test.py, test_dataset.py, utils.py）

## 项目结构

```
Shadow_R/
├── model.py                      # 最终网络 final_net
├── model_convnext.py             # ConvNeXt 融合网络
├── myFFCResblock0.py             # 自定义 FFC ResNet 块
├── test.py                       # 测试脚本
├── test_dataset.py               # 测试数据集
├── Restormer/
│   ├── restormer_arch.py         # Restormer 架构
│   └── arch_util.py              # 架构工具
├── saicinpainting/
│   ├── training/
│   │   ├── data/
│   │   │   ├── datasets.py       # 训练/验证数据集
│   │   │   ├── masks.py          # 掩码生成器
│   │   │   └── aug.py            # 图像增强
│   │   ├── losses/
│   │   │   ├── perceptual.py     # 感知损失
│   │   │   ├── adversarial.py    # 对抗损失
│   │   │   ├── style_loss.py     # 风格损失
│   │   │   ├── feature_matching.py # 特征匹配
│   │   │   ├── distance_weighting.py # 距离加权
│   │   │   └── segmentation.py   # 分割损失
│   │   ├── modules/
│   │   │   ├── ffc.py / ffc0.py  # Fast Fourier Convolution
│   │   │   ├── pix2pixhd.py      # Pix2PixHD 网络
│   │   │   ├── multidilated_conv.py # 多膨胀卷积
│   │   │   ├── depthwise_sep_conv.py # 深度可分离卷积
│   │   │   ├── squeeze_excitation.py # SE模块
│   │   │   ├── spatial_transform.py # 空间变换
│   │   │   ├── multiscale.py     # 多尺度网络
│   │   │   ├── fake_fakes.py     # 虚假样本生成
│   │   │   └── base.py           # 基础模块
│   │   ├── trainers/
│   │   │   ├── base.py           # 训练基类
│   │   │   └── default.py        # 默认训练实现
│   │   └── visualizers/
│   │       ├── base.py           # 可视化基类
│   │       ├── directory.py      # 目录可视化
│   │       ├── colors.py         # 颜色工具
│   │       └── noop.py           # 空可视化
│   ├── evaluation/
│   │   ├── evaluator.py          # 评估器
│   │   ├── data.py               # 评估数据集
│   │   ├── losses/
│   │   │   └── base_loss.py      # 评估损失基类
│   │   ├── refinement.py         # 细化后处理
│   │   └── utils.py              # 评估工具
│   └── utils.py                  # 通用工具
└── doc/                          # 项目文档
    ├── README.md                 # 本文档
    └── *.md                      # 各模块详细文档
```

## 快速开始

### 环境要求
- Python 3.8
- PyTorch 1.11
- CUDA 11.3

### 安装依赖
```bash
conda create --name shadowrefiner python=3.8
conda activate shadowrefiner
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
pip install numpy matplotlib scikit-learn scikit-image opencv-python timm kornia einops pytorch_lightning
```

### 测试模型
```bash
python test.py --test_dir ./ShadowDataset/test/ --output_dir results/
```

## 核心算法

### 1. DWT-FFC 分支
- 使用离散小波变换分解图像为低频/高频分量
- 使用 Fast Fourier Convolution 处理全局特征
- UNet架构进行多尺度特征融合

### 2. 知识适应分支
- ConvNeXt作为编码器提取语义特征
- 多尺度注意力机制（通道注意力和像素注意力）
- 渐进式上采样解码

### 3. Restormer 增强
- 傅里叶频域自注意力（FSAS）
- 门控深度可分离前馈网络（GDFN）
- 多尺度Transformer块

## 损失函数

- **L1 损失**: 像素级重建损失
- **感知损失**: VGG19/ResNet特征匹配
- **对抗损失**: 非饱和损失 + R1梯度惩罚
- **特征匹配损失**: 判别器中间层特征匹配

## 评估指标

- **SSIM**: 结构相似性指数
- **LPIPS**: 学习感知图像块相似度
- **FID**: Fréchet Inception Distance
- **F1综合指标**: SSIM/FID或LPIPS/FID的调和平均

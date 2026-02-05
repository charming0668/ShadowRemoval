# Shadow_R 项目实施规划

## 项目目标

基于 Shadow_R 模型实现一个精简的阴影去除训练和评估系统，仅使用阴影去除模块（remove_model），不使用 mask 和 Restormer 精细化模块。

---

## 第一阶段：数据处理模块

### 1.1 数据集类实现
**目标**: 创建自定义 Dataset 类，支持配对的阴影/无阴影图像加载

**实现要点**:
- 读取 `Dataset/train/` 和 `Dataset/val/` 目录
- 配对加载 shadow 和 no_shadow 图像
- 支持图像预处理（归一化到 [0, 1] 或 [-1, 1]）
- 返回格式：`(shadow_image, no_shadow_image, filename)`

**文件**: `dataset.py`

### 1.2 数据增强实现
**目标**: 实现训练时的数据增强策略

**增强方法**:
- 随机裁剪：384 × 384 像素
- 随机旋转：90°, 180°, 270°
- 随机翻转：水平翻转、垂直翻转

**实现方式**:
- 使用 torchvision.transforms 或自定义增强函数
- 确保 shadow 和 no_shadow 图像应用相同的增强操作

**文件**: `dataset.py` 或 `augmentation.py`

### 1.3 DataLoader 配置
**目标**: 创建训练和验证的 DataLoader

**配置参数**:
- batch_size: 可配置（建议 4-8）
- num_workers: 可配置（建议 4）
- shuffle: 训练集 True，验证集 False
- pin_memory: True（加速 GPU 传输）

---

## 第二阶段：模型模块

### 2.1 模型加载
**目标**: 加载并配置 remove_model

**实现要点**:
- 从 `model.py` 导入模型
- 移除 Restormer 相关代码
- 仅保留阴影去除模块
- 支持从检查点恢复（可选）

**文件**: `train.py`

### 2.2 模型初始化
**目标**: 正确初始化模型权重

**实现要点**:
- 检查是否有预训练权重
- 支持从头训练或断点续训
- 将模型移至 GPU/CPU

---

## 第三阶段：损失函数与优化器

### 3.1 损失函数实现
**目标**: 实现组合损失函数

**损失公式**:
```
L_total = L1_loss + α * SSIM_loss
```

**实现要点**:
- L1 Loss: `torch.nn.L1Loss()`
- SSIM Loss: 使用现有库（如 pytorch-msssim）或自实现
- α 参数可配置（建议 0.1-0.5）

**文件**: `losses.py` 或 `train.py`

### 3.2 优化器配置
**目标**: 配置 Adam 优化器

**参数设置**:
- optimizer: Adam
- β₁ = 0.9
- β₂ = 0.999
- 初始学习率: 1e-4
- 仅优化 remove_model 参数

**文件**: `train.py`

### 3.3 学习率调度器
**目标**: 实现学习率衰减策略

**策略**:
需要warm-up
- 前 10% epoch 使用 warm-up
- 从 1e-4 逐渐降低至 6.25e-6
- 可选方案：
  - CosineAnnealingLR

**文件**: `train.py`

---

## 第四阶段：评估指标

### 4.1 PSNR 实现
**目标**: 实现峰值信噪比计算

**实现要点**:
- 输入：预测图像和真实图像
- 输出：PSNR 值（dB）
- 支持 batch 计算

**文件**: `metrics.py`

### 4.2 SSIM 实现
**目标**: 实现结构相似性计算

**实现要点**:
- 使用 pytorch-msssim 库或自实现
- 输入：预测图像和真实图像
- 输出：SSIM 值 [0, 1]
- 支持 batch 计算

**文件**: `metrics.py`

---

## 第五阶段：训练循环

### 5.1 训练主循环
**目标**: 实现完整的训练流程

**流程**:
1. 遍历训练 DataLoader
2. 前向传播：模型预测
3. 计算损失：L1 + α*SSIM
4. 反向传播：梯度更新
5. 记录训练损失

**文件**: `train.py`

### 5.2 验证循环
**目标**: 实现验证流程

**流程**:
1. 设置模型为评估模式
2. 遍历验证 DataLoader
3. 前向传播（无梯度）
4. 计算 PSNR 和 SSIM
5. 返回平均指标

**文件**: `train.py`

### 5.3 训练监控
**目标**: 每个 epoch 输出关键信息

**输出内容**:
- Epoch 编号
- 训练损失
- 验证 PSNR
- 验证 SSIM
- 当前学习率

---

## 第六阶段：实验管理

### 6.1 SwanLab 集成
**目标**: 使用 SwanLab 记录实验

**记录内容**:
- 训练损失（每个 step）
- 验证 PSNR（每个 epoch）
- 验证 SSIM（每个 epoch）
- 学习率（每个 epoch）
- 可选：可视化样本图像

**文件**: `train.py`

### 6.2 检查点保存
**目标**: 保存模型检查点

**保存策略**:
- **最新模型**: 每个 epoch 保存 `latest_model.pth`
- **最佳模型**: 根据验证 PSNR 保存 `best_model.pth`

**保存内容**:
```python
{
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_psnr': best_psnr,
    'best_ssim': best_ssim
}
```

**文件**: `train.py`

### 6.3 断点续训
**目标**: 支持从检查点恢复训练

**实现要点**:
- 加载模型权重
- 加载优化器状态
- 加载学习率调度器状态
- 恢复 epoch 和最佳指标

**文件**: `train.py`

---

## 第七阶段：配置管理

### 7.1 配置文件
**目标**: 创建训练配置文件

**配置项**:
```yaml
# 数据配置
data:
  train_dir: "Dataset/train"
  val_dir: "Dataset/val"
  crop_size: 384
  batch_size: 4
  num_workers: 4

# 模型配置
model:
  name: "remove_model"
  resume: null  # 检查点路径

# 训练配置
training:
  epochs: 100
  lr: 1e-4
  lr_min: 6.25e-6
  alpha: 0.2  # SSIM 权重
  
# 保存配置
checkpoint:
  save_dir: "checkpoints"
  save_freq: 1  # 每个 epoch 保存

# 日志配置
logging:
  use_swanlab: true
  log_freq: 10  # 每 10 个 step 记录一次
```

**文件**: `config.yaml`
omegaconf来进行加载配置
---

## 第八阶段：测试与评估

### 8.1 测试脚本
**目标**: 创建独立的测试脚本

**功能**:
- 加载训练好的模型
- 在测试集上评估
- 计算 PSNR 和 SSIM
- 保存去除阴影后的图像

**文件**: `test.py` 或 `evaluate.py`

### 8.2 可视化
**目标**: 可视化去除阴影效果

**实现**:
- 并排显示：输入 | 预测 | 真实
- 保存对比图像
- 可选：生成 HTML 报告

---

## 项目文件结构

```
Shadow_R/
├── dataset.py              # 数据集类
├── model.py                # 模型定义（已有，需修改）
├── losses.py               # 损失函数
├── metrics.py              # 评估指标
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
├── config.yaml             # 配置文件
├── utils.py                # 工具函数
├── checkpoints/            # 检查点保存目录
│   ├── best_model.pth
│   └── latest_model.pth
├── Dataset/                # 数据集（已有）
│   ├── train/
│   └── val/
└── README.md               # 项目说明
```

---

## 实施优先级

### 高优先级（核心功能）
1. ✅ 数据集类和 DataLoader
2. ✅ 模型加载和初始化
3. ✅ 损失函数实现
4. ✅ 训练和验证循环
5. ✅ 检查点保存

### 中优先级（增强功能）
6. ✅ 数据增强
7. ✅ 评估指标（PSNR, SSIM）
8. ✅ SwanLab 集成
9. ✅ 学习率调度

### 低优先级（可选功能）
10. ⭕ 断点续训
11. ⭕ 配置文件管理
12. ⭕ 测试脚本
13. ⭕ 可视化工具

---

## 开发建议

### 代码规范
1. **精简原则**: 避免过度的错误检查和冗余逻辑
2. **清晰结构**: 每个模块职责单一
3. **中文注释**: 关键步骤添加中文注释
4. **函数文档**: 每个函数添加 docstring



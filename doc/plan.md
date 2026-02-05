# Shadow_R 训练计划

本计划基于项目现有代码与文档要求，给出从数据集到训练与评估的完整实现路径。目标是以 PyTorch Lightning 为训练框架，使用 `final_net`（[model.py](../model.py)）构建两阶段训练流程，并支持 SwanLab 记录与最佳/最新模型保存。

## 1. 总体目标

- 构建面向成对数据集（shadow/no_shadow）的数据集与数据加载器。
- 以 LightningModule 方式封装 `final_net`，完成两阶段训练。
- 复用 saicinpainting 的损失与评估组件，补齐 MS-SSIM 与 PSNR。
- 完整支持 DDP、多卡日志约束、SwanLab、模型断点与最佳模型保存。

## 2. 需要新增或修改的模块

### 2.1 数据集与数据加载器

- 新增 `ShadowRemovalDataset`，按文件名匹配 `shadow/` 与 `no_shadow/`。
- 训练增强：随机裁剪 384、随机 90/180/270 旋转、水平/垂直翻转。
- 生成 `train_dataloader` 与 `val_dataloader`。
- 推荐位置：新建 [shadow_dataset.py](../shadow_dataset.py) 或扩展 [saicinpainting/training/data/datasets.py](../saicinpainting/training/data/datasets.py)。

### 2.2 LightningModule 封装

- 新增 `ShadowRemovalLightningModule`，内部使用 `final_net`。
- 提供 `training_step`、`validation_step`、`configure_optimizers`。
- 在 `on_train_epoch_start` 中进行阶段切换：
  - Stage I：仅训练去阴影模块。
  - Stage II：联合训练去阴影 + 精细化模块。
- 日志输出需仅在 rank=0。

### 2.3 损失函数

- L1：`torch.nn.L1Loss`。
- MS-SSIM：新增实现（可独立文件）。
- 感知损失：优先复用 `saicinpainting/training/losses/perceptual.py`，若依赖缺失则自建 VGG19 版本。
- 对抗损失：Stage I 使用 `saicinpainting/training/losses/adversarial.py`。

### 2.4 评估指标

- 训练/验证记录 PSNR、SSIM、LPIPS。
- 若使用 evaluator：扩展 [saicinpainting/evaluation](../saicinpainting/evaluation) 支持 PSNR。
- 若直接在 LightningModule 中评估：在 `validation_step` 计算并累计。

### 2.5 训练入口脚本

- 新建 [train_pl.py](../train_pl.py)。
- 完成配置加载、模型实例化、数据加载、Trainer 配置、启动训练。
- 支持 DDP 多卡与断点恢复。

### 2.6 配置文件

- 重要/高频参数放入 config（路径、batch size、epochs、lr、阶段切换、损失权重）。
- 低频参数硬编码（增强方式、日志频率等）。
- 推荐：新建 [configs/train.yaml](../configs/train.yaml)。

### 2.7 SwanLab 记录

- 在 `train_pl.py` 中集成 SwanLab logger。
- 记录训练损失、验证指标、学习率。

### 2.8 Checkpoint 保存

- 使用 `ModelCheckpoint`：
  - `best_model.pth`（按 PSNR 评估）。
  - `latest_model.pth`（每个 epoch）。
- 支持 `resume` 参数恢复训练。

## 3. 两阶段训练策略

### Stage I

- 优化目标：仅优化去阴影模块。
- 学习率：从 1e-4 逐步下降至 6.25e-6（cosine）。
- 损失：L1 + α·MS-SSIM + β·Percep + γ·Adv。

### Stage II

- 优化目标：联合优化去阴影 + 精细化模块。
- 学习率：固定 1e-5。
- 损失：L1 + α·MS-SSIM + β·Percep。

## 4. 关键实现顺序

1. 实现配对数据集与增强逻辑。
2. 封装 LightningModule，接入 `final_net`。
3. 补齐 MS-SSIM、PSNR、Perceptual loss 的依赖。
4. 完成训练入口与配置系统。
5. 加入 SwanLab 与 Checkpoint。
6. 进行单卡与多卡小规模验证。

## 5. 验证与运行建议

- 先使用小样本进行 sanity check，确保 loss 与指标正常。
- 再进行完整训练并检查 SwanLab 记录。
- 多卡训练验证日志是否仅 rank=0 输出。

## 6. 交付清单

- 数据集与 dataloader 模块。
- LightningModule 封装。
- Loss 与 metrics 实现补齐。
- 训练入口脚本与配置文件。
- SwanLab 与 checkpoint 机制。

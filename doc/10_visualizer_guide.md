# Visualizer 可视化器使用指南

## 概述

Visualizer（可视化器）用于在模型训练过程中可视化修复结果，帮助监控训练进度和质量。它会将输入图像、模型预测结果、修复后图像等拼接成对比图，并保存到指定目录。

## Visualizer 的工作原理

### 1. 调用时机

Visualizer 在以下时机被调用：

```python
# 在 base.py 的 _do_step 方法中
if self.get_ddp_rank() in (None, 0) and (batch_idx % self.visualize_each_iters == 0 or mode == 'test'):
    self.visualizer(self.current_epoch, batch_idx, batch, suffix=vis_suffix)
```

- **训练阶段**：每隔 `visualize_each_iters` 个批次（默认100）保存一次
- **验证/测试阶段**：每个批次都保存
- **分布式训练**：只在主进程（rank 0）执行

### 2. 处理流程

```
输入批次数据
    ↓
提取指定键的图像（如 'image', 'predicted_image', 'inpainted'）
    ↓
应用图像变换（归一化、维度调整）
    ↓
在图像上叠加掩码边界（红色轮廓）
    ↓
水平拼接多组图像
    ↓
垂直拼接批次中的多个样本
    ↓
保存为图片文件
```

## 生成的结果结构

### 目录结构

```
output_dir/                     # 可视化输出根目录
└── epoch{NNNN}{suffix}/        # 每个epoch一个目录
    └── batch{MMMMMMM}{rank}.jpg  # 每个保存批次一张图片
```

**命名规则：**
- `epoch{NNNN}{suffix}`: 周期目录
  - `NNNN`: 4位数字，如 `0001`, `0010`
  - `suffix`: 模式后缀，如 `_train`, `_val`, `_test`, `_extra_val_xxx`
  
- `batch{MMMMMMM}{rank}.jpg`: 批次图片文件
  - `MMMMMMM`: 7位批次编号，如 `0000100`
  - `rank`: 分布式训练的rank后缀（如 `_r0`），单卡训练无此后缀

**示例：**
```
visualizations/
├── epoch0001_train/
│   ├── batch0000000.jpg        # 训练第0批次
│   ├── batch0000100.jpg        # 训练第100批次
│   └── batch0000200.jpg
├── epoch0001_val/              # 验证结果
│   └── batch0000000.jpg
├── epoch0001_test/             # 测试结果
│   └── batch0000000.jpg
├── epoch0002_train/
│   └── ...
```

### 图片内容结构

#### 单张图片布局

```
┌─────────────────────────────────────────────────────────────────┐
│ Sample 1:                                                       │
│ ┌─────────────┬─────────────┬─────────────┐                     │
│ │             │             │             │                     │
│ │   Input     │  Predicted  │  Inpainted  │   ← 水平拼接        │
│ │   (原图)     │  (模型输出)  │  (最终修复)  │                     │
│ │             │             │             │                     │
│ │  [红框边界]  │  [红框边界]  │  (无边界)    │                     │
│ └─────────────┴─────────────┴─────────────┘                     │
├─────────────────────────────────────────────────────────────────┤
│ Sample 2:                                                       │
│ ┌─────────────┬─────────────┬─────────────┐                     │
│ │   Input     │  Predicted  │  Inpainted  │   ← 垂直拼接        │
│ └─────────────┴─────────────┴─────────────┘                     │
├─────────────────────────────────────────────────────────────────┤
│ Sample 3:                                                       │
│ ...                                                             │
└─────────────────────────────────────────────────────────────────┘
```

#### 视觉元素说明

1. **输入图像 (Input)**
   - 带阴影/损坏的原始图像
   - 显示红色掩码边界（轮廓）
   - 白色边界线标注

2. **预测图像 (Predicted)**
   - 模型的直接输出
   - 显示红色掩码边界
   - 可以看到模型在掩码区域内的生成结果

3. **修复结果 (Inpainted)**
   - 最终修复图像（掩码区域用预测填充）
   - **不显示**掩码边界（`last_without_mask=True`）
   - 展示最终效果

#### 边界标注样式

- **红色粗线**：表示掩码区域的边界
- **白色轮廓**：增强可见性
- **模式**：`'thick'`（粗线）

```
图像示例：
┌────────────────────┐
│ 正常区域            │
│    ┌──────────┐    │
│    │ ████████ │    │  ← 红色边界（厚）
│    │ █ 掩码 █ │    │
│    │ █ 区域 █ │    │
│    └──────────┘    │
│ 正常区域            │
└────────────────────┘
```

## 关键配置参数

### DirectoryVisualizer 参数

```python
DirectoryVisualizer(
    outdir='visualizations',           # 输出目录
    key_order=['image', 'predicted_image', 'inpainted'],  # 图像键顺序
    max_items_in_batch=10,              # 每批最多显示样本数
    last_without_mask=True,             # 最后一列是否不加掩码边界
    rescale_keys=None                   # 需要重新缩放的键
)
```

### 配置影响

| 参数 | 说明 | 示例 |
|------|------|------|
| `key_order` | 控制显示哪些图像及顺序 | `['image', 'predicted_image']` 只显示两列 |
| `max_items_in_batch` | 每张大图包含的样本数 | 10表示一张图显示10个样本 |
| `last_without_mask` | 最后一列是否去除掩码边界 | True时修复结果更清晰 |
| `rescale_keys` | 指定需要重新归一化的图像 | 用于显示注意力图等非RGB数据 |

## 使用示例

### 1. 基本使用（训练配置中）

```python
# config.yaml
visualizer:
  kind: directory
  outdir: ./visualizations
  key_order:
    - image
    - predicted_image
    - inpainted
  max_items_in_batch: 10
  last_without_mask: true
```

### 2. 自定义键顺序

```python
# 显示更多中间结果
visualizer = DirectoryVisualizer(
    outdir='visualizations',
    key_order=['image', 'predicted_image', 'inpainted', 'discr_output_fake'],
    max_items_in_batch=5
)
```

### 3. 禁用可视化

```python
# 使用 NoopVisualizer（不保存任何内容）
visualizer:
  kind: noop
```

## 代码实现细节

### 图像处理流程

```python
# 1. 提取批次数据为numpy数组
batch = {k: tens.detach().cpu().numpy() for k, tens in batch.items() if k in keys}

# 2. 对每个样本处理
for i in range(items_to_vis):
    cur_dct = {k: tens[i] for k, tens in batch.items()}
    
    # 3. 处理每个图像
    for k in keys:
        img = cur_dct[k]
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        
        # 4. 叠加掩码边界（除最后一个）
        if need_mark_boundaries:
            img = mark_boundaries(img, mask[0], color=(1., 0., 0.), outline_color=(1., 1., 1.))
    
    # 5. 水平拼接
    row = np.concatenate(result, axis=1)

# 6. 垂直拼接所有样本
final_image = np.concatenate(all_rows, axis=0)

# 7. 转RGB并保存
vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')
vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(out_fname, vis_img)
```

### 颜色转换说明

```
PyTorch 张量格式: (C, H, W), RGB, 范围 [0, 1]
         ↓ transpose((1, 2, 0))
NumPy 数组格式: (H, W, C), RGB, 范围 [0, 1]
         ↓ mark_boundaries
添加边界后的数组: (H, W, C), RGB, 范围 [0, 1]
         ↓ *255 + astype(uint8)
图像数组: (H, W, C), RGB, 范围 [0, 255]
         ↓ cvtColor(COLOR_RGB2BGR)
OpenCV格式: (H, W, C), BGR, 范围 [0, 255]
         ↓ imwrite
保存为JPEG文件
```

## 调试与排错

### 常见问题

1. **图片全黑或全白**
   - 检查输入张量范围是否在 [0, 1]
   - 使用 `check_and_warn_input_range` 函数验证

2. **掩码边界不显示**
   - 检查 `mask` 键是否存在于 batch 中
   - 检查 `last_without_mask` 是否为 False

3. **颜色失真**
   - 确认 `cvtColor(COLOR_RGB2BGR)` 已正确调用
   - OpenCV 默认使用 BGR 格式

### 验证可视化

```python
# 手动测试可视化
from saicinpainting.training.visualizers import DirectoryVisualizer
import torch
import numpy as np

# 创建测试批次
batch = {
    'image': torch.rand(2, 3, 256, 256),  # 2个样本
    'predicted_image': torch.rand(2, 3, 256, 256),
    'inpainted': torch.rand(2, 3, 256, 256),
    'mask': torch.rand(2, 1, 256, 256) > 0.5  # 二值掩码
}

# 创建可视化器并执行
visualizer = DirectoryVisualizer(outdir='test_vis')
visualizer(epoch_i=0, batch_i=0, batch=batch, suffix='_test')

# 检查 test_vis/epoch0000_test/batch0000000.jpg
```

## 扩展开发

### 自定义 Visualizer

```python
from saicinpainting.training.visualizers.base import BaseVisualizer

class CustomVisualizer(BaseVisualizer):
    def __init__(self, outdir, **kwargs):
        self.outdir = outdir
        # 自定义初始化
    
    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        # 自定义可视化逻辑
        # 例如：保存为GIF、生成对比视频、上传到云端等
        pass

# 注册到工厂函数
def make_visualizer(kind, **kwargs):
    if kind == 'custom':
        return CustomVisualizer(**kwargs)
    # ...
```

## 总结

Visualizer 是训练监控的重要工具，通过定期保存可视化结果，可以：

1. **监控训练进度**：观察模型输出随 epoch 的变化
2. **发现问题**：如模式坍塌、颜色偏移、边界伪影等
3. **对比实验**：保存不同配置的结果便于对比
4. **生成展示材料**：用于论文、报告、演示

生成的目录结构清晰，便于批量查看和分析训练效果。

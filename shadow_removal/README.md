# Shadow Removal 训练系统

精简的阴影去除训练代码，支持单GPU和多GPU DDP训练。

## 快速开始

### 单GPU训练
```bash
cd shadow_removal
python train.py
```

### 多GPU DDP训练
```bash
cd shadow_removal
torchrun --nproc_per_node=4 train.py
```

### 测试模型
```bash
python test.py --checkpoint ../checkpoints/best_model.pth
```

## 配置文件

训练配置: `config/train_config.yaml`
测试配置: `config/test_config.yaml`

## 数据集格式

```
Dataset/
├── train/
│   ├── shadow/
│   └── no_shadow/
└── val/
    ├── shadow/
    └── no_shadow/
```

## 特性

- ✅ DDP多GPU训练支持
- ✅ 训练和验证都使用多卡
- ✅ 多卡指标自动同步
- ✅ 主进程打印信息和进度条
- ✅ Warm-up + 余弦退火学习率
- ✅ L1 + SSIM组合损失
- ✅ PSNR/SSIM评估
- ✅ SwanLab实验记录
- ✅ 自动保存最佳模型
- ✅ 断点续训

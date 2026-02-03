# Training 训练器模块文档

## base.py

基础训练模块，PyTorch Lightning 封装。

### 核心函数

#### `make_optimizer(parameters, kind='adamw', **kwargs)`

```python
def make_optimizer(parameters, kind='adamw', **kwargs):
    """
    创建优化器。
    
    Args:
        parameters: 优化参数
        kind (str): 优化器类型 ('adam', 'adamw')
        **kwargs: 优化器参数
    
    Returns:
        Optimizer: 优化器实例
    """
    pass
```

#### `update_running_average(result, new_iterate_model, decay=0.999)`

```python
def update_running_average(result, new_iterate_model, decay=0.999):
    """
    更新运行平均（EMA）。
    
    Args:
        result (nn.Module): 目标模型（EMA模型）
        new_iterate_model (nn.Module): 新模型
        decay (float): 衰减系数
    """
    pass
```

#### `make_multiscale_noise(base_tensor, scales=6, scale_mode='bilinear')`

```python
def make_multiscale_noise(base_tensor, scales=6, scale_mode='bilinear'):
    """
    生成多尺度噪声。
    
    Args:
        base_tensor (Tensor): 基础张量（用于设备参考）
        scales (int): 噪声尺度数量
        scale_mode (str): 缩放模式
    
    Returns:
        Tensor: 多尺度噪声
    """
    pass
```

### 核心类

#### `BaseInpaintingTrainingModule(ptl.LightningModule)`

```python
class BaseInpaintingTrainingModule(ptl.LightningModule):
    """
    图像修复训练基类。
    
    基于PyTorch Lightning的训练模块，包含生成器、判别器、损失和评估器。
    
    Args:
        config: 训练配置
        use_ddp (bool): 是否使用分布式训练
        predict_only (bool): 是否仅预测（不训练）
        visualize_each_iters (int): 可视化间隔
        average_generator (bool): 是否使用生成器EMA
        generator_avg_beta (float): EMA衰减系数
        average_generator_start_step (int): EMA开始步数
        average_generator_period (int): EMA更新周期
        store_discr_outputs_for_vis (bool): 是否存储判别器输出用于可视化
    
    Attributes:
        generator (nn.Module): 生成器
        discriminator (nn.Module): 判别器
        adversarial_loss: 对抗损失
        visualizer: 可视化器
        val_evaluator: 验证评估器
        test_evaluator: 测试评估器
    """
    
    def __init__(self, config, use_ddp, *args, predict_only=False, 
                 visualize_each_iters=100, average_generator=False,
                 generator_avg_beta=0.999, average_generator_start_step=30000,
                 average_generator_period=10, store_discr_outputs_for_vis=False, **kwargs):
        """初始化训练模块。"""
        pass
    
    def configure_optimizers(self):
        """
        配置优化器。
        
        Returns:
            list: 生成器和判别器的优化器配置
        """
        pass
    
    def train_dataloader(self):
        """
        获取训练数据加载器。
        
        Returns:
            DataLoader: 训练数据加载器
        """
        pass
    
    def val_dataloader(self):
        """
        获取验证数据加载器。
        
        Returns:
            list: 验证数据加载器列表
        """
        pass
    
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """
        训练步骤。
        
        Args:
            batch (dict): 数据批次
            batch_idx (int): 批次索引
            optimizer_idx (int, optional): 优化器索引（0=生成器, 1=判别器）
        
        Returns:
            Tensor: 损失值
        """
        pass
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        """
        验证步骤。
        
        Args:
            batch (dict): 数据批次
            batch_idx (int): 批次索引
            dataloader_idx (int): 数据加载器索引
        
        Returns:
            dict: 验证结果
        """
        pass
    
    def training_step_end(self, batch_parts_outputs):
        """
        训练步骤结束（处理分布式）。
        
        Args:
            batch_parts_outputs: 批次输出
        
        Returns:
            Tensor: 聚合后的损失
        """
        pass
    
    def validation_epoch_end(self, outputs):
        """
        验证周期结束。
        
        Args:
            outputs (list): 验证输出列表
        """
        pass
    
    def _do_step(self, batch, batch_idx, mode='train', optimizer_idx=None, extra_val_key=None):
        """
        执行单步训练/验证。
        
        Args:
            batch (dict): 数据批次
            batch_idx (int): 批次索引
            mode (str): 模式 ('train', 'val', 'test', 'extra_val')
            optimizer_idx (int): 优化器索引
            extra_val_key (str, optional): 额外验证键
        
        Returns:
            dict: 步骤结果
        """
        pass
    
    def get_current_generator(self, no_average=False):
        """
        获取当前生成器（或EMA生成器）。
        
        Args:
            no_average (bool): 是否不使用EMA
        
        Returns:
            nn.Module: 生成器
        """
        pass
    
    def forward(self, batch):
        """
        前向传播（抽象方法）。
        
        Args:
            batch (dict): 输入批次
        
        Returns:
            dict: 至少包含'predicted_image'和'inpainted'
        """
        pass
    
    def generator_loss(self, batch):
        """
        计算生成器损失（抽象方法）。
        
        Args:
            batch (dict): 数据批次
        
        Returns:
            tuple: (总损失, 指标字典)
        """
        pass
    
    def discriminator_loss(self, batch):
        """
        计算判别器损失（抽象方法）。
        
        Args:
            batch (dict): 数据批次
        
        Returns:
            tuple: (总损失, 指标字典)
        """
        pass
    
    def store_discr_outputs(self, batch):
        """
        存储判别器输出用于可视化。
        
        Args:
            batch (dict): 数据批次
        """
        pass
    
    def get_ddp_rank(self):
        """
        获取DDP进程rank。
        
        Returns:
            int or None: rank或None（非分布式）
        """
        pass
```

---

## default.py

默认训练实现。

### 核心函数

#### `make_constant_area_crop_batch(batch, **kwargs)`

```python
def make_constant_area_crop_batch(batch, **kwargs):
    """
    对批次应用恒定面积裁剪。
    
    Args:
        batch (dict): 数据批次
        **kwargs: 裁剪参数
    
    Returns:
        dict: 裁剪后的批次
    """
    pass
```

### 核心类

#### `DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule)`

```python
class DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule):
    """
    默认图像修复训练模块。
    
    实现了标准的图像修复训练流程，包含L1、感知、对抗和特征匹配损失。
    
    Args:
        config: 训练配置
        concat_mask (bool): 是否拼接掩码到输入，默认True
        rescale_scheduler_kwargs (dict, optional): 重缩放调度器参数
        image_to_discriminator (str): 传递给判别器的图像键，默认'predicted_image'
        add_noise_kwargs (dict, optional): 添加噪声参数
        noise_fill_hole (bool): 是否用噪声填充空洞
        const_area_crop_kwargs (dict, optional): 恒定面积裁剪参数
        distance_weighter_kwargs (dict, optional): 距离加权器参数
        distance_weighted_mask_for_discr (bool): 判别器是否使用距离加权掩码
        fake_fakes_proba (float): 虚假样本概率，默认0
        fake_fakes_generator_kwargs (dict, optional): 虚假样本生成器参数
    
    Attributes:
        concat_mask (bool): 是否拼接掩码
        rescale_size_getter: 重缩放尺寸获取器
        image_to_discriminator (str): 判别器图像键
        add_noise_kwargs (dict): 噪声参数
        noise_fill_hole (bool): 噪声填充空洞标志
        const_area_crop_kwargs (dict): 裁剪参数
        refine_mask_for_losses: 损失掩码细化器
        distance_weighted_mask_for_discr (bool): 判别器距离加权标志
        fake_fakes_proba (float): 虚假样本概率
        fake_fakes_gen (FakeFakesGenerator): 虚假样本生成器
    """
    
    def __init__(self, *args, concat_mask=True, rescale_scheduler_kwargs=None,
                 image_to_discriminator='predicted_image', add_noise_kwargs=None,
                 noise_fill_hole=False, const_area_crop_kwargs=None,
                 distance_weighter_kwargs=None, distance_weighted_mask_for_discr=False,
                 fake_fakes_proba=0, fake_fakes_generator_kwargs=None, **kwargs):
        """初始化默认训练模块。"""
        pass
    
    def forward(self, batch):
        """
        前向传播。
        
        处理批次数据，应用变换，生成修复图像。
        
        Args:
            batch (dict): 输入批次，包含'image', 'mask'
        
        Returns:
            dict: 输出批次，包含'predicted_image', 'inpainted', 'mask_for_losses'等
        """
        pass
    
    def generator_loss(self, batch):
        """
        计算生成器损失。
        
        包含：
        - L1损失
        - VGG感知损失
        - 对抗损失
        - 特征匹配损失
        - ResNet感知损失
        
        Args:
            batch (dict): 数据批次
        
        Returns:
            tuple: (总损失, 指标字典)
        """
        pass
    
    def discriminator_loss(self, batch):
        """
        计算判别器损失。
        
        Args:
            batch (dict): 数据批次
        
        Returns:
            tuple: (总损失, 指标字典)
        """
        pass
```

---

## __init__.py

训练器工厂函数。

### 核心函数

#### `get_training_model_class(kind)`

```python
def get_training_model_class(kind):
    """
    获取训练模型类。
    
    Args:
        kind (str): 模型类型 ('default')
    
    Returns:
        type: 训练模型类
    """
    pass
```

#### `make_training_model(config)`

```python
def make_training_model(config):
    """
    创建训练模型。
    
    Args:
        config: 训练配置
    
    Returns:
        BaseInpaintingTrainingModule: 训练模型实例
    """
    pass
```

#### `load_checkpoint(train_config, path, map_location='cuda', strict=True)`

```python
def load_checkpoint(train_config, path, map_location='cuda', strict=True):
    """
    加载检查点。
    
    Args:
        train_config: 训练配置
        path (str): 检查点路径
        map_location (str): 设备映射
        strict (bool): 是否严格匹配
    
    Returns:
        BaseInpaintingTrainingModule: 加载的模型
    """
    pass
```

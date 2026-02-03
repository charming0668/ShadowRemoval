# Training 损失函数模块文档

## perceptual.py

感知损失，基于预训练网络特征。

### 核心类

#### `PerceptualLoss(nn.Module)`

```python
class PerceptualLoss(nn.Module):
    """
    基于VGG19的感知损失。
    
    使用VGG19特征提取器计算预测图像与目标图像的特征差异。
    将MaxPool替换为AvgPool以获得更平滑的特征。
    
    Args:
        normalize_inputs (bool): 是否归一化输入，默认True
    
    Attributes:
        vgg (nn.Sequential): 修改后的VGG19特征提取器
        mean_ (Tensor): ImageNet均值
        std_ (Tensor): ImageNet标准差
    """
    
    def __init__(self, normalize_inputs=True):
        """初始化感知损失。"""
        pass
    
    def do_normalize_inputs(self, x):
        """
        对输入进行ImageNet归一化。
        
        Args:
            x (Tensor): 输入图像，范围[0, 1]
        
        Returns:
            Tensor: 归一化后的图像
        """
        pass
    
    def partial_losses(self, input, target, mask=None):
        """
        计算各层的部分损失。
        
        Args:
            input (Tensor): 预测图像
            target (Tensor): 目标图像
            mask (Tensor, optional): 掩码
        
        Returns:
            list: 各ReLU层的MSE损失列表
        """
        pass
    
    def forward(self, input, target, mask=None):
        """
        前向传播计算感知损失。
        
        Args:
            input (Tensor): 预测图像
            target (Tensor): 目标图像
            mask (Tensor, optional): 掩码
        
        Returns:
            Tensor: 总感知损失
        """
        pass
    
    def get_global_features(self, input):
        """
        获取全局特征。
        
        Args:
            input (Tensor): 输入图像
        
        Returns:
            Tensor: VGG全局特征
        """
        pass
```

#### `ResNetPL(nn.Module)`

```python
class ResNetPL(nn.Module):
    """
    基于ResNet的感知损失。
    
    使用ADE20K分割编码器作为特征提取器。
    
    Args:
        weight (float): 损失权重
        weights_path (str): 预训练权重路径
        arch_encoder (str): 编码器架构
        segmentation (bool): 是否使用分割
    """
    
    def __init__(self, weight=1, weights_path=None, 
                 arch_encoder='resnet50dilated', segmentation=True):
        """初始化ResNet感知损失。"""
        pass
    
    def forward(self, pred, target):
        """
        计算ResNet感知损失。
        
        Args:
            pred (Tensor): 预测图像
            target (Tensor): 目标图像
        
        Returns:
            Tensor: 感知损失值
        """
        pass
```

---

## adversarial.py

对抗损失函数。

### 核心类

#### `BaseAdversarialLoss`

```python
class BaseAdversarialLoss:
    """
    对抗损失基类。
    
    定义生成器和判别器损失的接口。
    """
    
    def pre_generator_step(self, real_batch, fake_batch, generator, discriminator):
        """
        生成器步骤前的准备。
        
        Args:
            real_batch (Tensor): 真实样本批次
            fake_batch (Tensor): 生成样本批次
            generator (nn.Module): 生成器
            discriminator (nn.Module): 判别器
        """
        pass
    
    def pre_discriminator_step(self, real_batch, fake_batch, generator, discriminator):
        """
        判别器步骤前的准备。
        
        Args:
            real_batch (Tensor): 真实样本批次
            fake_batch (Tensor): 生成样本批次
            generator (nn.Module): 生成器
            discriminator (nn.Module): 判别器
        """
        pass
    
    def generator_loss(self, real_batch, fake_batch, discr_real_pred, 
                       discr_fake_pred, mask=None):
        """
        计算生成器损失。
        
        Args:
            real_batch (Tensor): 真实样本
            fake_batch (Tensor): 生成样本
            discr_real_pred (Tensor): 判别器对真实的预测
            discr_fake_pred (Tensor): 判别器对生成的预测
            mask (Tensor, optional): 掩码
        
        Returns:
            tuple: (损失值, 日志字典)
        """
        pass
    
    def discriminator_loss(self, real_batch, fake_batch, discr_real_pred,
                           discr_fake_pred, mask=None):
        """
        计算判别器损失。
        
        Args:
            real_batch (Tensor): 真实样本
            fake_batch (Tensor): 生成样本
            discr_real_pred (Tensor): 判别器对真实的预测
            discr_fake_pred (Tensor): 判别器对生成的预测
            mask (Tensor, optional): 掩码
        
        Returns:
            tuple: (损失值, 日志字典)
        """
        pass
```

#### `NonSaturatingWithR1(BaseAdversarialLoss)`

```python
class NonSaturatingWithR1(BaseAdversarialLoss):
    """
    非饱和对抗损失 + R1梯度惩罚。
    
    使用softplus损失和R1正则化训练GAN。
    
    Args:
        gp_coef (float): 梯度惩罚系数
        weight (float): 损失权重
        mask_as_fake_target (bool): 是否将掩码作为假目标
        allow_scale_mask (bool): 是否允许缩放掩码
        mask_scale_mode (str): 掩码缩放模式
        extra_mask_weight_for_gen (float): 生成器额外掩码权重
        use_unmasked_for_gen (bool): 生成器是否使用未掩码区域
        use_unmasked_for_discr (bool): 判别器是否使用未掩码区域
    """
    
    def __init__(self, gp_coef=5, weight=1, mask_as_fake_target=False,
                 allow_scale_mask=False, mask_scale_mode='nearest',
                 extra_mask_weight_for_gen=0, use_unmasked_for_gen=True,
                 use_unmasked_for_discr=True):
        """初始化R1对抗损失。"""
        pass
    
    def generator_loss(self, real_batch, fake_batch, discr_real_pred,
                       discr_fake_pred, mask=None):
        """计算生成器损失。"""
        pass
    
    def discriminator_loss(self, real_batch, fake_batch, discr_real_pred,
                           discr_fake_pred, mask=None):
        """计算判别器损失（含R1惩罚）。"""
        pass
```

### 核心函数

#### `make_r1_gp(discr_real_pred, real_batch)`

```python
def make_r1_gp(discr_real_pred, real_batch):
    """
    计算R1梯度惩罚。
    
    Args:
        discr_real_pred (Tensor): 判别器对真实样本的预测
        real_batch (Tensor): 真实样本批次
    
    Returns:
        Tensor: R1梯度惩罚值
    """
    pass
```

#### `make_discrim_loss(kind, **kwargs)`

```python
def make_discrim_loss(kind, **kwargs):
    """
    创建判别器损失。
    
    Args:
        kind (str): 损失类型 ('r1', 'bce')
        **kwargs: 损失参数
    
    Returns:
        BaseAdversarialLoss: 对抗损失实例
    """
    pass
```

---

## style_loss.py

风格损失。

### 核心类

#### `VGG19(nn.Module)`

```python
class VGG19(nn.Module):
    """
    VGG19特征提取器。
    
    提取VGG19各阶段的ReLU激活特征。
    """
    
    def __init__(self):
        """初始化VGG19特征提取器。"""
        pass
    
    def forward(self, x):
        """
        提取多尺度特征。
        
        Args:
            x (Tensor): 输入图像
        
        Returns:
            dict: 各层ReLU特征 {'relu1_1', 'relu1_2', ...}
        """
        pass
```

#### `PerceptualLoss(nn.Module)`

```python
class PerceptualLoss(nn.Module):
    """
    感知/风格损失。
    
    基于VGG19特征的L1感知损失。
    
    Args:
        weights (list): 各层权重，默认[1.0, 1.0, 1.0, 1.0, 1.0]
    """
    
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        """初始化感知损失。"""
        pass
    
    def __call__(self, x, y):
        """
        计算感知损失。
        
        Args:
            x (Tensor): 预测图像
            y (Tensor): 目标图像
        
        Returns:
            float: 感知损失值
        """
        pass
```

---

## feature_matching.py

特征匹配损失。

### 核心函数

#### `masked_l2_loss(pred, target, mask, weight_known, weight_missing)`

```python
def masked_l2_loss(pred, target, mask, weight_known, weight_missing):
    """
    带掩码的L2损失。
    
    对已知区域和缺失区域使用不同权重。
    
    Args:
        pred (Tensor): 预测值
        target (Tensor): 目标值
        mask (Tensor): 掩码
        weight_known (float): 已知区域权重
        weight_missing (float): 缺失区域权重
    
    Returns:
        Tensor: 加权L2损失
    """
    pass
```

#### `masked_l1_loss(pred, target, mask, weight_known, weight_missing)`

```python
def masked_l1_loss(pred, target, mask, weight_known, weight_missing):
    """
    带掩码的L1损失。
    
    Args:
        pred (Tensor): 预测值
        target (Tensor): 目标值
        mask (Tensor): 掩码
        weight_known (float): 已知区域权重
        weight_missing (float): 缺失区域权重
    
    Returns:
        Tensor: 加权L1损失
    """
    pass
```

#### `feature_matching_loss(fake_features, target_features, mask=None)`

```python
def feature_matching_loss(fake_features, target_features, mask=None):
    """
    特征匹配损失。
    
    计算生成图像与目标图像在判别器中间层特征的差异。
    
    Args:
        fake_features (list[Tensor]): 生成图像的特征列表
        target_features (list[Tensor]): 目标图像的特征列表
        mask (Tensor, optional): 掩码
    
    Returns:
        Tensor: 特征匹配损失
    """
    pass
```

---

## distance_weighting.py

距离加权掩码。

### 核心函数

#### `dummy_distance_weighter(real_img, pred_img, mask)`

```python
def dummy_distance_weighter(real_img, pred_img, mask):
    """
    虚拟距离加权器，直接返回原掩码。
    
    Args:
        real_img (Tensor): 真实图像
        pred_img (Tensor): 预测图像
        mask (Tensor): 输入掩码
    
    Returns:
        Tensor: 原掩码
    """
    pass
```

#### `get_gauss_kernel(kernel_size, width_factor=1)`

```python
def get_gauss_kernel(kernel_size, width_factor=1):
    """
    生成高斯核。
    
    Args:
        kernel_size (int): 核大小
        width_factor (float): 宽度因子
    
    Returns:
        Tensor: 2D高斯核
    """
    pass
```

### 核心类

#### `BlurMask(nn.Module)`

```python
class BlurMask(nn.Module):
    """
    掩码模糊模块。
    
    使用高斯滤波模糊掩码边缘。
    
    Args:
        kernel_size (int): 高斯核大小，默认5
        width_factor (float): 宽度因子，默认1
    """
    
    def __init__(self, kernel_size=5, width_factor=1):
        """初始化模糊模块。"""
        pass
    
    def forward(self, real_img, pred_img, mask):
        """
        模糊掩码。
        
        Args:
            real_img (Tensor): 真实图像
            pred_img (Tensor): 预测图像
            mask (Tensor): 输入掩码
        
        Returns:
            Tensor: 模糊后的掩码
        """
        pass
```

#### `EmulatedEDTMask(nn.Module)`

```python
class EmulatedEDTMask(nn.Module):
    """
    模拟欧氏距离变换掩码。
    
    通过膨胀和模糊模拟距离变换效果。
    
    Args:
        dilate_kernel_size (int): 膨胀核大小，默认5
        blur_kernel_size (int): 模糊核大小，默认5
        width_factor (float): 宽度因子，默认1
    """
    
    def __init__(self, dilate_kernel_size=5, blur_kernel_size=5, width_factor=1):
        """初始化EDT掩码模块。"""
        pass
    
    def forward(self, real_img, pred_img, mask):
        """生成EDT加权掩码。"""
        pass
```

#### `PropagatePerceptualSim(nn.Module)`

```python
class PropagatePerceptualSim(nn.Module):
    """
    感知相似性传播掩码。
    
    基于VGG特征的感知相似性传播掩码边界。
    
    Args:
        level (int): VGG特征层级别，默认2
        max_iters (int): 最大迭代次数，默认10
        temperature (float): 相似性温度参数，默认500
        erode_mask_size (int): 掩码腐蚀核大小，默认3
    """
    
    def __init__(self, level=2, max_iters=10, temperature=500, erode_mask_size=3):
        """初始化感知传播模块。"""
        pass
    
    def forward(self, real_img, pred_img, mask):
        """
        传播掩码边界。
        
        Args:
            real_img (Tensor): 真实图像
            pred_img (Tensor): 预测图像
            mask (Tensor): 输入掩码
        
        Returns:
            Tensor: 传播后的掩码
        """
        pass
```

### 核心函数

#### `make_mask_distance_weighter(kind='none', **kwargs)`

```python
def make_mask_distance_weighter(kind='none', **kwargs):
    """
    创建掩码距离加权器。
    
    Args:
        kind (str): 加权器类型 ('none', 'blur', 'edt', 'pps')
        **kwargs: 加权器参数
    
    Returns:
        callable: 掩码加权函数或模块
    """
    pass
```

---

## segmentation.py

分割损失。

### 核心类

#### `CrossEntropy2d(nn.Module)`

```python
class CrossEntropy2d(nn.Module):
    """
    2D交叉熵损失。
    
    用于语义分割任务的交叉熵损失。
    
    Args:
        reduction (str): 损失缩减方式，默认'mean'
        ignore_label (int): 忽略标签，默认255
        weights (str, optional): 类别权重名称
    """
    
    def __init__(self, reduction='mean', ignore_label=255, weights=None):
        """初始化交叉熵损失。"""
        pass
    
    def forward(self, predict, target):
        """
        计算交叉熵损失。
        
        Args:
            predict (Tensor): 预测结果，形状 (N, C, H, W)
            target (Tensor): 目标标签，形状 (N, 1, H, W)
        
        Returns:
            Tensor: 交叉熵损失
        """
        pass
```

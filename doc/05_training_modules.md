# Training 网络模块文档

## base.py

基础模块和工具函数。

### 核心类

#### `BaseDiscriminator(nn.Module)`

```python
class BaseDiscriminator(nn.Module):
    """
    判别器基类。
    
    定义判别器的接口，支持特征匹配损失。
    """
    
    @abc.abstractmethod
    def forward(self, x):
        """
        预测分数并获取中间激活。
        
        Args:
            x (Tensor): 输入图像
        
        Returns:
            tuple: (分数, 中间激活列表)
        """
        pass
```

#### `SimpleMultiStepGenerator(nn.Module)`

```python
class SimpleMultiStepGenerator(nn.Module):
    """
    简单多步生成器。
    
    顺序执行多个生成步骤，每步输出与输入拼接。
    
    Args:
        steps (list[nn.Module]): 生成步骤列表
    """
    
    def __init__(self, steps):
        """初始化多步生成器。"""
        pass
    
    def forward(self, x):
        """
        顺序执行各步骤。
        
        Args:
            x (Tensor): 输入
        
        Returns:
            Tensor: 所有步骤输出的拼接
        """
        pass
```

### 核心函数

#### `get_conv_block_ctor(kind='default')`

```python
def get_conv_block_ctor(kind='default'):
    """
    获取卷积块构造器。
    
    Args:
        kind (str): 卷积类型 ('default', 'depthwise', 'multidilated')
    
    Returns:
        nn.Module: 卷积层类
    """
    pass
```

#### `get_norm_layer(kind='bn')`

```python
def get_norm_layer(kind='bn'):
    """
    获取归一化层。
    
    Args:
        kind (str): 归一化类型 ('bn', 'in')
    
    Returns:
        nn.Module: 归一化层类
    """
    pass
```

#### `get_activation(kind='tanh')`

```python
def get_activation(kind='tanh'):
    """
    获取激活函数。
    
    Args:
        kind (str): 激活类型 ('tanh', 'sigmoid', False)
    
    Returns:
        nn.Module: 激活函数
    """
    pass
```

#### `deconv_factory(kind, ngf, mult, norm_layer, activation, max_features)`

```python
def deconv_factory(kind, ngf, mult, norm_layer, activation, max_features):
    """
    创建反卷积块。
    
    Args:
        kind (str): 反卷积类型 ('convtranspose', 'bilinear')
        ngf (int): 基础特征数
        mult (int): 特征数乘数
        norm_layer: 归一化层
        activation: 激活函数
        max_features (int): 最大特征数
    
    Returns:
        list: 反卷积层列表
    """
    pass
```

---

## ffc.py / ffc0.py

Fast Fourier Convolution 模块。

### 核心类

#### `FourierUnit(nn.Module)`

```python
class FourierUnit(nn.Module):
    """
    傅里叶单元。
    
    在频域执行卷积操作。
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        groups (int): 分组卷积数，默认1
        spatial_scale_factor (float, optional): 空间缩放因子
        spatial_scale_mode (str): 空间缩放模式，默认'bilinear'
        spectral_pos_encoding (bool): 是否使用频谱位置编码
        use_se (bool): 是否使用SE模块
        se_kwargs (dict): SE模块参数
        ffc3d (bool): 是否使用3D FFT
        fft_norm (str): FFT归一化模式，默认'ortho'
    """
    
    def __init__(self, in_channels, out_channels, groups=1, 
                 spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None,
                 ffc3d=False, fft_norm='ortho'):
        """初始化傅里叶单元。"""
        pass
    
    def forward(self, x):
        """
        频域卷积前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 频域处理后的特征
        """
        pass
```

#### `SpectralTransform(nn.Module)`

```python
class SpectralTransform(nn.Module):
    """
    频谱变换模块。
    
    包含下采样、1x1卷积、傅里叶单元和局部傅里叶单元。
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        stride (int): 步长，默认1
        groups (int): 分组数，默认1
        enable_lfu (bool): 是否启用局部傅里叶单元，默认True
    """
    
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        """初始化频谱变换。"""
        pass
    
    def forward(self, x):
        """
        频谱变换前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 变换后的特征
        """
        pass
```

#### `FFC(nn.Module)`

```python
class FFC(nn.Module):
    """
    Fast Fourier Convolution 模块。
    
    将输入分为局部和全局通道，分别处理。
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        ratio_gin (float): 输入全局通道比例，默认0.5
        ratio_gout (float): 输出全局通道比例，默认0.5
        stride (int): 步长，默认1
        padding (int): 填充，默认0
        dilation (int): 膨胀率，默认1
        groups (int): 分组数，默认1
        bias (bool): 是否使用偏置
        enable_lfu (bool): 是否启用LFU，默认True
        padding_type (str): 填充类型，默认'reflect'
        gated (bool): 是否使用门控
    """
    
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.5, ratio_gout=0.5, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        """初始化FFC模块。"""
        pass
    
    def forward(self, x):
        """
        FFC前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            tuple: (局部特征, 全局特征)
        """
        pass
```

#### `FFC_BN_ACT(nn.Module)`

```python
class FFC_BN_ACT(nn.Module):
    """
    FFC + 批归一化 + 激活。
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        ratio_gin (float): 输入全局比例
        ratio_gout (float): 输出全局比例
        stride (int): 步长
        padding (int): 填充
        dilation (int): 膨胀率
        groups (int): 分组数
        bias (bool): 是否偏置
        norm_layer: 归一化层
        activation_layer: 激活层
        padding_type (str): 填充类型
        enable_lfu (bool): 是否启用LFU
    """
    
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.5, ratio_gout=0.5, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.Identity, padding_type='reflect', enable_lfu=True, **kwargs):
        """初始化FFC-BN-ACT。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            tuple: (局部特征, 全局特征)
        """
        pass
```

#### `FFCResnetBlock(nn.Module)`

```python
class FFCResnetBlock(nn.Module):
    """
    FFC ResNet块。
    
    双卷积残差块，使用FFC-BN-ACT。
    
    Args:
        dim (int): 通道数
        padding_type (str): 填充类型
        norm_layer: 归一化层
        activation_layer: 激活层
        dilation (int): 膨胀率，默认1
        spatial_transform_kwargs (dict, optional): 空间变换参数
        inline (bool): 是否内联处理
    """
    
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU,
                 dilation=1, spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        """初始化FFC ResNet块。"""
        pass
    
    def forward(self, x_l, x_g):
        """
        前向传播。
        
        Args:
            x_l (Tensor): 局部特征
            x_g (Tensor): 全局特征
        
        Returns:
            tuple: (局部残差输出, 全局残差输出)
        """
        pass
```

#### `FFCResNetGenerator(nn.Module)`

```python
class FFCResNetGenerator(nn.Module):
    """
    FFC ResNet生成器。
    
    基于FFC的图像生成网络。
    
    Args:
        input_nc (int): 输入通道数
        output_nc (int): 输出通道数
        ngf (int): 基础特征数，默认64
        n_downsampling (int): 下采样次数，默认3
        n_blocks (int): 残差块数量，默认9
        norm_layer: 归一化层
        padding_type (str): 填充类型
        activation_layer: 激活层
        up_norm_layer: 上采样归一化层
        up_activation: 上采样激活
        init_conv_kwargs (dict): 初始卷积参数
        downsample_conv_kwargs (dict): 下采样卷积参数
        resnet_conv_kwargs (dict): ResNet卷积参数
        spatial_transform_layers (list): 空间变换层索引
        spatial_transform_kwargs (dict): 空间变换参数
        add_out_act (bool): 是否添加输出激活
        max_features (int): 最大特征数，默认1024
        out_ffc (bool): 是否FFC输出
        out_ffc_kwargs (dict): 输出FFC参数
    """
    
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        """初始化FFC ResNet生成器。"""
        pass
    
    def forward(self, input):
        """
        生成图像。
        
        Args:
            input (Tensor): 输入
        
        Returns:
            Tensor: 生成图像
        """
        pass
```

---

## pix2pixhd.py

Pix2PixHD 网络架构。

### 核心类

#### `ResnetBlock(nn.Module)`

```python
class ResnetBlock(nn.Module):
    """
    ResNet块。
    
    标准残差块，支持深度可分离卷积和膨胀卷积。
    
    Args:
        dim (int): 通道数
        padding_type (str): 填充类型
        norm_layer: 归一化层
        activation: 激活函数
        use_dropout (bool): 是否使用dropout
        conv_kind (str): 卷积类型
        dilation (int): 膨胀率
        in_dim (int, optional): 输入维度（用于维度变化）
        groups (int): 分组数
        second_dilation (int): 第二个卷积的膨胀率
    """
    
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True),
                 use_dropout=False, conv_kind='default', dilation=1, 
                 in_dim=None, groups=1, second_dilation=None):
        """初始化ResNet块。"""
        pass
    
    def build_conv_block(self, dim, padding_type, norm_layer, activation, 
                         use_dropout, conv_kind, dilation, in_dim, groups, second_dilation):
        """构建卷积块。"""
        pass
    
    def forward(self, x):
        """前向传播。"""
        pass
```

#### `GlobalGenerator(nn.Module)`

```python
class GlobalGenerator(nn.Module):
    """
    全局生成器。
    
    Pix2PixHD的全局生成器，支持膨胀卷积和FFC。
    
    Args:
        input_nc (int): 输入通道数
        output_nc (int): 输出通道数
        ngf (int): 基础特征数，默认64
        n_downsampling (int): 下采样次数，默认3
        n_blocks (int): 残差块数，默认9
        norm_layer: 归一化层
        padding_type (str): 填充类型
        conv_kind (str): 卷积类型
        activation: 激活函数
        up_norm_layer: 上采样归一化
        affine (bool): 是否仿射变换
        up_activation: 上采样激活
        dilated_blocks_n (int): 末端膨胀块数
        dilated_blocks_n_start (int): 起始膨胀块数
        dilated_blocks_n_middle (int): 中间膨胀块数
        add_out_act (bool): 添加输出激活
        max_features (int): 最大特征数
        is_resblock_depthwise (bool): ResNet块是否深度可分离
        ffc_positions (list): FFC位置列表
        ffc_kwargs (dict): FFC参数
        dilation (int): 膨胀率
        second_dilation (int): 第二个膨胀率
        dilation_block_kind (str): 膨胀块类型
        multidilation_kwargs (dict): 多膨胀参数
    """
    
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', conv_kind='default',
                 activation=nn.ReLU(True), up_norm_layer=nn.BatchNorm2d, affine=None,
                 up_activation=nn.ReLU(True), dilated_blocks_n=0, dilated_blocks_n_start=0,
                 dilated_blocks_n_middle=0, add_out_act=True, max_features=1024,
                 is_resblock_depthwise=False, ffc_positions=None, ffc_kwargs={},
                 dilation=1, second_dilation=None, dilation_block_kind='simple', 
                 multidilation_kwargs={}):
        """初始化全局生成器。"""
        pass
    
    def forward(self, input):
        """生成图像。"""
        pass
```

#### `NLayerDiscriminator(BaseDiscriminator)`

```python
class NLayerDiscriminator(BaseDiscriminator):
    """
    N层PatchGAN判别器。
    
    Args:
        input_nc (int): 输入通道数
        ndf (int): 基础特征数，默认64
        n_layers (int): 判别器层数，默认3
        norm_layer: 归一化层
    """
    
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """初始化N层判别器。"""
        pass
    
    def get_all_activations(self, x):
        """获取所有层激活。"""
        pass
    
    def forward(self, x):
        """
        判别图像。
        
        Args:
            x (Tensor): 输入图像
        
        Returns:
            tuple: (预测分数, 中间激活列表)
        """
        pass
```

---

## multidilated_conv.py

多膨胀率卷积。

### 核心类

#### `MultidilatedConv(nn.Module)`

```python
class MultidilatedConv(nn.Module):
    """
    多膨胀率卷积。
    
    使用多个不同膨胀率的卷积并行处理。
    
    Args:
        in_dim (int): 输入通道数
        out_dim (int): 输出通道数
        kernel_size (int): 卷积核大小
        dilation_num (int): 膨胀率数量，默认3
        comb_mode (str): 组合模式 ('sum', 'cat_out', 'cat_in', 'cat_both')
        equal_dim (bool): 是否等分通道
        shared_weights (bool): 是否共享权重
        padding (int): 填充
        min_dilation (int): 最小膨胀率
        shuffle_in_channels (bool): 是否打乱输入通道
        use_depthwise (bool): 是否使用深度可分离卷积
    """
    
    def __init__(self, in_dim, out_dim, kernel_size, dilation_num=3, 
                 comb_mode='sum', equal_dim=True, shared_weights=False,
                 padding=1, min_dilation=1, shuffle_in_channels=False, 
                 use_depthwise=False, **kwargs):
        """初始化多膨胀卷积。"""
        pass
    
    def forward(self, x):
        """
        多膨胀卷积前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 输出特征
        """
        pass
```

---

## depthwise_sep_conv.py

深度可分离卷积。

### 核心类

#### `DepthWiseSeperableConv(nn.Module)`

```python
class DepthWiseSeperableConv(nn.Module):
    """
    深度可分离卷积。
    
    先深度卷积，再逐点卷积，减少参数量。
    
    Args:
        in_dim (int): 输入通道数
        out_dim (int): 输出通道数
        *args: 卷积参数
        **kwargs: 卷积关键字参数
    
    Attributes:
        depthwise (Conv2d): 深度卷积（groups=in_dim）
        pointwise (Conv2d): 逐点卷积（1x1）
    """
    
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        """初始化深度可分离卷积。"""
        pass
    
    def forward(self, x):
        """
        深度可分离卷积前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 输出特征
        """
        pass
```

---

## squeeze_excitation.py

Squeeze-and-Excitation 模块。

### 核心类

#### `SELayer(nn.Module)`

```python
class SELayer(nn.Module):
    """
    Squeeze-and-Excitation层。
    
    通道注意力机制，学习通道重要性权重。
    
    Args:
        channel (int): 通道数
        reduction (int): 降维比率，默认16
    
    Attributes:
        avg_pool (AdaptiveAvgPool2d): 全局平均池化
        fc (Sequential): 全连接层（降维-ReLU-升维-Sigmoid）
    """
    
    def __init__(self, channel, reduction=16):
        """初始化SE层。"""
        pass
    
    def forward(self, x):
        """
        SE前向传播。
        
        Args:
            x (Tensor): 输入特征，形状 (B, C, H, W)
        
        Returns:
            Tensor: 通道加权后的特征
        """
        pass
```

---

## spatial_transform.py

可学习空间变换。

### 核心类

#### `LearnableSpatialTransformWrapper(nn.Module)`

```python
class LearnableSpatialTransformWrapper(nn.Module):
    """
    可学习空间变换包装器。
    
    对输入进行随机旋转的空间变换。
    
    Args:
        impl (nn.Module): 要包装的模块
        pad_coef (float): 填充系数，默认0.5
        angle_init_range (float): 角度初始化范围，默认80
        train_angle (bool): 是否训练角度，默认True
    """
    
    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        """初始化空间变换包装器。"""
        pass
    
    def forward(self, x):
        """
        空间变换前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 变换并恢复后的特征
        """
        pass
    
    def transform(self, x):
        """
        应用空间变换（旋转）。
        
        Args:
            x (Tensor): 输入
        
        Returns:
            Tensor: 变换后的特征
        """
        pass
    
    def inverse_transform(self, y_padded_rotated, orig_x):
        """
        逆空间变换（恢复）。
        
        Args:
            y_padded_rotated (Tensor): 变换后的特征
            orig_x (Tensor): 原始输入（用于获取尺寸）
        
        Returns:
            Tensor: 恢复后的特征
        """
        pass
```

---

## multiscale.py

多尺度网络。

### 核心类

#### `ResNetHead(nn.Module)`

```python
class ResNetHead(nn.Module):
    """
    ResNet编码头。
    
    多尺度网络的编码部分。
    
    Args:
        input_nc (int): 输入通道数
        ngf (int): 基础特征数
        n_downsampling (int): 下采样次数
        n_blocks (int): 残差块数
        norm_layer: 归一化层
        padding_type (str): 填充类型
        conv_kind (str): 卷积类型
        activation: 激活函数
    """
    
    def __init__(self, input_nc, ngf=64, n_downsampling=3, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', 
                 conv_kind='default', activation=nn.ReLU(True)):
        """初始化ResNet头。"""
        pass
    
    def forward(self, input):
        """提取多尺度特征。"""
        pass
```

#### `ResNetTail(nn.Module)`

```python
class ResNetTail(nn.Module):
    """
    ResNet解码尾。
    
    多尺度网络的解码部分。
    
    Args:
        output_nc (int): 输出通道数
        ngf (int): 基础特征数
        n_downsampling (int): 下采样次数
        n_blocks (int): 残差块数
        norm_layer: 归一化层
        padding_type (str): 填充类型
        conv_kind (str): 卷积类型
        activation: 激活函数
        up_norm_layer: 上采样归一化
        up_activation: 上采样激活
        add_out_act (bool): 添加输出激活
        out_extra_layers_n (int): 额外输出层数
        add_in_proj (int, optional): 输入投影维度
    """
    
    def __init__(self, output_nc, ngf=64, n_downsampling=3, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', 
                 conv_kind='default', activation=nn.ReLU(True),
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 add_out_act=False, out_extra_layers_n=0, add_in_proj=None):
        """初始化ResNet尾。"""
        pass
    
    def forward(self, input, return_last_act=False):
        """
        解码特征。
        
        Args:
            input (Tensor): 输入特征
            return_last_act (bool): 是否返回最后的激活
        
        Returns:
            Tensor: 解码图像（或元组）
        """
        pass
```

#### `MultiscaleResNet(nn.Module)`

```python
class MultiscaleResNet(nn.Module):
    """
    多尺度ResNet。
    
    处理多分辨率输入并产生多分辨率输出。
    
    Args:
        input_nc (int): 输入通道数
        output_nc (int): 输出通道数
        ngf (int): 基础特征数
        n_downsampling (int): 下采样次数
        n_blocks_head (int): 头的残差块数
        n_blocks_tail (int): 尾的残差块数
        n_scales (int): 尺度数量
        norm_layer: 归一化层
        padding_type (str): 填充类型
        conv_kind (str): 卷积类型
        activation: 激活函数
        up_norm_layer: 上采样归一化
        up_activation: 上采样激活
        add_out_act (bool): 添加输出激活
        out_extra_layers_n (int): 额外输出层数
        out_cumulative (bool): 是否累加输出
        return_only_hr (bool): 是否只返回高分辨率
    """
    
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, 
                 n_blocks_head=2, n_blocks_tail=6, n_scales=3,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', 
                 conv_kind='default', activation=nn.ReLU(True),
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 add_out_act=False, out_extra_layers_n=0, 
                 out_cumulative=False, return_only_hr=False):
        """初始化多尺度ResNet。"""
        pass
    
    def forward(self, ms_inputs, smallest_scales_num=None):
        """
        多尺度前向传播。
        
        Args:
            ms_inputs (list[Tensor]): 多尺度输入（从高分辨率到低分辨率）
            smallest_scales_num (int, optional): 使用的最小尺度数量
        
        Returns:
            Tensor 或 list[Tensor]: 输出图像或列表
        """
        pass
```

---

## fake_fakes.py

虚假样本生成。

### 核心类

#### `FakeFakesGenerator`

```python
class FakeFakesGenerator:
    """
    虚假样本生成器。
    
    通过混合和增强生成虚假样本用于训练。
    
    Args:
        aug_proba (float): 增强概率，默认0.5
        img_aug_degree (float): 图像增强角度，默认30
        img_aug_translate (float): 图像增强平移，默认0.2
    
    Attributes:
        grad_aug (RandomAffine): 梯度增强
        img_aug (RandomAffine): 图像增强
    """
    
    def __init__(self, aug_proba=0.5, img_aug_degree=30, img_aug_translate=0.2):
        """初始化虚假样本生成器。"""
        pass
    
    def __call__(self, input_images, masks):
        """
        生成虚假样本。
        
        Args:
            input_images (Tensor): 输入图像批次
            masks (Tensor): 掩码批次
        
        Returns:
            tuple: (虚假样本, 混合掩码)
        """
        pass
    
    def _make_blend_target(self, input_images):
        """创建混合目标。"""
        pass
    
    def _fill_masks_with_gradient(self, masks):
        """用渐变填充掩码。"""
        pass
```

---

## __init__.py

模块工厂函数。

### 核心函数

#### `make_generator(config, kind, **kwargs)`

```python
def make_generator(config, kind, **kwargs):
    """
    创建生成器。
    
    Args:
        config: 配置对象
        kind (str): 生成器类型 ('pix2pixhd_multidilated', 'pix2pixhd_global', 'ffc_resnet')
        **kwargs: 生成器参数
    
    Returns:
        nn.Module: 生成器实例
    """
    pass
```

#### `make_discriminator(kind, **kwargs)`

```python
def make_discriminator(kind, **kwargs):
    """
    创建判别器。
    
    Args:
        kind (str): 判别器类型 ('pix2pixhd_nlayer_multidilated', 'pix2pixhd_nlayer')
        **kwargs: 判别器参数
    
    Returns:
        BaseDiscriminator: 判别器实例
    """
    pass
```

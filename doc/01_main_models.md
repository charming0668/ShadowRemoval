# 主模型模块文档

## model.py

主模型入口文件，整合阴影去除和图像增强网络。

### 核心类

#### `final_net(nn.Module)`

整体阴影去除网络，包含两个阶段：
1. 阴影去除模型 (remove_model)
2. 图像增强模型 (enhancement_model)

```python
class final_net(nn.Module):
    """
    最终的阴影去除网络，融合阴影去除和图像增强两个阶段。
    
    网络结构：
    - remove_model: fusion_net，用于去除阴影
    - enhancement_model: Restormer，用于图像细节增强
    
    Args:
        无初始化参数
    
    Attributes:
        remove_model (fusion_net): DWT分支与知识适应分支的融合网络
        enhancement_model (Restormer): 基于Transformer的图像增强网络
    """
    
    def __init__(self):
        """
        初始化final_net，创建两个子网络实例。
        """
        pass
    
    def forward(self, input, scale=0.05):
        """
        前向传播，执行阴影去除和图像增强。
        
        Args:
            input (Tensor): 输入带阴影的图像，形状为 (B, C, H, W)
            scale (float): 增强模型的残差缩放系数，默认为0.05
        
        Returns:
            Tensor: 去除阴影后的图像，形状为 (B, C, H, W)
        """
        pass
```

---

## model_convnext.py

ConvNeXt 架构的融合网络，结合 DWT 小波变换和知识适应分支。

### 核心函数

#### `dwt_init(x)`

```python
def dwt_init(x):
    """
    离散小波变换（DWT）初始化函数。
    
    对输入图像执行2D Haar小波变换，分解为LL（低频）、HL（水平高频）、
    LH（垂直高频）、HH（对角高频）四个子带。
    
    Args:
        x (Tensor): 输入图像张量，形状为 (B, C, H, W)
    
    Returns:
        tuple: (x_LL, x_high)，其中：
            - x_LL: 低频分量，形状为 (B, C, H/2, W/2)
            - x_high: 高频分量拼接，形状为 (B, C*3, H/2, W/2)
    """
    pass
```

### 核心类

#### `DWT(nn.Module)`

```python
class DWT(nn.Module):
    """
    离散小波变换模块。
    
    将输入图像分解为低频和高频分量。
    
    Attributes:
        requires_grad (bool): 不需要梯度计算，设为False
    """
    
    def __init__(self):
        """初始化DWT模块。"""
        pass
    
    def forward(self, x):
        """
        执行小波变换。
        
        Args:
            x (Tensor): 输入图像
        
        Returns:
            tuple: (低频分量, 高频分量)
        """
        pass
```

#### `DWT_transform(nn.Module)`

```python
class DWT_transform(nn.Module):
    """
    带卷积的小波变换模块。
    
    对DWT分解后的低频和高频分量分别应用1x1卷积进行通道调整。
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    """
    
    def __init__(self, in_channels, out_channels):
        """初始化变换模块。"""
        pass
    
    def forward(self, x):
        """
        执行小波变换和卷积。
        
        Args:
            x (Tensor): 输入特征图
        
        Returns:
            tuple: (dwt_low_frequency, dwt_high_frequency)
        """
        pass
```

#### `ConvNeXt0(nn.Module)`

```python
class ConvNeXt0(nn.Module):
    """
    ConvNeXt 编码器（完整版）。
    
    用于分类任务的ConvNeXt实现，包含4个下采样阶段。
    
    Args:
        block (nn.Module): 基础块类型（如Block）
        in_chans (int): 输入通道数，默认3
        num_classes (int): 分类类别数，默认1000
        depths (list): 每个阶段的块数量，默认[3, 3, 27, 3]
        dims (list): 每个阶段的特征维度，默认[256, 512, 1024, 2048]
        drop_path_rate (float): 随机深度比率，默认0
        layer_scale_init_value (float): 层缩放初始化值，默认1e-6
        head_init_scale (float): 分类头初始化缩放，默认1
    
    Attributes:
        downsample_layers (ModuleList): 下采样层
        stages (ModuleList): 各阶段的残差块
        norm (LayerNorm): 最终的归一化层
        head (Linear): 分类头
    """
    
    def __init__(self, block, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, 
                 head_init_scale=1.):
        """初始化ConvNeXt0编码器。"""
        pass
    
    def _init_weights(self, m):
        """
        权重初始化。
        
        Args:
            m (nn.Module): 要初始化的模块
        """
        pass
    
    def forward_features(self, x):
        """
        提取特征（不含分类头）。
        
        Args:
            x (Tensor): 输入图像
        
        Returns:
            Tensor: 全局平均池化后的特征
        """
        pass
    
    def forward(self, x):
        """
        完整前向传播（含分类）。
        
        Args:
            x (Tensor): 输入图像
        
        Returns:
            Tensor: 分类 logits
        """
        pass
```

#### `Block(nn.Module)`

```python
class Block(nn.Module):
    """
    ConvNeXt 基础块。
    
    实现深度可分离卷积 + 层归一化 + 逐点卷积的残差块。
    
    Args:
        dim (int): 输入通道数
        drop_path (float): 随机深度丢弃率，默认0
        layer_scale_init_value (float): 层缩放初始化值，默认1e-6
    
    Attributes:
        dwconv (Conv2d): 深度可分离卷积（7x7）
        norm (LayerNorm): 层归一化
        pwconv1 (Linear): 第一个逐点卷积（扩展4倍）
        act (GELU): GELU激活函数
        pwconv2 (Linear): 第二个逐点卷积（恢复原维度）
        gamma (Parameter): 可学习的层缩放参数
        drop_path (DropPath): 随机深度模块
    """
    
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        """初始化ConvNeXt块。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征，形状 (B, C, H, W)
        
        Returns:
            Tensor: 输出特征，形状 (B, C, H, W)
        """
        pass
```

#### `ConvNeXt(nn.Module)`

```python
class ConvNeXt(nn.Module):
    """
    ConvNeXt 编码器（特征提取版）。
    
    用于特征提取的ConvNeXt，返回多尺度特征。
    
    Args:
        block (nn.Module): 基础块类型
        in_chans (int): 输入通道数，默认3
        num_classes (int): 分类类别数，默认1000
        depths (list): 每个阶段的块数量
        dims (list): 每个阶段的特征维度
        drop_path_rate (float): 随机深度比率
        layer_scale_init_value (float): 层缩放初始化值
        head_init_scale (float): 分类头初始化缩放
    
    Returns:
        tuple: (x_layer1, x_layer2, out) 多尺度特征
    """
    
    def __init__(self, block, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, 
        """初始化特征提取ConvNeXt。"""
        pass
    
    def forward(self, x):
        """
        前向传播，返回多尺度特征。
        
        Args:
            x (Tensor): 输入图像
        
        Returns:
            tuple: (stage1特征, stage2特征, stage3特征)
        """
        pass
```

#### `LayerNorm(nn.Module)`

```python
class LayerNorm(nn.Module):
    """
    支持两种数据格式的层归一化。
    
    支持 channels_last (N, H, W, C) 和 channels_first (N, C, H, W) 格式。
    
    Args:
        normalized_shape (int): 归一化的特征维度
        eps (float): 数值稳定性常数，默认1e-6
        data_format (str): 数据格式，"channels_last" 或 "channels_first"
    """
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """初始化LayerNorm。"""
        pass
    
    def forward(self, x):
        """
        层归一化前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 归一化后的特征
        """
        pass
```

#### `PALayer(nn.Module)`

```python
class PALayer(nn.Module):
    """
    像素注意力层（Pixel Attention Layer）。
    
    通过学习空间注意力图来增强特征表示。
    
    Args:
        channel (int): 输入通道数
    """
    
    def __init__(self, channel):
        """初始化PA层。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 注意力加权后的特征
        """
        pass
```

#### `CALayer(nn.Module)`

```python
class CALayer(nn.Module):
    """
    通道注意力层（Channel Attention Layer）。
    
    类似SE-Net的通道注意力机制。
    
    Args:
        channel (int): 输入通道数
    """
    
    def __init__(self, channel):
        """初始化CA层。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 通道注意力加权后的特征
        """
        pass
```

#### `CP_Attention_block(nn.Module)`

```python
class CP_Attention_block(nn.Module):
    """
    通道-像素注意力块（Channel-Pixel Attention Block）。
    
    结合通道注意力和像素注意力的残差块。
    
    Args:
        conv (nn.Module): 卷积层类型
        dim (int): 通道数
        kernel_size (int): 卷积核大小
    """
    
    def __init__(self, conv, dim, kernel_size):
        """初始化CP注意力块。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 输出特征
        """
        pass
```

#### `knowledge_adaptation_convnext(nn.Module)`

```python
class knowledge_adaptation_convnext(nn.Module):
    """
    知识适应ConvNeXt解码器。
    
    使用ConvNeXt作为编码器，结合注意力机制的上采样解码器。
    
    Attributes:
        encoder (ConvNeXt): ConvNeXt编码器
        up_block (PixelShuffle): 像素洗牌上采样
        attention0-4 (CP_Attention_block): 多级注意力块
        tail (Sequential): 输出卷积层
    """
    
    def __init__(self):
        """初始化知识适应网络。"""
        pass
    
    def forward(self, input):
        """
        前向传播。
        
        Args:
            input (Tensor): 输入图像
        
        Returns:
            Tensor: 输出特征（用于后续融合）
        """
        pass
```

#### `dwt_ffc_UNet2(nn.Module)`

```python
class dwt_ffc_UNet2(nn.Module):
    """
    DWT-FFC UNet网络。
    
    结合离散小波变换和快速傅里叶卷积的U型网络。
    
    Args:
        output_nc (int): 输出通道数，默认3
        nf (int): 基础特征数，默认16
    
    Attributes:
        DWT_down_0-4 (DWT_transform): 多级小波下采样
        layer1-6 (Sequential): 编码器层
        dlayer1-6 (Sequential): 解码器层
        FFCResNet (myFFCResblock): FFC残差块
    """
    
    def __init__(self, output_nc=3, nf=16):
        """初始化DWT-FFC UNet。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入图像
        
        Returns:
            Tensor: 去阴影后的图像
        """
        pass
```

#### `fusion_net(nn.Module)`

```python
class fusion_net(nn.Module):
    """
    融合网络。
    
    融合DWT分支和知识适应分支的输出。
    
    Attributes:
        dwt_branch (dwt_ffc_UNet2): DWT-FFC分支
        knowledge_adaptation_branch (knowledge_adaptation_convnext): 知识适应分支
        fusion (Sequential): 融合卷积层
    """
    
    def __init__(self):
        """初始化融合网络。"""
        pass
    
    def forward(self, input):
        """
        前向传播。
        
        Args:
            input (Tensor): 输入带阴影图像
        
        Returns:
            Tensor: 融合后的去阴影图像
        """
        pass
```

#### `Discriminator(nn.Module)`

```python
class Discriminator(nn.Module):
    """
    判别器网络。
    
    用于对抗训练的PatchGAN风格判别器。
    """
    
    def __init__(self):
        """初始化判别器。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入图像，形状 (B, 3, H, W)
        
        Returns:
            Tensor: 判别分数，形状 (B,)
        """
        pass
```

---

## myFFCResblock0.py

自定义 FFC ResNet 块包装器。

#### `myFFCResblock(nn.Module)`

```python
class myFFCResblock(nn.Module):
    """
    自定义FFC残差块。
    
    包装FFC_BN_ACT和FFCResnetBlock，提供简洁接口。
    
    Args:
        input_nc (int): 输入通道数
        output_nc (int): 输出通道数
        n_blocks (int): FFC残差块数量，默认2
        norm_layer (nn.Module): 归一化层，默认BatchNorm2d
        padding_type (str): 填充类型，默认'reflect'
        activation_layer (nn.Module): 激活函数，默认ReLU
        resnet_conv_kwargs (dict): FFC卷积参数
        spatial_transform_layers (list): 空间变换层索引
        spatial_transform_kwargs (dict): 空间变换参数
        add_out_act (bool): 是否添加输出激活
        max_features (int): 最大特征数
        out_ffc (bool): 是否使用FFC输出
        out_ffc_kwargs (dict): 输出FFC参数
    """
    
    def __init__(self, input_nc, output_nc, n_blocks=2, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 resnet_conv_kwargs={}, spatial_transform_layers=None, 
                 spatial_transform_kwargs={}, add_out_act=True, 
                 max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        """初始化FFC残差块。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 输出特征
        """
        pass
```

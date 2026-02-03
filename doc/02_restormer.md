# Restormer 模块文档

## restormer_arch.py

Restormer 图像恢复网络架构，使用傅里叶频域注意力机制。

### 核心函数

#### `to_3d(x)`

```python
def to_3d(x):
    """
    将4D张量转换为3D，用于Transformer处理。
    
    Args:
        x (Tensor): 输入张量，形状 (B, C, H, W)
    
    Returns:
        Tensor: 重塑后的张量，形状 (B, H*W, C)
    """
    pass
```

#### `to_4d(x, h, w)`

```python
def to_4d(x, h, w):
    """
    将3D张量恢复为4D。
    
    Args:
        x (Tensor): 输入张量，形状 (B, H*W, C)
        h (int): 目标高度
        w (int): 目标宽度
    
    Returns:
        Tensor: 重塑后的张量，形状 (B, C, H, W)
    """
    pass
```

### 核心类

#### `BiasFree_LayerNorm(nn.Module)`

```python
class BiasFree_LayerNorm(nn.Module):
    """
    无偏置的层归一化。
    
    Args:
        normalized_shape (int): 归一化的特征维度
    """
    
    def __init__(self, normalized_shape):
        """初始化无偏置层归一化。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 归一化后的特征
        """
        pass
```

#### `WithBias_LayerNorm(nn.Module)`

```python
class WithBias_LayerNorm(nn.Module):
    """
    带偏置的层归一化。
    
    Args:
        normalized_shape (int): 归一化的特征维度
    """
    
    def __init__(self, normalized_shape):
        """初始化带偏置层归一化。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 归一化后的特征
        """
        pass
```

#### `LayerNorm(nn.Module)`

```python
class LayerNorm(nn.Module):
    """
    层归一化包装器。
    
    支持BiasFree和WithBias两种模式。
    
    Args:
        dim (int): 特征维度
        LayerNorm_type (str): 层归一化类型，'BiasFree' 或 'WithBias'
    """
    
    def __init__(self, dim, LayerNorm_type):
        """初始化层归一化。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征，形状 (B, C, H, W)
        
        Returns:
            Tensor: 归一化后的特征
        """
        pass
```

#### `FeedForward(nn.Module)`

```python
class FeedForward(nn.Module):
    """
    门控深度可分离前馈网络（GDFN）。
    
    Args:
        dim (int): 输入维度
        ffn_expansion_factor (float): 前馈网络扩展因子
        bias (bool): 是否使用偏置
    """
    
    def __init__(self, dim, ffn_expansion_factor, bias):
        """初始化前馈网络。"""
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

#### `Attention(nn.Module)`

```python
class Attention(nn.Module):
    """
    傅里叶频域自注意力（FSAS）。
    
    使用FFT在频域计算自注意力，替代传统的MDTA。
    
    Args:
        dim (int): 输入维度
        num_heads (int): 注意力头数
        bias (bool): 是否使用偏置
    
    Attributes:
        patch_size (int): 分块大小，默认2
    """
    
    def __init__(self, dim, num_heads, bias):
        """初始化频域注意力。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 注意力输出
        """
        pass
```

#### `TransformerBlock(nn.Module)`

```python
class TransformerBlock(nn.Module):
    """
    Transformer块。
    
    包含层归一化、频域注意力和门控前馈网络。
    
    Args:
        dim (int): 输入维度
        num_heads (int): 注意力头数
        ffn_expansion_factor (float): 前馈扩展因子
        bias (bool): 是否使用偏置
        LayerNorm_type (str): 层归一化类型
    """
    
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        """初始化Transformer块。"""
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

#### `OverlapPatchEmbed(nn.Module)`

```python
class OverlapPatchEmbed(nn.Module):
    """
    重叠图像块嵌入。
    
    使用3x3卷积进行重叠分块嵌入。
    
    Args:
        in_c (int): 输入通道数，默认3
        embed_dim (int): 嵌入维度，默认48
        bias (bool): 是否使用偏置
    """
    
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        """初始化重叠分块嵌入。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入图像
        
        Returns:
            Tensor: 嵌入特征
        """
        pass
```

#### `Downsample(nn.Module)`

```python
class Downsample(nn.Module):
    """
    下采样模块。
    
    使用PixelUnshuffle进行下采样。
    
    Args:
        n_feat (int): 特征通道数
    """
    
    def __init__(self, n_feat):
        """初始化下采样。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 下采样后的特征
        """
        pass
```

#### `Upsample(nn.Module)`

```python
class Upsample(nn.Module):
    """
    上采样模块。
    
    使用PixelShuffle进行上采样。
    
    Args:
        n_feat (int): 特征通道数
    """
    
    def __init__(self, n_feat):
        """初始化上采样。"""
        pass
    
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征
        
        Returns:
            Tensor: 上采样后的特征
        """
        pass
```

#### `Restormer(nn.Module)`

```python
class Restormer(nn.Module):
    """
    Restormer图像恢复网络。
    
    基于Transformer的高效高分辨率图像恢复网络。
    
    Args:
        inp_channels (int): 输入通道数，默认3
        out_channels (int): 输出通道数，默认3
        dim (int): 基础维度，默认48
        num_blocks (list): 各阶段的Transformer块数量，默认[4,6,6,8]
        num_refinement_blocks (int): 细化块数量，默认4
        heads (list): 各阶段的注意力头数，默认[1,2,4,8]
        ffn_expansion_factor (float): 前馈扩展因子，默认2.66
        bias (bool): 是否使用偏置，默认False
        LayerNorm_type (str): 层归一化类型，默认'WithBias'
        dual_pixel_task (bool): 双像素去模糊任务标志
    
    Attributes:
        patch_embed: 重叠分块嵌入
        encoder_level1-3: 编码器阶段
        latent: 潜在空间
        decoder_level1-3: 解码器阶段
        refinement: 细化模块
    """
    
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4,6,6,8], num_refinement_blocks=4,
                 heads=[1,2,4,8], ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias',
                 dual_pixel_task=False):
        """初始化Restormer网络。"""
        pass
    
    def forward(self, inp_img):
        """
        前向传播。
        
        Args:
            inp_img (Tensor): 输入图像，形状 (B, 3, H, W)
        
        Returns:
            Tensor: 恢复后的图像，形状 (B, 3, H, W)
        """
        pass
```

---

## arch_util.py

架构工具函数和模块。

### 核心函数

#### `default_init_weights(module_list, scale=1, bias_fill=0, **kwargs)`

```python
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """
    初始化网络权重。
    
    使用Kaiming正态初始化卷积和线性层权重。
    
    Args:
        module_list (list[nn.Module] | nn.Module): 要初始化的模块
        scale (float): 权重缩放因子，默认1
        bias_fill (float): 偏置填充值，默认0
        kwargs (dict): 初始化函数的其他参数
    """
    pass
```

#### `make_layer(basic_block, num_basic_block, **kwarg)`

```python
def make_layer(basic_block, num_basic_block, **kwarg):
    """
    通过堆叠相同块创建层。
    
    Args:
        basic_block (nn.Module): 基础块类
        num_basic_block (int): 块的数量
    
    Returns:
        nn.Sequential: 堆叠的块
    """
    pass
```

### 核心类

#### `ResidualBlockNoBN(nn.Module)`

```python
class ResidualBlockNoBN(nn.Module):
    """
    无BN的残差块。
    
    结构: Conv-ReLU-Conv + 残差连接
    
    Args:
        num_feat (int): 特征通道数，默认64
        res_scale (float): 残差缩放，默认1
        pytorch_init (bool): 是否使用PyTorch默认初始化
    """
    
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        """初始化残差块。"""
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

#### `Upsample(nn.Sequential)`

```python
class Upsample(nn.Sequential):
    """
    上采样模块。
    
    支持2^n和3倍上采样。
    
    Args:
        scale (int): 缩放因子
        num_feat (int): 特征通道数
    """
    
    def __init__(self, scale, num_feat):
        """初始化上采样模块。"""
        pass
```

#### `flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True)`

```python
def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """
    使用光流扭曲图像或特征图。
    
    Args:
        x (Tensor): 输入张量，形状 (N, C, H, W)
        flow (Tensor): 光流场，形状 (N, H, W, 2)
        interp_mode (str): 插值模式，默认'bilinear'
        padding_mode (str): 填充模式，默认'zeros'
        align_corners (bool): 是否对齐角点
    
    Returns:
        Tensor: 扭曲后的图像/特征
    """
    pass
```

#### `resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False)`

```python
def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """
    根据比例或形状调整光流大小。
    
    Args:
        flow (Tensor): 光流张量，形状 (N, 2, H, W)
        size_type (str): 'ratio' 或 'shape'
        sizes (list): 比例或目标形状
        interp_mode (str): 插值模式
        align_corners (bool): 是否对齐角点
    
    Returns:
        Tensor: 调整后的光流
    """
    pass
```

#### `pixel_unshuffle(x, scale)`

```python
def pixel_unshuffle(x, scale):
    """
    像素反洗牌（Pixel Unshuffle）。
    
    Args:
        x (Tensor): 输入特征，形状 (B, C, H*scale, W*scale)
        scale (int): 下采样比例
    
    Returns:
        Tensor: 反洗牌后的特征，形状 (B, C*scale^2, H, W)
    """
    pass
```

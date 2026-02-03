# 评估模块文档

## evaluator.py

图像修复评估器。

### 核心函数

#### `ssim_fid100_f1(metrics, fid_scale=100)`

```python
def ssim_fid100_f1(metrics, fid_scale=100):
    """
    计算SSIM和FID的F1综合指标。
    
    F1 = 2 * ssim * fid_rel / (ssim + fid_rel + 1e-3)
    
    Args:
        metrics (dict): 评估指标字典
        fid_scale (float): FID缩放因子，默认100
    
    Returns:
        float: F1综合指标
    """
    pass
```

#### `lpips_fid100_f1(metrics, fid_scale=100)`

```python
def lpips_fid100_f1(metrics, fid_scale=100):
    """
    计算LPIPS和FID的F1综合指标。
    
    Args:
        metrics (dict): 评估指标字典
        fid_scale (float): FID缩放因子
    
    Returns:
        float: F1综合指标
    """
    pass
```

### 核心类

#### `InpaintingEvaluator`

```python
class InpaintingEvaluator:
    """
    图像修复评估器（离线）。
    
    在完整数据集上评估修复质量。
    
    Args:
        dataset: 数据集
        scores (dict): 评估指标字典
        area_grouping (bool): 是否按掩码面积分组
        bins (int): 分组数量，默认10
        batch_size (int): 批次大小，默认32
        device (str): 设备，默认'cuda'
        integral_func (callable, optional): 综合指标函数
        integral_title (str, optional): 综合指标名称
        clamp_image_range (tuple, optional): 图像裁剪范围
    """
    
    def __init__(self, dataset, scores, area_grouping=True, bins=10, 
                 batch_size=32, device='cuda', integral_func=None,
                 integral_title=None, clamp_image_range=None):
        """初始化评估器。"""
        pass
    
    def _get_bin_edges(self):
        """
        获取分组边界。
        
        Returns:
            tuple: (分组索引数组, 区间名称列表)
        """
        pass
    
    def evaluate(self, model=None):
        """
        评估模型。
        
        Args:
            model (callable, optional): 修复模型，为None时使用预计算结果
        
        Returns:
            dict: 评估结果
        """
        pass
```

#### `InpaintingEvaluatorOnline(nn.Module)`

```python
class InpaintingEvaluatorOnline(nn.Module):
    """
    在线图像修复评估器。
    
    在训练过程中实时评估，使用PyTorch Lightning。
    
    Args:
        scores (dict): 评估指标模块字典
        bins (int): 分组数量，默认10
        image_key (str): 图像键，默认'image'
        inpainted_key (str): 修复结果键，默认'inpainted'
        integral_func (callable, optional): 综合指标函数
        integral_title (str, optional): 综合指标名称
        clamp_image_range (tuple, optional): 图像裁剪范围
    """
    
    def __init__(self, scores, bins=10, image_key='image', inpainted_key='inpainted',
                 integral_func=None, integral_title=None, clamp_image_range=None):
        """初始化在线评估器。"""
        pass
    
    def _get_bins(self, mask_batch):
        """
        获取批次分组。
        
        Args:
            mask_batch (Tensor): 掩码批次
        
        Returns:
            ndarray: 分组索引
        """
        pass
    
    def forward(self, batch):
        """
        评估批次。
        
        Args:
            batch (dict): 数据批次
        
        Returns:
            dict: 批次评估结果
        """
        pass
    
    def process_batch(self, batch):
        """
        处理批次（forward的别名）。
        
        Args:
            batch (dict): 数据批次
        
        Returns:
            dict: 评估结果
        """
        pass
    
    def evaluation_end(self, states=None):
        """
        评估结束，汇总结果。
        
        Args:
            states (list, optional): 批次状态列表
        
        Returns:
            dict: 汇总评估结果
        """
        pass
```

---

## data.py

评估数据集类。

### 核心函数

#### `load_image(fname, mode='RGB', return_orig=False)`

```python
def load_image(fname, mode='RGB', return_orig=False):
    """
    加载图像。
    
    Args:
        fname (str): 文件路径
        mode (str): 图像模式，默认'RGB'
        return_orig (bool): 是否返回原始数据
    
    Returns:
        ndarray: 图像数组 [0, 1] 或 (图像, 原始数据)
    """
    pass
```

#### `ceil_modulo(x, mod)`

```python
def ceil_modulo(x, mod):
    """
    向上取整到模数倍数。
    
    Args:
        x (int): 数值
        mod (int): 模数
    
    Returns:
        int: 向上取整后的值
    """
    pass
```

#### `pad_img_to_modulo(img, mod)`

```python
def pad_img_to_modulo(img, mod):
    """
    将图像填充到模数倍数。
    
    Args:
        img (ndarray): 图像，形状 (C, H, W)
        mod (int): 模数
    
    Returns:
        ndarray: 填充后的图像
    """
    pass
```

#### `pad_tensor_to_modulo(img, mod)`

```python
def pad_tensor_to_modulo(img, mod):
    """
    将张量填充到模数倍数。
    
    Args:
        img (Tensor): 图像张量，形状 (B, C, H, W)
        mod (int): 模数
    
    Returns:
        Tensor: 填充后的张量
    """
    pass
```

#### `scale_image(img, factor, interpolation=cv2.INTER_AREA)`

```python
def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    """
    缩放图像。
    
    Args:
        img (ndarray): 图像
        factor (float): 缩放因子
        interpolation: 插值方法
    
    Returns:
        ndarray: 缩放后的图像
    """
    pass
```

### 核心类

#### `InpaintingDataset(Dataset)`

```python
class InpaintingDataset(Dataset):
    """
    图像修复评估数据集。
    
    加载图像和对应的掩码。
    
    Args:
        datadir (str): 数据目录
        img_suffix (str): 图像后缀，默认'.jpg'
        pad_out_to_modulo (int, optional): 填充模数
        scale_factor (float, optional): 缩放因子
    """
    
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        """初始化数据集。"""
        pass
    
    def __len__(self):
        """获取数据集大小。"""
        pass
    
    def __getitem__(self, i):
        """
        获取样本。
        
        Args:
            i (int): 索引
        
        Returns:
            dict: 包含'image', 'mask'的字典
        """
        pass
```

#### `InpaintingEvalOnlineDataset(Dataset)`

```python
class InpaintingEvalOnlineDataset(Dataset):
    """
    在线评估数据集。
    
    动态生成掩码用于在线评估。
    
    Args:
        indir (str): 图像目录
        mask_generator: 掩码生成器
        img_suffix (str): 图像后缀，默认'.jpg'
        pad_out_to_modulo (int, optional): 填充模数
        scale_factor (float, optional): 缩放因子
    """
    
    def __init__(self, indir, mask_generator, img_suffix='.jpg', 
                 pad_out_to_modulo=None, scale_factor=None, **kwargs):
        """初始化在线评估数据集。"""
        pass
    
    def __len__(self):
        """获取数据集大小。"""
        pass
    
    def __getitem__(self, i):
        """获取样本。"""
        pass
```

---

## base_loss.py

评估损失基类。

### 核心函数

#### `get_groupings(groups)`

```python
def get_groupings(groups):
    """
    获取分组索引。
    
    Args:
        groups (ndarray): 组号数组
    
    Returns:
        dict: {组号: 索引列表}
    """
    pass
```

#### `fid_calculate_activation_statistics(act)`

```python
def fid_calculate_activation_statistics(act):
    """
    计算FID的激活统计量。
    
    Args:
        act (ndarray): 激活值
    
    Returns:
        tuple: (均值, 协方差矩阵)
    """
    pass
```

#### `calculate_frechet_distance(activations_pred, activations_target, eps=1e-6)`

```python
def calculate_frechet_distance(activations_pred, activations_target, eps=1e-6):
    """
    计算Fréchet距离（FID）。
    
    Args:
        activations_pred (ndarray): 预测激活
        activations_target (ndarray): 目标激活
        eps (float): 数值稳定性常数
    
    Returns:
        float: Fréchet距离
    """
    pass
```

### 核心类

#### `EvaluatorScore(nn.Module)`

```python
class EvaluatorScore(nn.Module):
    """
    评估分数基类。
    
    评估指标的抽象基类。
    """
    
    @abstractmethod
    def forward(self, pred_batch, target_batch, mask):
        """
        计算分数。
        
        Args:
            pred_batch (Tensor): 预测批次
            target_batch (Tensor): 目标批次
            mask (Tensor): 掩码
        """
        pass
    
    @abstractmethod
    def get_value(self, groups=None, states=None):
        """
        获取评估值。
        
        Args:
            groups (ndarray, optional): 分组
            states (list, optional): 状态列表
        
        Returns:
            tuple: (总值, 分组值)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """重置评估器状态。"""
        pass
```

#### `PairwiseScore(EvaluatorScore)`

```python
class PairwiseScore(EvaluatorScore, ABC):
    """
    成对评估分数基类。
    
    用于SSIM、LPIPS等成对评估指标。
    """
    
    def __init__(self):
        """初始化成对分数。"""
        pass
    
    def get_value(self, groups=None, states=None):
        """获取统计值。"""
        pass
    
    def reset(self):
        """重置。"""
        pass
```

#### `SSIMScore(PairwiseScore)`

```python
class SSIMScore(PairwiseScore):
    """
    SSIM评估分数。
    
    Args:
        window_size (int): SSIM窗口大小，默认11
    """
    
    def __init__(self, window_size=11):
        """初始化SSIM评估器。"""
        pass
    
    def forward(self, pred_batch, target_batch, mask=None):
        """计算SSIM分数。"""
        pass
```

#### `LPIPSScore(PairwiseScore)`

```python
class LPIPSScore(PairwiseScore):
    """
    LPIPS评估分数。
    
    学习感知图像块相似度。
    
    Args:
        model (str): 模型类型，默认'net-lin'
        net (str): 网络类型，默认'vgg'
        model_path (str, optional): 模型路径
        use_gpu (bool): 是否使用GPU
    """
    
    def __init__(self, model='net-lin', net='vgg', model_path=None, use_gpu=True):
        """初始化LPIPS评估器。"""
        pass
    
    def forward(self, pred_batch, target_batch, mask=None):
        """计算LPIPS分数。"""
        pass
```

#### `FIDScore(EvaluatorScore)`

```python
class FIDScore(EvaluatorScore):
    """
    FID评估分数。
    
    Fréchet Inception Distance。
    
    Args:
        dims (int): Inception特征维度，默认2048
        eps (float): 数值稳定性常数
    """
    
    def __init__(self, dims=2048, eps=1e-6):
        """初始化FID评估器。"""
        pass
    
    def forward(self, pred_batch, target_batch, mask=None):
        """计算FID分数。"""
        pass
    
    def get_value(self, groups=None, states=None):
        """获取FID值。"""
        pass
    
    def reset(self):
        """重置FID评估器。"""
        pass
```

---

## utils.py

评估工具函数。

### 核心函数

#### `load_yaml(path)`

```python
def load_yaml(path):
    """
    加载YAML配置文件。
    
    Args:
        path (str): YAML文件路径
    
    Returns:
        EasyDict: 配置字典
    """
    pass
```

#### `move_to_device(obj, device)`

```python
def move_to_device(obj, device):
    """
    将对象移动到指定设备。
    
    Args:
        obj: 对象（模块、张量、列表、字典）
        device: 目标设备
    
    Returns:
        移动后的对象
    """
    pass
```

---

## refinement.py

图像修复细化（后处理）。

### 核心函数

#### `_pyrdown(im, downsize=None)`

```python
def _pyrdown(im, downsize=None):
    """
    图像金字塔下采样。
    
    Args:
        im (Tensor): 输入图像，形状 (N, 3, H, W)
        downsize (tuple, optional): 目标尺寸
    
    Returns:
        Tensor: 下采样后的图像
    """
    pass
```

#### `_pyrdown_mask(mask, downsize=None, eps=1e-8, blur_mask=True, round_up=True)`

```python
def _pyrdown_mask(mask, downsize=None, eps=1e-8, blur_mask=True, round_up=True):
    """
    掩码金字塔下采样。
    
    Args:
        mask (Tensor): 输入掩码，形状 (B, 1, H, W)
        downsize (tuple, optional): 目标尺寸
        eps (float): 阈值
        blur_mask (bool): 是否模糊
        round_up (bool): 是否向上取整
    
    Returns:
        Tensor: 下采样后的掩码
    """
    pass
```

#### `_erode_mask(mask, ekernel=None, eps=1e-8)`

```python
def _erode_mask(mask, ekernel=None, eps=1e-8):
    """
    腐蚀掩码。
    
    Args:
        mask (Tensor): 输入掩码
        ekernel (Tensor, optional): 腐蚀核
        eps (float): 阈值
    
    Returns:
        Tensor: 腐蚀后的掩码
    """
    pass
```

#### `_infer(image, mask, forward_front, forward_rears, ref_lower_res, orig_shape, devices, scale_ind, n_iters=15, lr=0.002)`

```python
def _infer(image, mask, forward_front, forward_rears, ref_lower_res, 
           orig_shape, devices, scale_ind, n_iters=15, lr=0.002):
    """
    在指定尺度执行细化推理。
    
    Args:
        image (Tensor): 输入图像
        mask (Tensor): 输入掩码
        forward_front (nn.Module): 网络前部
        forward_rears (list[nn.Module]): 网络后部列表（多GPU）
        ref_lower_res (Tensor): 低分辨率参考
        orig_shape (tuple): 原始形状
        devices (list): 设备列表
        scale_ind (int): 尺度索引
        n_iters (int): 迭代次数
        lr (float): 学习率
    
    Returns:
        Tensor: 细化后的图像
    """
    pass
```

#### `_get_image_mask_pyramid(batch, min_side, max_scales, px_budget)`

```python
def _get_image_mask_pyramid(batch, min_side, max_scales, px_budget):
    """
    构建图像-掩码金字塔。
    
    Args:
        batch (dict): 批次数据
        min_side (int): 最小边长
        max_scales (int): 最大尺度数
        px_budget (int): 像素预算
    
    Returns:
        tuple: (图像列表, 掩码列表)，从低分辨率到高分辨率
    """
    pass
```

#### `refine_predict(batch, inpainter, gpu_ids, modulo, n_iters, lr, min_side, max_scales, px_budget)`

```python
def refine_predict(batch, inpainter, gpu_ids, modulo, n_iters, lr, 
                   min_side, max_scales, px_budget):
    """
    细化修复预测。
    
    使用多尺度细化提升修复质量。
    
    Args:
        batch (dict): 图像-掩码批次
        inpainter (nn.Module): 修复网络
        gpu_ids (str): GPU ID字符串，如"0,1"
        modulo (int): 填充模数
        n_iters (int): 每尺度迭代次数
        lr (float): 学习率
        min_side (int): 最小边长限制
        max_scales (int): 最大尺度数
        px_budget (int): 像素预算
    
    Returns:
        Tensor: 细化后的修复图像
    """
    pass
```

---

## __init__.py

评估器工厂函数。

### 核心函数

#### `make_evaluator(kind='default', ssim=True, lpips=True, fid=True, integral_kind=None, **kwargs)`

```python
def make_evaluator(kind='default', ssim=True, lpips=True, fid=True, 
                   integral_kind=None, **kwargs):
    """
    创建评估器。
    
    Args:
        kind (str): 评估器类型，默认'default'
        ssim (bool): 是否包含SSIM
        lpips (bool): 是否包含LPIPS
        fid (bool): 是否包含FID
        integral_kind (str, optional): 综合指标类型
        **kwargs: 其他参数
    
    Returns:
        InpaintingEvaluatorOnline: 评估器实例
    """
    pass
```

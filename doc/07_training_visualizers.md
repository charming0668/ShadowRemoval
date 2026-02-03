# Training 可视化模块文档

## base.py

基础可视化类。

### 核心类

#### `BaseVisualizer`

```python
class BaseVisualizer:
    """
    可视化器基类。
    
    定义可视化器的接口。
    """
    
    @abc.abstractmethod
    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        """
        可视化批次数据。
        
        Args:
            epoch_i (int): 当前周期
            batch_i (int): 当前批次索引
            batch (dict): 数据批次
            suffix (str): 文件后缀
            rank (int, optional): 分布式rank
        """
        pass
```

### 核心函数

#### `visualize_mask_and_images(images_dict, keys, last_without_mask=True, rescale_keys=None, mask_only_first=None, black_mask=False)`

```python
def visualize_mask_and_images(images_dict, keys, last_without_mask=True, 
                               rescale_keys=None, mask_only_first=None, black_mask=False):
    """
    可视化掩码和图像。
    
    将多组图像水平拼接，并在非最后一幅图像上叠加掩码边界。
    
    Args:
        images_dict (dict): 图像字典，包含'mask'和各图像键
        keys (list): 要可视化的图像键列表
        last_without_mask (bool): 最后一幅图是否不加掩码
        rescale_keys (list, optional): 需要重新缩放的键
        mask_only_first (bool, optional): 是否只在第一幅图加掩码
        black_mask (bool): 是否将掩码区域置黑
    
    Returns:
        ndarray: 拼接后的可视化图像
    """
    pass
```

#### `visualize_mask_and_images_batch(batch, keys, max_items=10, last_without_mask=True, rescale_keys=None)`

```python
def visualize_mask_and_images_batch(batch, keys, max_items=10, 
                                     last_without_mask=True, rescale_keys=None):
    """
    批次可视化掩码和图像。
    
    Args:
        batch (dict): 批次数据
        keys (list): 图像键列表
        max_items (int): 最大可视化样本数
        last_without_mask (bool): 最后一幅是否不加掩码
        rescale_keys (list, optional): 重新缩放键
    
    Returns:
        ndarray: 拼接后的批次可视化图像
    """
    pass
```

---

## directory.py

目录可视化器。

### 核心类

#### `DirectoryVisualizer(BaseVisualizer)`

```python
class DirectoryVisualizer(BaseVisualizer):
    """
    目录可视化器。
    
    将可视化结果保存到指定目录。
    
    Args:
        outdir (str): 输出目录
        key_order (list): 图像键顺序，默认['image', 'predicted_image', 'inpainted']
        max_items_in_batch (int): 批次中最大可视化项数，默认10
        last_without_mask (bool): 最后一项是否不加掩码，默认True
        rescale_keys (list, optional): 需要重新缩放的键
    
    Attributes:
        outdir (str): 输出目录
        key_order (list): 键顺序
        max_items_in_batch (int): 最大项数
        last_without_mask (bool): 最后无掩码标志
        rescale_keys (list): 重新缩放键
    """
    
    DEFAULT_KEY_ORDER = 'image predicted_image inpainted'.split(' ')
    
    def __init__(self, outdir, key_order=DEFAULT_KEY_ORDER, max_items_in_batch=10,
                 last_without_mask=True, rescale_keys=None):
        """初始化目录可视化器。"""
        pass
    
    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        """
        保存可视化结果到目录。
        
        Args:
            epoch_i (int): 当前周期
            batch_i (int): 批次索引
            batch (dict): 数据批次
            suffix (str): 文件名后缀
            rank (int, optional): 分布式rank
        """
        pass
```

---

## colors.py

颜色生成工具。

### 核心函数

#### `generate_colors(nlabels, type='bright', first_color_black=False, last_color_black=True, verbose=False)`

```python
def generate_colors(nlabels, type='bright', first_color_black=False, 
                    last_color_black=True, verbose=False):
    """
    生成随机颜色映射。
    
    用于分割任务的可视化。
    
    Args:
        nlabels (int): 标签数量（颜色映射大小）
        type (str): 颜色类型 ('bright' 或 'soft')
        first_color_black (bool): 第一种颜色是否为黑色
        last_color_black (bool): 最后一种颜色是否为黑色
        verbose (bool): 是否打印信息并显示颜色映射
    
    Returns:
        tuple: (RGB颜色列表, matplotlib颜色映射)
    """
    pass
```

---

## noop.py

空操作可视化器。

### 核心类

#### `NoopVisualizer(BaseVisualizer)`

```python
class NoopVisualizer(BaseVisualizer):
    """
    空操作可视化器。
    
    不执行任何操作，用于禁用可视化。
    """
    
    def __init__(self, *args, **kwargs):
        """初始化空可视化器。"""
        pass
    
    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        """
        空操作。
        
        Args:
            epoch_i (int): 当前周期
            batch_i (int): 批次索引
            batch (dict): 数据批次
            suffix (str): 后缀
            rank (int, optional): rank
        """
        pass
```

---

## __init__.py

可视化器工厂函数。

### 核心函数

#### `make_visualizer(kind, **kwargs)`

```python
def make_visualizer(kind, **kwargs):
    """
    创建可视化器。
    
    Args:
        kind (str): 可视化器类型 ('directory', 'noop')
        **kwargs: 可视化器参数
    
    Returns:
        BaseVisualizer: 可视化器实例
    """
    pass
```

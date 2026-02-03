# 测试脚本文档

## test.py

主测试脚本，用于模型推理。

### 核心流程

```python
"""
阴影去除测试脚本。

加载训练好的模型，对测试集进行推理并保存结果。

主要步骤：
1. 解析命令行参数
2. 创建测试数据集和数据加载器
3. 加载final_net模型
4. 加载预训练权重（shadowremoval.pkl 和 refinement.pkl）
5. 对测试图像进行推理
6. 保存结果图像

命令行参数：
    --test_dir (str): 测试数据目录，默认'./ShadowDataset/test/'
    --output_dir (str): 输出目录，默认'results/'
    -test_batch_size (int): 测试批次大小，默认1
"""
```

---

## test_dataset.py

测试数据集定义。

### 核心类

#### `dehaze_test_dataset(Dataset)`

```python
class dehaze_test_dataset(Dataset):
    """
    去阴影测试数据集。
    
    加载测试图像用于阴影去除。
    
    Args:
        test_dir (str): 测试目录，应包含'LQ/'子目录
    
    Attributes:
        transform (Compose): ToTensor变换
        list_test_hazy (list): 测试图像文件名列表
        root_hazy (str): 测试图像目录路径
        file_len (int): 数据集长度
    """
    
    def __init__(self, test_dir):
        """
        初始化测试数据集。
        
        Args:
            test_dir (str): 测试数据目录
        """
        pass
    
    def __getitem__(self, index, is_train=True):
        """
        获取单个样本。
        
        Args:
            index (int): 样本索引
            is_train (bool): 是否为训练模式（此处忽略）
        
        Returns:
            tuple: (图像张量, 文件名)
        """
        pass
    
    def __len__(self):
        """
        获取数据集长度。
        
        Returns:
            int: 数据集长度
        """
        pass
```

---

## utils.py (saicinpainting)

通用工具函数。

### 核心函数

#### `check_and_warn_input_range(tensor, min_value, max_value, name)`

```python
def check_and_warn_input_range(tensor, min_value, max_value, name):
    """
    检查并警告输入范围。
    
    Args:
        tensor (Tensor): 输入张量
        min_value (float): 最小值
        max_value (float): 最大值
        name (str): 张量名称（用于警告信息）
    """
    pass
```

#### `sum_dict_with_prefix(target, cur_dict, prefix, default=0)`

```python
def sum_dict_with_prefix(target, cur_dict, prefix, default=0):
    """
    带前缀累加字典值。
    
    Args:
        target (dict): 目标字典
        cur_dict (dict): 当前字典
        prefix (str): 键前缀
        default: 默认值
    """
    pass
```

#### `average_dicts(dict_list)`

```python
def average_dicts(dict_list):
    """
    平均多个字典。
    
    Args:
        dict_list (list): 字典列表
    
    Returns:
        dict: 平均后的字典
    """
    pass
```

#### `add_prefix_to_keys(dct, prefix)`

```python
def add_prefix_to_keys(dct, prefix):
    """
    为字典键添加前缀。
    
    Args:
        dct (dict): 原字典
        prefix (str): 前缀
    
    Returns:
        dict: 键添加前缀后的字典
    """
    pass
```

#### `set_requires_grad(module, value)`

```python
def set_requires_grad(module, value):
    """
    设置模块参数是否需要梯度。
    
    Args:
        module (nn.Module): 模块
        value (bool): 是否需要梯度
    """
    pass
```

#### `flatten_dict(dct)`

```python
def flatten_dict(dct):
    """
    展平嵌套字典。
    
    Args:
        dct (dict): 嵌套字典
    
    Returns:
        dict: 展平后的字典
    """
    pass
```

#### `get_shape(t)`

```python
def get_shape(t):
    """
    获取张量/字典/列表的形状。
    
    Args:
        t: 输入对象
    
    Returns:
        对象的形状信息
    """
    pass
```

### 核心类

#### `LinearRamp`

```python
class LinearRamp:
    """
    线性斜坡调度器。
    
    Args:
        start_value (float): 起始值，默认0
        end_value (float): 结束值，默认1
        start_iter (int): 起始迭代，默认-1
        end_iter (int): 结束迭代，默认0
    """
    
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        """初始化线性斜坡。"""
        pass
    
    def __call__(self, i):
        """
        获取当前值。
        
        Args:
            i (int): 当前迭代
        
        Returns:
            float: 当前斜坡值
        """
        pass
```

#### `LadderRamp`

```python
class LadderRamp:
    """
    阶梯斜坡调度器。
    
    Args:
        start_iters (list): 起始迭代列表
        values (list): 对应值列表
    """
    
    def __init__(self, start_iters, values):
        """初始化阶梯斜坡。"""
        pass
    
    def __call__(self, i):
        """
        获取当前阶梯值。
        
        Args:
            i (int): 当前迭代
        
        Returns:
            float: 当前值
        """
        pass
```

#### `get_ramp(kind='ladder', **kwargs)`

```python
def get_ramp(kind='ladder', **kwargs):
    """
    获取斜坡调度器。
    
    Args:
        kind (str): 类型 ('linear', 'ladder')
        **kwargs: 调度器参数
    
    Returns:
        LinearRamp 或 LadderRamp: 斜坡调度器
    """
    pass
```

### DDP相关函数

#### `get_has_ddp_rank()`

```python
def get_has_ddp_rank():
    """
    检查是否有DDP rank环境变量。
    
    Returns:
        bool: 是否为DDP环境
    """
    pass
```

#### `handle_ddp_subprocess()`

```python
def handle_ddp_subprocess():
    """
    处理DDP子进程的装饰器。
    
    用于多进程分布式训练的主函数装饰。
    
    Returns:
        decorator: 装饰器函数
    """
    pass
```

#### `handle_ddp_parent_process()`

```python
def handle_ddp_parent_process():
    """
    处理DDP父进程。
    
    Returns:
        bool: 是否有父进程
    """
    pass
```

#### `handle_deterministic_config(config)`

```python
def handle_deterministic_config(config):
    """
    处理确定性配置。
    
    Args:
        config: 配置对象
    
    Returns:
        bool: 是否设置了确定性模式
    """
    pass
```

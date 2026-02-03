# Training 数据模块文档

## datasets.py

数据集类和数据加载器构建。

### 核心类

#### `InpaintingTrainDataset(Dataset)`

```python
class InpaintingTrainDataset(Dataset):
    """
    图像修复训练数据集。
    
    加载图像并应用变换和掩码生成。
    
    Args:
        indir (str): 输入图像目录
        mask_generator: 掩码生成器
        transform: 图像变换
    
    Attributes:
        in_files (list): 图像文件路径列表
        mask_generator: 掩码生成器实例
        transform: 图像变换
        iter_i (int): 迭代计数器
    """
    
    def __init__(self, indir, mask_generator, transform):
        """初始化训练数据集。"""
        pass
    
    def __len__(self):
        """
        获取数据集大小。
        
        Returns:
            int: 数据集大小
        """
        pass
    
    def __getitem__(self, item):
        """
        获取单个样本。
        
        Args:
            item (int): 样本索引
        
        Returns:
            dict: 包含'image'和'mask'的字典
        """
        pass
```

#### `InpaintingTrainWebDataset(IterableDataset)`

```python
class InpaintingTrainWebDataset(IterableDataset):
    """
    WebDataset格式的训练数据集。
    
    用于大规模数据集的高效加载。
    
    Args:
        indir (str): 输入目录
        mask_generator: 掩码生成器
        transform: 图像变换
        shuffle_buffer (int): 打乱缓冲区大小
    """
    
    def __init__(self, indir, mask_generator, transform, shuffle_buffer=200):
        """初始化Web数据集。"""
        pass
    
    def __iter__(self):
        """
        迭代器。
        
        Yields:
            dict: 包含'image'和'mask'的字典
        """
        pass
```

#### `ImgSegmentationDataset(Dataset)`

```python
class ImgSegmentationDataset(Dataset):
    """
    带语义分割的图像数据集。
    
    Args:
        indir (str): 图像目录
        mask_generator: 掩码生成器
        transform: 图像变换
        out_size (int): 输出图像尺寸
        segm_indir (str): 分割标注目录
        semantic_seg_n_classes (int): 语义分割类别数
    """
    
    def __init__(self, indir, mask_generator, transform, out_size, 
                 segm_indir, semantic_seg_n_classes):
        """初始化分割数据集。"""
        pass
    
    def __len__(self):
        """获取数据集大小。"""
        pass
    
    def __getitem__(self, item):
        """
        获取样本。
        
        Returns:
            dict: 包含'image', 'mask', 'segm', 'segm_classes'
        """
        pass
    
    def load_semantic_segm(self, img_path):
        """
        加载语义分割标注。
        
        Args:
            img_path (str): 图像路径
        
        Returns:
            tuple: (one_hot编码, 类别张量)
        """
        pass
```

### 核心函数

#### `get_transforms(transform_variant, out_size)`

```python
def get_transforms(transform_variant, out_size):
    """
    获取图像变换组合。
    
    支持的变换变体：
    - 'default': 默认变换（随机缩放、裁剪、翻转、颜色增强）
    - 'distortions': 包含畸变的变换
    - 'distortions_scale05_1': 缩放范围0.5-1.0的畸变
    - 'distortions_scale03_12': 缩放范围0.3-1.2的畸变
    - 'distortions_scale03_07': 缩放范围0.3-0.7的畸变
    - 'distortions_light': 轻度畸变
    - 'non_space_transform': 非空间变换
    - 'no_augs': 无增强
    
    Args:
        transform_variant (str): 变换变体名称
        out_size (int): 输出图像尺寸
    
    Returns:
        A.Compose: albumentations变换组合
    """
    pass
```

#### `make_default_train_dataloader(indir, kind='default', out_size=512, mask_gen_kwargs=None, transform_variant='default', mask_generator_kind='mixed', dataloader_kwargs=None, ddp_kwargs=None, **kwargs)`

```python
def make_default_train_dataloader(indir, kind='default', out_size=512, 
                                   mask_gen_kwargs=None, transform_variant='default',
                                   mask_generator_kind='mixed', dataloader_kwargs=None, 
                                   ddp_kwargs=None, **kwargs):
    """
    创建默认训练数据加载器。
    
    Args:
        indir (str): 输入目录
        kind (str): 数据集类型，'default', 'default_web', 'img_with_segm'
        out_size (int): 输出图像尺寸
        mask_gen_kwargs (dict): 掩码生成器参数
        transform_variant (str): 变换变体
        mask_generator_kind (str): 掩码生成器类型
        dataloader_kwargs (dict): DataLoader参数
        ddp_kwargs (dict): 分布式训练参数
    
    Returns:
        DataLoader: 训练数据加载器
    """
    pass
```

#### `make_default_val_dataset(indir, kind='default', out_size=512, transform_variant='default', **kwargs)`

```python
def make_default_val_dataset(indir, kind='default', out_size=512, 
                              transform_variant='default', **kwargs):
    """
    创建默认验证数据集。
    
    Args:
        indir (str): 输入目录或目录列表
        kind (str): 数据集类型
        out_size (int): 输出尺寸
        transform_variant (str): 变换变体
    
    Returns:
        Dataset: 验证数据集
    """
    pass
```

#### `make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs)`

```python
def make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs):
    """
    创建默认验证数据加载器。
    
    Args:
        dataloader_kwargs (dict): DataLoader参数
    
    Returns:
        DataLoader: 验证数据加载器
    """
    pass
```

#### `make_constant_area_crop_params(img_height, img_width, min_size=128, max_size=512, area=256*256, round_to_mod=16)`

```python
def make_constant_area_crop_params(img_height, img_width, min_size=128, 
                                    max_size=512, area=256*256, round_to_mod=16):
    """
    生成恒定面积裁剪参数。
    
    Args:
        img_height (int): 图像高度
        img_width (int): 图像宽度
        min_size (int): 最小裁剪尺寸
        max_size (int): 最大裁剪尺寸
        area (int): 目标面积
        round_to_mod (int): 对齐模数
    
    Returns:
        tuple: (start_y, start_x, crop_height, crop_width)
    """
    pass
```

---

## masks.py

掩码生成器，支持多种掩码类型。

### 核心枚举

#### `DrawMethod(Enum)`

```python
class DrawMethod(Enum):
    """绘制方法枚举。"""
    LINE = 'line'      # 线条绘制
    CIRCLE = 'circle'  # 圆形绘制
    SQUARE = 'square'  # 方形绘制
```

### 核心函数

#### `make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10, draw_method=DrawMethod.LINE)`

```python
def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, 
                                min_times=0, max_times=10, draw_method=DrawMethod.LINE):
    """
    生成随机不规则掩码。
    
    通过随机绘制线条/圆/方形创建不规则掩码。
    
    Args:
        shape (tuple): 掩码形状 (H, W)
        max_angle (int): 最大角度
        max_len (int): 最大线条长度
        max_width (int): 最大笔刷宽度
        min_times (int): 最小绘制次数
        max_times (int): 最大绘制次数
        draw_method (DrawMethod): 绘制方法
    
    Returns:
        ndarray: 掩码，形状 (1, H, W)
    """
    pass
```

#### `make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3)`

```python
def make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, 
                                bbox_max_size=100, min_times=0, max_times=3):
    """
    生成随机矩形掩码。
    
    Args:
        shape (tuple): 掩码形状
        margin (int): 边距
        bbox_min_size (int): 边界框最小尺寸
        bbox_max_size (int): 边界框最大尺寸
        min_times (int): 最小矩形数量
        max_times (int): 最大矩形数量
    
    Returns:
        ndarray: 掩码
    """
    pass
```

#### `make_random_superres_mask(shape, min_step=2, max_step=4, min_width=1, max_width=3)`

```python
def make_random_superres_mask(shape, min_step=2, max_step=4, min_width=1, max_width=3):
    """
    生成随机超分辨率掩码（网格状）。
    
    Args:
        shape (tuple): 掩码形状
        min_step (int): 最小步长
        max_step (int): 最大步长
        min_width (int): 最小线宽
        max_width (int): 最大线宽
    
    Returns:
        ndarray: 网格状掩码
    """
    pass
```

### 核心类

#### `RandomIrregularMaskGenerator`

```python
class RandomIrregularMaskGenerator:
    """
    随机不规则掩码生成器。
    
    Args:
        max_angle (int): 最大角度
        max_len (int): 最大长度
        max_width (int): 最大宽度
        min_times (int): 最小绘制次数
        max_times (int): 最大绘制次数
        ramp_kwargs (dict): 渐进参数
        draw_method (DrawMethod): 绘制方法
    """
    
    def __init__(self, max_angle=4, max_len=60, max_width=20, min_times=0, 
                 max_times=10, ramp_kwargs=None, draw_method=DrawMethod.LINE):
        """初始化生成器。"""
        pass
    
    def __call__(self, img, iter_i=None, raw_image=None):
        """
        生成掩码。
        
        Args:
            img (ndarray): 输入图像
            iter_i (int): 当前迭代次数（用于渐进）
            raw_image: 原始图像
        
        Returns:
            ndarray: 生成的掩码
        """
        pass
```

#### `RandomRectangleMaskGenerator`

```python
class RandomRectangleMaskGenerator:
    """
    随机矩形掩码生成器。
    
    Args:
        margin (int): 边距
        bbox_min_size (int): 最小边界框尺寸
        bbox_max_size (int): 最大边界框尺寸
        min_times (int): 最小矩形数
        max_times (int): 最大矩形数
        ramp_kwargs (dict): 渐进参数
    """
    
    def __init__(self, margin=10, bbox_min_size=30, bbox_max_size=100, 
                 min_times=0, max_times=3, ramp_kwargs=None):
        """初始化生成器。"""
        pass
    
    def __call__(self, img, iter_i=None, raw_image=None):
        """生成矩形掩码。"""
        pass
```

#### `RandomSegmentationMaskGenerator`

```python
class RandomSegmentationMaskGenerator:
    """
    基于分割的随机掩码生成器。
    
    使用分割模型生成对象级别的掩码。
    
    Args:
        **kwargs: 分割掩码参数
    """
    
    def __init__(self, **kwargs):
        """初始化生成器。"""
        pass
    
    def __call__(self, img, iter_i=None, raw_image=None):
        """生成分割掩码。"""
        pass
```

#### `OutpaintingMaskGenerator`

```python
class OutpaintingMaskGenerator:
    """
    外绘掩码生成器。
    
    生成图像边缘的掩码，用于外绘任务。
    
    Args:
        min_padding_percent (float): 最小填充比例
        max_padding_percent (float): 最大填充比例
        left_padding_prob (float): 左侧填充概率
        top_padding_prob (float): 顶部填充概率
        right_padding_prob (float): 右侧填充概率
        bottom_padding_prob (float): 底部填充概率
        is_fixed_randomness (bool): 是否固定随机性
    """
    
    def __init__(self, min_padding_percent=0.04, max_padding_percent=0.25,
                 left_padding_prob=0.5, top_padding_prob=0.5,
                 right_padding_prob=0.5, bottom_padding_prob=0.5,
                 is_fixed_randomness=False):
        """初始化外绘掩码生成器。"""
        pass
    
    def __call__(self, img, iter_i=None, raw_image=None):
        """生成外绘掩码。"""
        pass
```

#### `MixedMaskGenerator`

```python
class MixedMaskGenerator:
    """
    混合掩码生成器。
    
    按概率混合多种掩码类型。
    
    Args:
        irregular_proba (float): 不规则掩码概率
        irregular_kwargs (dict): 不规则掩码参数
        box_proba (float): 矩形掩码概率
        box_kwargs (dict): 矩形掩码参数
        segm_proba (float): 分割掩码概率
        segm_kwargs (dict): 分割掩码参数
        squares_proba (float): 方形掩码概率
        squares_kwargs (dict): 方形掩码参数
        superres_proba (float): 超分辨率掩码概率
        superres_kwargs (dict): 超分辨率掩码参数
        outpainting_proba (float): 外绘掩码概率
        outpainting_kwargs (dict): 外绘掩码参数
        invert_proba (float): 掩码反转概率
    """
    
    def __init__(self, irregular_proba=1/3, irregular_kwargs=None,
                 box_proba=1/3, box_kwargs=None, segm_proba=1/3, 
                 segm_kwargs=None, squares_proba=0, squares_kwargs=None,
                 superres_proba=0, superres_kwargs=None,
                 outpainting_proba=0, outpainting_kwargs=None, invert_proba=0):
        """初始化混合生成器。"""
        pass
    
    def __call__(self, img, iter_i=None, raw_image=None):
        """生成混合掩码。"""
        pass
```

### 核心函数

#### `get_mask_generator(kind, kwargs)`

```python
def get_mask_generator(kind, kwargs):
    """
    获取掩码生成器工厂函数。
    
    Args:
        kind (str): 生成器类型 ('mixed', 'outpainting', 'dumb')
        kwargs (dict): 生成器参数
    
    Returns:
        MaskGenerator: 掩码生成器实例
    """
    pass
```

---

## aug.py

自定义图像增强变换。

### 核心类

#### `IAAAffine2(DualIAATransform)`

```python
class IAAAffine2(DualIAATransform):
    """
    仿射变换增强。
    
    在输入上放置规则网格点并随机移动这些点邻域，通过仿射变换实现。
    
    注意：如果掩码有除{0,1}外的值，会引入插值伪影。
    
    Args:
        scale (tuple): 缩放范围，默认(0.7, 1.3)
        translate_percent (float): 平移百分比
        translate_px (int): 平移像素数
        rotate (float): 旋转角度范围
        shear (tuple): 剪切角度范围
        order (int): 插值顺序
        cval (int): 常数值填充
        mode (str): 填充模式
        always_apply (bool): 是否总是应用
        p (float): 应用概率
    
    Targets:
        image, mask
    """
    
    def __init__(self, scale=(0.7, 1.3), translate_percent=None, 
                 translate_px=None, rotate=0.0, shear=(-0.1, 0.1),
                 order=1, cval=0, mode="reflect", always_apply=False, p=0.5):
        """初始化仿射变换。"""
        pass
    
    @property
    def processor(self):
        """获取imgaug处理器。"""
        pass
```

#### `IAAPerspective2(DualIAATransform)`

```python
class IAAPerspective2(DualIAATransform):
    """
    透视变换增强。
    
    对输入执行随机四点透视变换。
    
    注意：如果掩码有除{0,1}外的值，会引入插值伪影。
    
    Args:
        scale (tuple): 标准差范围，默认(0.05, 0.1)
        keep_size (bool): 是否保持尺寸
        always_apply (bool): 是否总是应用
        p (float): 应用概率
        order (int): 插值顺序
        cval (int): 常数值填充
        mode (str): 填充模式
    
    Targets:
        image, mask
    """
    
    def __init__(self, scale=(0.05, 0.1), keep_size=True, 
                 always_apply=False, p=0.5, order=1, cval=0, mode="replicate"):
        """初始化透视变换。"""
        pass
    
    @property
    def processor(self):
        """获取imgaug处理器。"""
        pass
```

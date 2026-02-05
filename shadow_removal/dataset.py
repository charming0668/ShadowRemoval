"""数据集加载模块"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class ShadowRemovalDataset(Dataset):
    """阴影去除数据集"""
    
    def __init__(self, data_dir, crop_size=384, is_train=True):
        self.shadow_dir = os.path.join(data_dir, 'shadow')
        self.no_shadow_dir = os.path.join(data_dir, 'no_shadow')
        self.crop_size = crop_size
        self.is_train = is_train
        self.image_files = sorted(os.listdir(self.shadow_dir))
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        shadow_img = Image.open(os.path.join(self.shadow_dir, filename)).convert('RGB')
        no_shadow_img = Image.open(os.path.join(self.no_shadow_dir, filename)).convert('RGB')
        
        if self.is_train:
            shadow_img, no_shadow_img = self._apply_augmentation(shadow_img, no_shadow_img)
        
        return self.to_tensor(shadow_img), self.to_tensor(no_shadow_img), filename
    
    def _apply_augmentation(self, shadow_img, no_shadow_img):
        """数据增强"""
        # 随机裁剪
        i, j, h, w = transforms.RandomCrop.get_params(shadow_img, output_size=(self.crop_size, self.crop_size))
        shadow_img = TF.crop(shadow_img, i, j, h, w)
        no_shadow_img = TF.crop(no_shadow_img, i, j, h, w)
        
        # 随机翻转
        if random.random() > 0.5:
            shadow_img = TF.hflip(shadow_img)
            no_shadow_img = TF.hflip(no_shadow_img)
        
        if random.random() > 0.5:
            shadow_img = TF.vflip(shadow_img)
            no_shadow_img = TF.vflip(no_shadow_img)
        
        # 随机旋转
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            shadow_img = TF.rotate(shadow_img, angle)
            no_shadow_img = TF.rotate(no_shadow_img, angle)
        
        return shadow_img, no_shadow_img


def create_dataloaders(train_dir, val_dir, batch_size=4, num_workers=4, crop_size=384):
    """创建数据加载器"""
    train_dataset = ShadowRemovalDataset(train_dir, crop_size=crop_size, is_train=True)
    val_dataset = ShadowRemovalDataset(val_dir, crop_size=crop_size, is_train=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

"""
损失函数模块
实现L1 + SSIM组合损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CombinedLoss(nn.Module):
    """组合损失函数: L1 + α * SSIM"""
    
    def __init__(self, alpha=0.2):
        """
        初始化损失函数
        
        Args:
            alpha: SSIM损失的权重
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target):
        """
        计算组合损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 真实图像 [B, C, H, W]
            
        Returns:
            总损失值
        """
        # L1损失
        l1 = self.l1_loss(pred, target)
        
        # SSIM损失 (1 - SSIM，因为SSIM越大越好)
        ssim_val = self._ssim_loss(pred, target)
        ssim_loss = 1 - ssim_val
        
        # 组合损失
        total_loss = l1 + self.alpha * ssim_loss
        
        return total_loss
    
    def _ssim_loss(self, img1, img2, window_size=11):
        """计算SSIM损失"""
        channel = img1.size(1)
        window = self._create_window(window_size, channel).to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def _create_window(self, window_size, channel):
        """创建高斯窗口"""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
                for x in range(window_size)
            ])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

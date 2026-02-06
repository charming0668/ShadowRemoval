"""
评估指标模块
实现PSNR、SSIM和LPIPS计算
"""
import torch
import torch.nn.functional as F
import math
import os
import sys

# Add parent directory to path for LPIPS import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saicinpainting.evaluation.losses.lpips import PerceptualLoss


def calculate_psnr(img1, img2, max_val=1.0):
    """
    计算PSNR (峰值信噪比)
    
    Args:
        img1: 预测图像 [B, C, H, W], 范围[0, 1]
        img2: 真实图像 [B, C, H, W], 范围[0, 1]
        max_val: 像素最大值
        
    Returns:
        PSNR值 (dB)
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val / math.sqrt(mse.item()))
    return psnr


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    计算SSIM (结构相似性)
    
    Args:
        img1: 预测图像 [B, C, H, W]
        img2: 真实图像 [B, C, H, W]
        window_size: 窗口大小
        size_average: 是否对batch求平均
        
    Returns:
        SSIM值 [0, 1]
    """
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
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
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    """生成高斯窗口"""
    gauss = torch.Tensor([
        math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """创建SSIM计算窗口"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class LPIPSMetric:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) 指标
    使用预训练的深度网络计算感知相似度
    """
    def __init__(self, net='vgg', use_gpu=True):
        """
        初始化LPIPS模型
        
        Args:
            net: 使用的网络类型 ('alex', 'vgg', 'squeeze')
            use_gpu: 是否使用GPU
        """
        self.model = PerceptualLoss(model='net-lin', net=net, use_gpu=use_gpu, spatial=False)
        self.model.eval()
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def calculate(self, img1, img2):
        """
        计算LPIPS距离
        
        Args:
            img1: 预测图像 [B, C, H, W], 范围[0, 1]
            img2: 真实图像 [B, C, H, W], 范围[0, 1]
            
        Returns:
            LPIPS距离值 (越小越好，0表示完全相同)
        """
        with torch.no_grad():
            # LPIPS expects images in [0, 1] range and will normalize internally
            lpips_value = self.model(img1, img2, normalize=True)
            return lpips_value.item()


def calculate_lpips(img1, img2, lpips_model=None, net='vgg'):
    """
    计算LPIPS (学习感知图像块相似性)
    
    Args:
        img1: 预测图像 [B, C, H, W], 范围[0, 1]
        img2: 真实图像 [B, C, H, W], 范围[0, 1]
        lpips_model: 预先初始化的LPIPS模型（可选，用于避免重复初始化）
        net: 使用的网络类型 ('alex', 'vgg', 'squeeze')
        
    Returns:
        LPIPS距离值 (越小越好)
    """
    if lpips_model is None:
        use_gpu = img1.is_cuda
        lpips_model = LPIPSMetric(net=net, use_gpu=use_gpu)
    
    return lpips_model.calculate(img1, img2)

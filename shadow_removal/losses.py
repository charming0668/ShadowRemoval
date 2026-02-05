"""损失函数模块 - L1 + MS-SSIM + 感知损失 + 对抗损失"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_msssim import ms_ssim


class CombinedLoss(nn.Module):
    """组合损失: L1 + MS-SSIM + 感知损失 + 对抗损失"""
    
    def __init__(self, alpha=0.2, beta=0.01, gamma=0.0005):
        """
        Args:
            alpha: MS-SSIM损失权重
            beta: 感知损失权重
            gamma: 对抗损失权重
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred, target, discr_fake_pred=None):
        """
        Args:
            pred: 预测图像
            target: 真实图像
            discr_fake_pred: 判别器对预测图像的输出（可选）
        """
        # L1损失
        l1 = self.l1_loss(pred, target)
        
        # MS-SSIM损失
        ms_ssim_val = ms_ssim(pred, target, data_range=1.0, size_average=True)
        ms_ssim_loss = 1 - ms_ssim_val
        
        # 感知损失
        perceptual = self.perceptual_loss(pred, target)
        
        # 总损失
        total_loss = l1 + self.alpha * ms_ssim_loss + self.beta * perceptual
        
        # 对抗损失（如果提供）
        if discr_fake_pred is not None:
            adv_loss = F.softplus(-discr_fake_pred).mean()
            total_loss = total_loss + self.gamma * adv_loss
        
        return total_loss


class PerceptualLoss(nn.Module):
    """VGG感知损失"""
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # 加载预训练VGG19
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:30])
        
        # 冻结参数
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
        # ImageNet归一化
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """归一化到ImageNet标准"""
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        """计算感知损失"""
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        loss = 0.0
        for layer in self.vgg_layers:
            pred_norm = layer(pred_norm)
            target_norm = layer(target_norm)
            
            if isinstance(layer, nn.ReLU):
                loss += F.mse_loss(pred_norm, target_norm)
        
        return loss


class DiscriminatorLoss(nn.Module):
    """判别器损失（Non-Saturating + R1正则）"""
    
    def __init__(self, gp_coef=5.0):
        """
        Args:
            gp_coef: 梯度惩罚系数
        """
        super(DiscriminatorLoss, self).__init__()
        self.gp_coef = gp_coef
    
    def forward(self, real_batch, discr_real_pred, discr_fake_pred):
        """
        Args:
            real_batch: 真实图像（需要requires_grad=True）
            discr_real_pred: 判别器对真实图像的输出
            discr_fake_pred: 判别器对生成图像的输出
        """
        # 真实样本损失
        real_loss = F.softplus(-discr_real_pred).mean()
        
        # 假样本损失
        fake_loss = F.softplus(discr_fake_pred).mean()
        
        # R1梯度惩罚
        if torch.is_grad_enabled() and real_batch.requires_grad:
            grad_real = torch.autograd.grad(
                outputs=discr_real_pred.sum(),
                inputs=real_batch,
                create_graph=True
            )[0]
            grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = grad_penalty * self.gp_coef
        else:
            grad_penalty = 0.0
        
        total_loss = real_loss + fake_loss + grad_penalty
        
        return total_loss, {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'grad_penalty': grad_penalty if isinstance(grad_penalty, float) else grad_penalty.item()
        }

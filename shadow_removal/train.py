"""训练脚本"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
import swanlab

from model_convnext import fusion_net
from dataset import create_dataloaders
from losses import CombinedLoss
from metrics import calculate_psnr, calculate_ssim


class Trainer:
    """训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = fusion_net().to(self.device)
        
        self.train_loader, self.val_loader = create_dataloaders(
            train_dir=config.data.train_dir,
            val_dir=config.data.val_dir,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            crop_size=config.data.crop_size
        )
        
        self.criterion = CombinedLoss(alpha=config.training.alpha)
        
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.training.lr,
            betas=(0.9, 0.999)
        )
        
        self.warmup_epochs = int(config.training.epochs * 0.1)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs - self.warmup_epochs,
            eta_min=config.training.lr_min
        )
        
        self.start_epoch = 0
        self.best_psnr = 0.0
        self.global_step = 0
        
        os.makedirs(config.checkpoint.save_dir, exist_ok=True)
        
        if config.logging.use_swanlab:
            swanlab.init(project="shadow-removal", config=OmegaConf.to_container(config, resolve=True))
        
        if config.model.resume:
            self._load_checkpoint(config.model.resume)
    
    def train(self):
        """训练"""
        for epoch in range(self.start_epoch, self.config.training.epochs):
            train_loss = self._train_epoch(epoch)
            val_psnr, val_ssim = self._validate(epoch)
            current_lr = self._update_lr(epoch)
            
            print(f"Epoch [{epoch+1}/{self.config.training.epochs}] "
                  f"Loss: {train_loss:.4f} PSNR: {val_psnr:.2f} SSIM: {val_ssim:.4f} LR: {current_lr:.6f}")
            
            if self.config.logging.use_swanlab:
                swanlab.log({
                    "train/loss": train_loss,
                    "val/psnr": val_psnr,
                    "val/ssim": val_ssim,
                    "train/lr": current_lr
                }, step=epoch)
            
            self._save_checkpoint(epoch, val_psnr)
    
    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (shadow, no_shadow, _) in enumerate(self.train_loader):
            shadow = shadow.to(self.device)
            no_shadow = no_shadow.to(self.device)
            
            pred = self.model(shadow)
            loss = self.criterion(pred, no_shadow)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            if self.config.logging.use_swanlab and batch_idx % self.config.logging.log_freq == 0:
                swanlab.log({"train/step_loss": loss.item()}, step=self.global_step)
        
        return total_loss / len(self.train_loader)
    
    def _validate(self, epoch):
        """验证"""
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        
        with torch.no_grad():
            for shadow, no_shadow, _ in self.val_loader:
                shadow = shadow.to(self.device)
                no_shadow = no_shadow.to(self.device)
                
                pred = self.model(shadow)
                
                psnr = calculate_psnr(pred, no_shadow)
                ssim = calculate_ssim(pred, no_shadow)
                
                total_psnr += psnr
                total_ssim += ssim
        
        return total_psnr / len(self.val_loader), total_ssim / len(self.val_loader)
    
    def _update_lr(self, epoch):
        """更新学习率"""
        if epoch < self.warmup_epochs:
            lr = self.config.training.lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
        return lr
    
    def _save_checkpoint(self, epoch, val_psnr):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'global_step': self.global_step
        }
        
        torch.save(checkpoint, os.path.join(self.config.checkpoint.save_dir, 'latest_model.pth'))
        
        if val_psnr > self.best_psnr:
            self.best_psnr = val_psnr
            torch.save(checkpoint, os.path.join(self.config.checkpoint.save_dir, 'best_model.pth'))
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_psnr = checkpoint['best_psnr']
        self.global_step = checkpoint['global_step']


def main():
    config = OmegaConf.load('config/train_config.yaml')
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

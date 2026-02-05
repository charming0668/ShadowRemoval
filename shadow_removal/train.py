"""训练脚本 - 支持DDP多GPU训练"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
import swanlab
from tqdm import tqdm

from model_convnext import fusion_net
from dataset import ShadowRemovalDataset
from losses import CombinedLoss
from metrics import calculate_psnr, calculate_ssim


def setup_ddp():
    """初始化DDP"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """清理DDP"""
    dist.destroy_process_group()


def is_main_process():
    """判断是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_tensor(tensor):
    """多卡同步张量"""
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


class Trainer:
    """训练器"""
    
    def __init__(self, config, local_rank=-1):
        self.config = config
        self.local_rank = local_rank
        self.is_ddp = local_rank != -1
        
        if self.is_ddp:
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        if is_main_process():
            print(f"Loading model...")
        self.model = fusion_net().to(self.device)
        
        if self.is_ddp:
            self.model = DDP(
                self.model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=True  # 允许未使用的参数
            )
        
        # 创建数据加载器
        if is_main_process():
            print(f"Loading dataset from {config.data.train_dir}...")
        self.train_loader, self.val_loader = self._create_dataloaders()
        if is_main_process():
            print(f"Train samples: {len(self.train_loader.dataset)}, Val samples: {len(self.val_loader.dataset)}")
        
        self.criterion = CombinedLoss(
            alpha=config.training.alpha,
            beta=config.training.beta,
            gamma=config.training.gamma
        )
        
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
        
        if is_main_process():
            os.makedirs(config.checkpoint.save_dir, exist_ok=True)
            if config.logging.use_swanlab:
                swanlab.init(project="shadow-removal", config=OmegaConf.to_container(config, resolve=True))
        
        if config.model.resume:
            self._load_checkpoint(config.model.resume)
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        train_dataset = ShadowRemovalDataset(
            self.config.data.train_dir,
            crop_size=self.config.data.crop_size,
            is_train=True
        )
        val_dataset = ShadowRemovalDataset(
            self.config.data.val_dir,
            crop_size=self.config.data.crop_size,
            is_train=False
        )
        
        if self.is_ddp:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config.data.batch_size,
                sampler=train_sampler,
                num_workers=self.config.data.num_workers,
                pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.data.batch_size,
                sampler=val_sampler,
                num_workers=self.config.data.num_workers,
                pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=True,
                num_workers=self.config.data.num_workers,
                pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                pin_memory=True
            )
        
        return train_loader, val_loader
    
    def train(self):
        """训练"""
        if is_main_process():
            print(f"\nStarting training for {self.config.training.epochs} epochs...")
            print(f"Warmup epochs: {self.warmup_epochs}")
            print(f"Device: {self.device}\n")
        
        for epoch in range(self.start_epoch, self.config.training.epochs):
            if self.is_ddp:
                self.train_loader.sampler.set_epoch(epoch)
            
            train_loss = self._train_epoch(epoch)
            val_psnr, val_ssim = self._validate(epoch)
            current_lr = self._update_lr(epoch)
            
            if is_main_process():
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
        
        if is_main_process():
            print("\nTraining completed!")
    
    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        if is_main_process():
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        else:
            pbar = self.train_loader
        
        for batch_idx, (shadow, no_shadow, _) in enumerate(pbar):
            shadow = shadow.to(self.device)
            no_shadow = no_shadow.to(self.device)
            
            pred = self.model(shadow)
            loss = self.criterion(pred, no_shadow)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            if is_main_process():
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                if self.config.logging.use_swanlab and batch_idx % self.config.logging.log_freq == 0:
                    swanlab.log({"train/step_loss": loss.item()}, step=self.global_step)
        
        return total_loss / len(self.train_loader)
    
    def _validate(self, epoch):
        """验证 - 多卡同步"""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        
        if is_main_process():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        else:
            pbar = self.val_loader
        
        with torch.no_grad():
            for shadow, no_shadow, _ in pbar:
                shadow = shadow.to(self.device)
                no_shadow = no_shadow.to(self.device)
                
                pred = self.model(shadow)
                
                # 计算损失
                loss = self.criterion(pred, no_shadow)
                total_loss += loss.item()
                
                # 计算指标
                psnr = calculate_psnr(pred, no_shadow)
                ssim = calculate_ssim(pred, no_shadow)
                
                total_psnr += psnr
                total_ssim += ssim
                
                if is_main_process():
                    pbar.set_postfix({'psnr': f'{psnr:.2f}', 'ssim': f'{ssim:.4f}'})
        
        # 计算平均值
        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)
        
        # 多卡同步指标
        if self.is_ddp:
            metrics = torch.tensor([avg_loss, avg_psnr, avg_ssim], device=self.device)
            metrics = reduce_tensor(metrics)
            avg_loss, avg_psnr, avg_ssim = metrics[0].item(), metrics[1].item(), metrics[2].item()
        
        return avg_psnr, avg_ssim
    
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
        model_state = self.model.module.state_dict() if self.is_ddp else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'global_step': self.global_step
        }
        
        torch.save(checkpoint, os.path.join(self.config.checkpoint.save_dir, 'latest_model.pth'))
        
        if val_psnr > self.best_psnr:
            self.best_psnr = val_psnr
            torch.save(checkpoint, os.path.join(self.config.checkpoint.save_dir, 'best_model.pth'))
            print(f"  → Saved best model (PSNR: {val_psnr:.2f})")
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.is_ddp:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_psnr = checkpoint['best_psnr']
        self.global_step = checkpoint['global_step']
        
        if is_main_process():
            print(f"Loaded checkpoint from {checkpoint_path} (epoch {self.start_epoch})")


def main():
    config = OmegaConf.load('config/train_config.yaml')
    
    # 检查是否在DDP环境中
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    if local_rank != -1:
        local_rank = setup_ddp()
    
    trainer = Trainer(config, local_rank)
    trainer.train()
    
    if local_rank != -1:
        cleanup_ddp()


if __name__ == '__main__':
    main()

"""测试脚本"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from PIL import Image
import numpy as np

from model_convnext import fusion_net
from dataset import ShadowRemovalDataset
from metrics import calculate_psnr, calculate_ssim


def test(config, checkpoint_path):
    """测试模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = fusion_net().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_dataset = ShadowRemovalDataset(config.data.test_dir, crop_size=config.data.crop_size, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.data.num_workers)
    
    output_dir = config.test.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    total_psnr = 0.0
    total_ssim = 0.0
    
    with torch.no_grad():
        for idx, (shadow, no_shadow, filename) in enumerate(test_loader):
            shadow = shadow.to(device)
            no_shadow = no_shadow.to(device)
            
            pred = model(shadow)
            
            psnr = calculate_psnr(pred, no_shadow)
            ssim = calculate_ssim(pred, no_shadow)
            
            total_psnr += psnr
            total_ssim += ssim
            
            if config.test.save_images:
                pred_np = pred[0].cpu().numpy().transpose(1, 2, 0)
                pred_np = np.clip(pred_np * 255, 0, 255).astype(np.uint8)
                Image.fromarray(pred_np).save(os.path.join(output_dir, filename[0]))
            
            print(f"[{idx+1}/{len(test_loader)}] {filename[0]} - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
    
    print(f"\nAvg PSNR: {total_psnr/len(test_loader):.2f}, Avg SSIM: {total_ssim/len(test_loader):.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/test_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    test(config, args.checkpoint)


if __name__ == '__main__':
    main()

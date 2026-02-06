"""测试脚本"""
import os
import sys
import datetime
import shutil

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from PIL import Image
import numpy as np

from model_convnext import fusion_net
from dataset import ShadowRemovalDataset
from metrics import calculate_psnr, calculate_ssim, LPIPSMetric


def test(config, checkpoint_path):
    """测试模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = fusion_net().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_dataset = ShadowRemovalDataset(config.data.test_dir, crop_size=config.data.crop_size, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.data.num_workers)

    # Setup output directory with datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('test', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}")
    
    # Initialize Metrics
    lpips_metric = LPIPSMetric(use_gpu=torch.cuda.is_available())
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    
    results = [] # Store tuples of (psnr, ssim, lpips, temp_filename, original_name)
    
    with torch.no_grad():
        for idx, (shadow, no_shadow, filename) in enumerate(test_loader):
            shadow = shadow.to(device)
            no_shadow = no_shadow.to(device)
            
            pred = model(shadow)
            
            # Calculate metrics
            psnr = calculate_psnr(pred, no_shadow)
            ssim = calculate_ssim(pred, no_shadow)
            lpips_val = lpips_metric.calculate(pred, no_shadow)
            
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_val
            
            # Save concatenated image
            # Prepare images
            def tensor2img(t):
                return (t[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            
            img_shadow = tensor2img(shadow)
            img_pred = tensor2img(pred)
            img_gt = tensor2img(no_shadow)
            
            # Concatenate: Shadow | Pred | GT
            # Ensure heights match if needed, but usually they are same size
            concat_img = np.concatenate([img_shadow, img_pred, img_gt], axis=1)
            
            # Save to temporary file
            temp_filename = f"temp_{idx}.png"
            temp_path = os.path.join(output_dir, temp_filename)
            Image.fromarray(concat_img).save(temp_path)
            
            results.append({
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_val,
                "temp_path": temp_path,
                "original_name": filename[0]
            })
            
            print(f"[{idx+1}/{len(test_loader)}] {filename[0]} - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, LPIPS: {lpips_val:.4f}")
    
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    avg_lpips = total_lpips / len(test_loader)
    print(f"\nAvg PSNR: {avg_psnr:.2f}, Avg SSIM: {avg_ssim:.4f}, Avg LPIPS: {avg_lpips:.4f}")
    
    # Sort by PSNR descending
    results.sort(key=lambda x: x['psnr'], reverse=True)
    
    # Rename files
    print("Renaming files based on PSNR ranking...")
    for i, res in enumerate(results):
        base, ext = os.path.splitext(res['original_name'])
        if not ext: ext = ".png" 
        
        # Naming: {rank}_PSNR{val}_SSIM{val}_LPIPS{val}_{name}.png
        new_name = f"{i+1:03d}_PSNR{res['psnr']:.2f}_SSIM{res['ssim']:.3f}_LPIPS{res['lpips']:.3f}_{base}{ext}"
        new_path = os.path.join(output_dir, new_name)
        
        os.rename(res['temp_path'], new_path)
        
    print(f"All results saved to {output_dir}")


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

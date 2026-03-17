"""测试脚本"""
import os
import sys
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from PIL import Image
import numpy as np

from model_convnext import fusion_net
from shadow_removal.dataset import ShadowRemovalDataset
from shadow_removal.metrics import calculate_psnr, calculate_ssim, LPIPSMetric


def test(config, checkpoint_path):
    """测试模型
    
    Args:
        config: 配置对象
        checkpoint_path: 检查点路径
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = fusion_net().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # 数据集
    test_dataset = ShadowRemovalDataset(config.data.test_dir, crop_size=config.data.crop_size, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.data.num_workers)
    
    # 输出目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('test', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}")
    
    # 评估指标
    lpips_metric = LPIPSMetric(use_gpu=torch.cuda.is_available())
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    results = []
    
    with torch.no_grad():
        for idx, (shadow, no_shadow, filename) in enumerate(test_loader):
            shadow = shadow.to(device)
            no_shadow = no_shadow.to(device)
            
            # 前向传播
            pred = model(shadow)
            
            # 计算指标
            psnr = calculate_psnr(pred, no_shadow)
            ssim = calculate_ssim(pred, no_shadow)
            lpips_val = lpips_metric.calculate(pred, no_shadow)
            
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_val
            
            # 保存对比图
            def tensor2img(t):
                return (t[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            
            concat_img = np.concatenate([tensor2img(shadow), tensor2img(pred), tensor2img(no_shadow)], axis=1)
            temp_path = os.path.join(output_dir, f"temp_{idx}.png")
            Image.fromarray(concat_img).save(temp_path)
            
            results.append({
                "psnr": psnr, "ssim": ssim, "lpips": lpips_val,
                "temp_path": temp_path, "original_name": filename[0]
            })
            
            print(f"[{idx+1}/{len(test_loader)}] {filename[0]} - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, LPIPS: {lpips_val:.4f}")
    
    # 平均指标
    n = len(test_loader)
    print(f"\nAvg PSNR: {total_psnr/n:.2f}, Avg SSIM: {total_ssim/n:.4f}, Avg LPIPS: {total_lpips/n:.4f}")
    
    # 按PSNR排序并重命名
    results.sort(key=lambda x: x['psnr'], reverse=True)
    for i, res in enumerate(results):
        base, ext = os.path.splitext(res['original_name'])
        if not ext: ext = ".png"
        new_name = f"{i+1:03d}_PSNR{res['psnr']:.2f}_SSIM{res['ssim']:.3f}_LPIPS{res['lpips']:.3f}_{base}{ext}"
        os.rename(res['temp_path'], os.path.join(output_dir, new_name))
    
    print(f"All results saved to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Shadow Removal Test')
    parser.add_argument('--config', type=str, default='config/test_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    test(config, args.checkpoint)


if __name__ == '__main__':
    main()

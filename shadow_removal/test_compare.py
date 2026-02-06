"""模型对比测试脚本: 自训练模型 vs shadowremoval.pkl"""
import os
import sys
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from PIL import Image

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_convnext import fusion_net
from dataset import ShadowRemovalDataset
from metrics import calculate_psnr, calculate_ssim, LPIPSMetric

def load_weights(model, path, device, is_checkpoint=False):
    """加载权重辅助函数"""
    try:
        state_dict = torch.load(path, map_location=device)
        # 如果是训练的checkpoint，通常权重在'model_state_dict'键下
        if is_checkpoint and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # 尝试加载
        model.load_state_dict(state_dict, strict=True)
        print(f"[Success] Successfully loaded weights from: {path}")
        return True
    except Exception as e:
        print(f"[Error] Failed to load weights from {path}. Error: {e}")
        return False

def test_compare(config, my_checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ref_weight_path = '/data1/hmcai/Shadow_R/weights/shadowremoval.pkl'
    
    print("=== Initializing Models ===")
    # 1. 初始化并加载我的模型
    model_my = fusion_net().to(device)
    print(f"Loading My Checkpoint: {my_checkpoint_path}")
    if not load_weights(model_my, my_checkpoint_path, device, is_checkpoint=True):
        print("Aborting: Could not load my checkpoint.")
        return
    model_my.eval()

    # 2. 初始化并加载参考模型 (shadowremoval.pkl)
    model_ref = fusion_net().to(device)
    print(f"Loading Ref Weights: {ref_weight_path}")
    if not os.path.exists(ref_weight_path):
        print(f"Error: Reference weight file not found at {ref_weight_path}")
        return
    
    if not load_weights(model_ref, ref_weight_path, device, is_checkpoint=False):
        print("Aborting: Could not load reference weights. Structure might confirm to network?")
        return
    model_ref.eval()

    # 3. 数据集准备
    test_dataset = ShadowRemovalDataset(config.data.test_dir, crop_size=config.data.crop_size, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.data.num_workers)

    # 4. 输出设置
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('test', f"compare_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving comparison results to {output_dir}")
    
    lpips_metric = LPIPSMetric(use_gpu=torch.cuda.is_available())
    results = [] 

    print("=== Starting Comparison Test ===")
    with torch.no_grad():
        for idx, (shadow, no_shadow, filename) in enumerate(test_loader):
            shadow = shadow.to(device)
            no_shadow = no_shadow.to(device)
            
            # Forward pass
            out_my = model_my(shadow)
            out_ref = model_ref(shadow)
            
            # Calculated metrics (Using my model's performance for sorting)
            psnr_my = calculate_psnr(out_my, no_shadow)
            ssim_my = calculate_ssim(out_my, no_shadow)
            lpips_my = lpips_metric.calculate(out_my, no_shadow)
            
            psnr_ref = calculate_psnr(out_ref, no_shadow)
            
            # Prepare images for concatenation
            def tensor2img(t):
                return (t[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            
            img_shadow = tensor2img(shadow)
            img_my = tensor2img(out_my)
            img_ref = tensor2img(out_ref)
            img_gt = tensor2img(no_shadow)
            
            # Concatenate: Shadow | My Model | Ref Model | GT
            concat_img = np.concatenate([img_shadow, img_my, img_ref, img_gt], axis=1)
            
            # Save temp
            temp_filename = f"temp_{idx}.png"
            temp_path = os.path.join(output_dir, temp_filename)
            Image.fromarray(concat_img).save(temp_path)
            
            print(f"[{idx+1}/{len(test_loader)}] {filename[0]}")
            print(f"   My Model - PSNR: {psnr_my:.2f}, SSIM: {ssim_my:.4f}")
            print(f"   Ref Model - PSNR: {psnr_ref:.2f}")

            results.append({
                "psnr": psnr_my, # Sort by my model's PSNR
                "ssim": ssim_my,
                "lpips": lpips_my,
                "psnr_ref": psnr_ref,
                "temp_path": temp_path,
                "original_name": filename[0]
            })

    # Sort and Rename
    results.sort(key=lambda x: x['psnr'], reverse=True)
    
    print("\nRenaming files based on My Model PSNR ranking...")
    for i, res in enumerate(results):
        base, ext = os.path.splitext(res['original_name'])
        if not ext: ext = ".png"
        
        # Naming: Rank_MyPSNR_RefPSNR_Name.png
        new_name = (f"{i+1:03d}_MyPSNR{res['psnr']:.2f}_RefPSNR{res['psnr_ref']:.2f}_"
                    f"SSIM{res['ssim']:.3f}_LPIPS{res['lpips']:.3f}_{base}{ext}")
        
        os.rename(res['temp_path'], os.path.join(output_dir, new_name))
        
    print(f"Done. Check results in {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/test_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to your trained checkpoint")
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    test_compare(config, args.checkpoint)

if __name__ == '__main__':
    main()

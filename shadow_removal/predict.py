"""预测脚本 - 对图像进行阴影去除预测"""
import os
import sys
import argparse
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from model_convnext import fusion_net


class SingleImageDataset(Dataset):
    """单图像数据集"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = sorted([f for f in os.listdir(data_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.data_dir, filename)
        img = Image.open(img_path).convert('RGB')
        return self.to_tensor(img), filename


def predict(checkpoint_path, input_dir, output_dir=None, device='cuda'):
    """对输入目录中的图像进行阴影去除预测
    
    Args:
        checkpoint_path: 模型检查点路径
        input_dir: 输入图像目录
        output_dir: 输出目录
        device: 计算设备
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 输出目录
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"predict/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = fusion_net().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # 数据集
    dataset = SingleImageDataset(input_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Found {len(dataset)} images")
    
    with torch.no_grad():
        for idx, (image, filename) in enumerate(dataloader):
            image = image.to(device)
            pred = model(image)
            
            # 保存结果
            pred_np = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(pred_np).save(os.path.join(output_dir, filename[0]))
            print(f"[{idx+1}/{len(dataset)}] {filename[0]}")
    
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Shadow Removal Prediction')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    args = parser.parse_args()
    
    predict(args.checkpoint, args.input, args.output, args.device)


if __name__ == '__main__':
    main()

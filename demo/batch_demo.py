# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
import cv2
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from mmseg.apis import inference_segmentor, init_segmentor

# ================= 配置区 =================
# Pascal VOC / Context 标准调色板
PASCAL_PALETTE = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0],
    [128, 128, 0], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
    [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
    [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
    [128, 192, 0], [0, 64, 128], [128, 64, 128], [0, 192, 128], [128, 192, 128],
    [64, 64, 0], [192, 64, 0], [64, 192, 0], [192, 192, 0], [64, 64, 128],
    [192, 64, 128], [64, 192, 128], [192, 192, 128], [0, 0, 64], [128, 0, 64],
    [0, 128, 64], [128, 128, 64], [0, 0, 192], [128, 0, 192], [0, 128, 192],
    [128, 128, 192], [64, 0, 64], [192, 0, 64], [64, 128, 64], [192, 128, 64],
    [64, 0, 192], [192, 0, 192], [64, 128, 192], [192, 128, 192], [0, 64, 64]
]

def apply_mask(image, mask, color):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] + color[c],
                                  image[:, :, c])
    return image

def visualize_prediction(path, pred, palette):
    """
    修正后的可视化函数，接收外部传入的 palette
    """
    n, h, w = pred.shape
    # 初始化白色背景或黑色背景，这里用白色方便查看
    image = np.zeros((h, w, 3), dtype=np.float32)
    
    boundary_sum = np.zeros((h, w), dtype=np.float32)

    # 确保调色板长度足够
    if n > len(palette):
        print(f"Warning: Palette length {len(palette)} < Num classes {n}")
    
    for i in range(n):
        # 防止越界
        color = palette[i] if i < len(palette) else [255, 255, 255]
        boundary = pred[i,:,:]
        boundary_sum = boundary_sum + boundary
        image = apply_mask(image, boundary, color)

    boundary_sum_3c = np.stack([boundary_sum]*3, axis=2)
    idx = boundary_sum_3c > 0
    
    # 归一化处理重叠部分
    image[idx] = image[idx] / boundary_sum_3c[idx]
    
    # 背景设为白色 (255) 或保持黑色 (0)，原代码设为 255
    image[~idx] = 255 
    
    cv2.imwrite(path, image[..., ::-1].astype(np.uint8))

def mask_to_onehot(mask, num_classes):
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot_to_multiclass_boundarys(mask, radius, num_classes):
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='reflect')
    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)
    return np.array(channels)

def make_seg_overlay(img_path, seg_pred, palette, opacity=0.5):
    """生成原图+分割掩码的叠加图"""
    img = cv2.imread(img_path)
    h, w = seg_pred.shape
    seg_color = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label, color in enumerate(palette):
        seg_color[seg_pred == label] = color
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    seg_rgb = seg_color.astype(np.float32)
    overlay = img_rgb * (1 - opacity) + seg_rgb * opacity
    return cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)

def main():
    parser = ArgumentParser()
    # 将 img 参数改为 img_dir，支持文件夹
    parser.add_argument('img_dir', help='Path to image directory (e.g., JPEGImages)')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('out_dir', help='Directory to save results')
    
    parser.add_argument('--num_classes', type=int, default=60, 
                        help='Number of classes (Pascal Context=60, VOC=21)')
    parser.add_argument('--device', default='cuda', help='Device used for inference')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity')
    
    args = parser.parse_args()

    # 1. 准备输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 2. 获取所有图片
    if os.path.isdir(args.img_dir):
        # 查找 jpg 和 png
        img_list = glob(os.path.join(args.img_dir, '*.jpg')) + \
                   glob(os.path.join(args.img_dir, '*.png'))
        img_list.sort()
    else:
        print(f"Error: {args.img_dir} is not a directory.")
        return

    print(f"Found {len(img_list)} images in {args.img_dir}")
    if len(img_list) == 0:
        return

    # 3. 初始化模型
    print("Initializing model...")
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    # 4. 批量推理
    for img_path in tqdm(img_list):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        try:
            result = inference_segmentor(model, img_path)
            seg_pred, bound_pred = result
            # seg_pred 是 list, 取第一个元素 [H, W]
            seg_mask = seg_pred[0]
            
            # === 保存 1: 语义分割叠加图 (Overlay) ===
            overlay = make_seg_overlay(img_path, seg_mask, PASCAL_PALETTE, args.opacity)
            cv2.imwrite(os.path.join(args.out_dir, f"{img_name}_overlay.png"), overlay)

            # === 保存 2: 语义边界 (Semantic Boundary) ===
            # 使用传入的 num_classes
            onehot_mask = mask_to_onehot(seg_mask, args.num_classes) 
            sebound_mask = onehot_to_multiclass_boundarys(onehot_mask, 2, args.num_classes)
            
            visualize_prediction(
                os.path.join(args.out_dir, f"{img_name}_sebound.png"), 
                sebound_mask, 
                PASCAL_PALETTE
            )

            # === 保存 3: 二值边界 (Binary Boundary) ===
            # bound_pred 是 list
            b_bound = (bound_pred[0] * 255.0).astype(np.uint8)
            b_bound_color = cv2.applyColorMap(b_bound, 13) # 13 is a colormap style
            cv2.imwrite(os.path.join(args.out_dir, f"{img_name}_bibound.png"), b_bound_color)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print(f"Done! Results saved to {args.out_dir}")

if __name__ == '__main__':
    main()
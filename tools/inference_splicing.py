# inference_pascal.py
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from mmseg.apis import init_segmentor, inference_segmentor
from scipy.ndimage import distance_transform_edt

# ==================== 配置区 ====================
CONFIG_FILE = 'configs/Mobile_Seed/MS_tiny_pascal_transfer.py'
CHECKPOINT_FILE = '/root/Mobile-Seed/work_dirs/MS_tiny_pascal_transfer/v5/iter_1000.pth'
INPUT_PATH = '/root/Mobile-Seed/data/test/test_image/'
OUTPUT_DIR = '/root/Mobile-Seed/data/test/results_pascal_v5_1000/'
OPACITY = 0.5
NUM_CLASSES = 60     # Pascal Context
# ================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = init_segmentor(CONFIG_FILE, CHECKPOINT_FILE, device=device)

# ======= Pascal Palette（你之前给过，我完整复制） =======
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

# ============== 工具函数（与 Cityscapes 版一致） ==============

def mask_to_onehot(mask, num_classes):
    return np.array([mask == i for i in range(num_classes)], dtype=np.uint8)

def onehot_to_multiclass_boundarys(mask, radius, num_classes):
    mask_pad = np.pad(mask, ((0,0),(1,1),(1,1)), mode='reflect')
    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i]) + distance_transform_edt(1 - mask_pad[i])
        dist = dist[1:-1, 1:-1]
        dist = (dist <= radius).astype(np.uint8)
        channels.append(dist)
    return np.array(channels)

def apply_mask(image, mask, color):
    for c in range(3):
        image[:,:,c] = np.where(mask==1, image[:,:,c] + color[c], image[:,:,c])
    return image

def visualize_sebound_img(pred, palette):
    n, h, w = pred.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    boundary_sum = np.zeros((h, w), dtype=np.float32)

    for i in range(n):
        boundary = pred[i]
        boundary_sum += boundary
        color = palette[i % len(palette)]
        img = apply_mask(img, boundary, color)

    boundary_sum_img = np.stack([boundary_sum] * 3, axis=2)
    idx = boundary_sum_img > 0
    img[idx] = img[idx] / boundary_sum_img[idx]
    img[~idx] = 255

    img = np.clip(img, 0, 255).astype(np.uint8)
    # 返回 BGR
    return img[..., ::-1]

def make_seg_overlay(img_bgr, seg_pred, palette, opacity=0.5):
    h,w = seg_pred.shape
    seg_color = np.zeros((h,w,3), dtype=np.uint8)
    for label,color in enumerate(palette):
        seg_color[seg_pred==label] = color
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    seg_rgb = seg_color.astype(np.float32)
    overlay = img_rgb*(1-opacity) + seg_rgb*opacity
    return cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)

def make_1x4_grid(img_bgr, overlay_bgr, sebound_bgr, bibound_bgr):
    """按 1×4 横向排列"""
    h, w = img_bgr.shape[:2]
    overlay_bgr = cv2.resize(overlay_bgr, (w, h))
    sebound_bgr = cv2.resize(sebound_bgr, (w, h))
    bibound_bgr = cv2.resize(bibound_bgr, (w, h))
    grid = np.hstack((img_bgr, overlay_bgr, sebound_bgr, bibound_bgr))
    return grid

# ================= 图像列表 =================
input_path = Path(INPUT_PATH)
image_paths = [input_path] if input_path.is_file() else \
    [p for p in input_path.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']]

print(f"🔍 找到 {len(image_paths)} 张图像，开始推理...")

# ======================= 推理 =======================
for img_path in image_paths:
    try:
        img_bgr = cv2.imread(str(img_path))
        result = inference_segmentor(model, str(img_path))

        # ❗Pascal 返回两个：seg_list, bibound_list
        seg_list, bibound_list = result
        seg_pred = seg_list[0]
        bibound_pred = bibound_list[0]

        name = img_path.stem

        # 1️⃣ seg overlay
        overlay_bgr = make_seg_overlay(img_bgr, seg_pred, PASCAL_PALETTE, opacity=OPACITY)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_seg_overlay.png"), overlay_bgr)

        # 2️⃣ sebound（由 seg_pred 推导）
        onehot = mask_to_onehot(seg_pred, NUM_CLASSES)
        sebound = onehot_to_multiclass_boundarys(onehot, radius=2, num_classes=NUM_CLASSES)
        sebound_bgr = visualize_sebound_img(sebound, PASCAL_PALETTE)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_sebound.png"), sebound_bgr)

        # 3️⃣ bibound（二值边界，可视化）
        bibound = (bibound_pred * 255).astype(np.uint8)
        bibound_bgr = cv2.applyColorMap(bibound, 13)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_bibound.png"), bibound_bgr)

        # 4️⃣ 1×4 横向拼图
        grid_1x4 = make_1x4_grid(img_bgr, overlay_bgr, sebound_bgr, bibound_bgr)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_1x4.png"), grid_1x4)

        print(f"✅ 已处理: {img_path}")

    except Exception as e:
        print(f"❌ 处理失败 {img_path}: {e}")

print(f"\n🎉 Pascal Context 推理完成！输出目录：{OUTPUT_DIR}")

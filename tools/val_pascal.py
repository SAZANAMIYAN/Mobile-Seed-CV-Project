import os
import cv2
import torch
import numpy as np
from pathlib import Path
from mmseg.apis import init_segmentor, inference_segmentor
from scipy.ndimage import distance_transform_edt

# ==================== 配置区 ====================
CONFIG_FILE = 'configs/Mobile_Seed/MS_tiny_pascal_context.py'
CHECKPOINT_FILE = '/root/Mobile-Seed/work_dirs/MS_tiny_pascal_transfer/v5/iter_1000.pth'

# 您的数据集根目录
PASCAL_ROOT = '/root/Mobile-Seed/data/VOCdevkit/VOC2010'

OUTPUT_DIR = '/root/Mobile-Seed/data/test/results_pascal_train_v5_1000/'
OPACITY = 0.5
NUM_CLASSES = 60
# ================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"正在加载模型: {CHECKPOINT_FILE} ...")
model = init_segmentor(CONFIG_FILE, CHECKPOINT_FILE, device=device)

# ======= Pascal Palette =======
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

# ============== 工具函数 ==============
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

def visualize_prediction(path, pred, palette):
    n,h,w = pred.shape
    img = np.zeros((h,w,3), dtype=np.float32)
    sum_map = np.zeros((h,w), dtype=np.float32)
    for i in range(n):
        boundary = pred[i]
        sum_map += boundary
        img = apply_mask(img, boundary, palette[i])
    sum_map3 = np.stack([sum_map]*3, axis=2)
    idx = sum_map3 > 0
    img[idx] = img[idx] / sum_map3[idx]
    img[~idx] = 255
    cv2.imwrite(path, img[...,::-1].astype(np.uint8))

def make_seg_overlay(img_bgr, seg_pred, palette, opacity=0.5):
    h,w = seg_pred.shape
    seg_color = np.zeros((h,w,3), dtype=np.uint8)
    for label,color in enumerate(palette):
        seg_color[seg_pred==label] = color
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    seg_rgb = seg_color.astype(np.float32)
    overlay = img_rgb*(1-opacity) + seg_rgb*opacity
    return cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)

# ================= 获取验证集列表 =================
img_dir = os.path.join(PASCAL_ROOT, 'JPEGImages')

'''
# 尝试查找 val.txt 的所有可能位置
candidates = [
    os.path.join(PASCAL_ROOT, 'ImageSets', 'Segmentation', 'val.txt'),
    os.path.join(PASCAL_ROOT, 'ImageSets', 'Main', 'val.txt'),
    os.path.join(PASCAL_ROOT, 'ImageSets', 'Layout', 'val.txt'),
]
'''

# 尝试查找 train.txt 的所有可能位置
candidates = [
    os.path.join(PASCAL_ROOT, 'ImageSets', 'Segmentation', 'train.txt'),
    os.path.join(PASCAL_ROOT, 'ImageSets', 'Main', 'train.txt'),
    os.path.join(PASCAL_ROOT, 'ImageSets', 'Layout', 'train.txt'),
]

val_list_path = None
for p in candidates:
    if os.path.exists(p):
        val_list_path = p
        break

if val_list_path is None:
    print("\n❌ 严重错误：无法自动找到 val.txt！")
    print(f"请检查您的目录结构: {os.path.join(PASCAL_ROOT, 'ImageSets')}")
    print("请手动修改脚本中的 val_list_path 为正确的绝对路径。")
    raise FileNotFoundError(f"无法在 {PASCAL_ROOT} 中找到 val.txt")

print(f"📖 成功定位验证集列表: {val_list_path}")
with open(val_list_path, 'r') as f:
    file_names = [x.strip() for x in f.readlines()]

image_paths = []
for name in file_names:
    # 移除可能存在的扩展名（有些txt里带.jpg）
    clean_name = os.path.splitext(name)[0]
    
    p_jpg = os.path.join(img_dir, clean_name + '.jpg')
    p_png = os.path.join(img_dir, clean_name + '.png')
    
    if os.path.exists(p_jpg):
        image_paths.append(Path(p_jpg))
    elif os.path.exists(p_png):
        image_paths.append(Path(p_png))

if len(image_paths) == 0:
    print(f"⚠️ 警告：读取了 {len(file_names)} 个文件名，但在 {img_dir} 下没有找到对应的图片。")
    print("请检查 JPEGImages 文件夹是否为空，或者文件名是否匹配。")
else:
    print(f"🔍 验证集共包含 {len(file_names)} 个样本，找到 {len(image_paths)} 张图像文件。")
    print("🚀 开始推理...")

# ======================= 推理 =======================
for i, img_path in enumerate(image_paths):
    try:
        img_bgr = cv2.imread(str(img_path))
        result = inference_segmentor(model, str(img_path))

        seg_list, bibound_list = result
        seg_pred = seg_list[0]
        bibound_pred = bibound_list[0]

        name = img_path.stem

        # 1️⃣ seg overlay
        overlay = make_seg_overlay(img_bgr, seg_pred, PASCAL_PALETTE, opacity=OPACITY)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_seg_overlay.png"), overlay)

        # 2️⃣ sebound
        onehot = mask_to_onehot(seg_pred, NUM_CLASSES)
        sebound = onehot_to_multiclass_boundarys(onehot, 2, NUM_CLASSES)
        visualize_prediction(os.path.join(OUTPUT_DIR, f"{name}_sebound.png"), sebound, PASCAL_PALETTE)

        # 3️⃣ bibound
        bibound = (bibound_pred * 255).astype(np.uint8)
        bibound = cv2.applyColorMap(bibound, 13)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_bibound.png"), bibound)

        if i % 10 == 0:
            print(f"✅ [{i}/{len(image_paths)}] 已处理: {name}")

    except Exception as e:
        print(f"❌ 处理失败 {img_path}: {e}")

print(f"\n🎉 Pascal Context 验证集推理完成！输出目录：{OUTPUT_DIR}")
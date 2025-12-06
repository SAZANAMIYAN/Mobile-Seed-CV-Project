import os
from PIL import Image
from tqdm import tqdm

def splice_images_1x4(input_dir, output_dir):
    """
    将指定文件夹下的 原图(.jpg), seg_overlay, sebound, bibound 图片按顺序横向拼接。
    """
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有以 .jpg 结尾的文件作为基准（因为原图是 .jpg）
    # 排除可能存在的非目标jpg文件，这里假设文件名格式为 2008_xxxxxx.jpg
    files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    
    print(f"Found {len(files)} sets of images to process in {input_dir}...")

    for filename in tqdm(files):
        # 1. 解析文件 ID (例如: 2008_000009.jpg -> 2008_000009)
        file_id = os.path.splitext(filename)[0]

        # 2. 构建四个文件的完整路径
        # 顺序: 原图 -> seg_overlay -> sebound -> bibound
        path_origin = os.path.join(input_dir, f"{file_id}.jpg")
        path_seg = os.path.join(input_dir, f"{file_id}_seg_overlay.png")
        path_se = os.path.join(input_dir, f"{file_id}_sebound.png")
        path_bi = os.path.join(input_dir, f"{file_id}_bibound.png")

        # 检查所有组件是否存在
        if not (os.path.exists(path_origin) and os.path.exists(path_seg) and 
                os.path.exists(path_se) and os.path.exists(path_bi)):
            # 只有当缺少文件时才打印跳过信息，避免刷屏
            # print(f"Skipping {file_id}: Missing one or more components.")
            continue

        try:
            # 3. 打开图片
            img_origin = Image.open(path_origin).convert('RGB') # 确保原图也是RGB模式
            img_seg = Image.open(path_seg).convert('RGB')
            img_se = Image.open(path_se).convert('RGB')
            img_bi = Image.open(path_bi).convert('RGB')

            # 获取尺寸 (以原图为基准)
            width, height = img_origin.size

            # 简单的尺寸检查，如果后续图片尺寸不一致，强制缩放到原图大小
            # 这样可以防止因为尺寸微小差异导致的报错
            if img_seg.size != (width, height): img_seg = img_seg.resize((width, height))
            if img_se.size != (width, height): img_se = img_se.resize((width, height))
            if img_bi.size != (width, height): img_bi = img_bi.resize((width, height))

            # 4. 创建新的空白画布
            # 宽度 = 4张图的宽度之和
            total_width = width * 4
            new_img = Image.new('RGB', (total_width, height))

            # 5. 粘贴图片 (顺序: 原图 -> seg -> sebound -> bibound)
            new_img.paste(img_origin, (0, 0))         # 第一张: 原图
            new_img.paste(img_seg, (width, 0))        # 第二张: seg_overlay
            new_img.paste(img_se, (width * 2, 0))     # 第三张: sebound
            new_img.paste(img_bi, (width * 3, 0))     # 第四张: bibound

            # 6. 保存图片
            output_filename = f"{file_id}_1x4.png"
            output_path = os.path.join(output_dir, output_filename)
            new_img.save(output_path)

        except Exception as e:
            print(f"Error processing {file_id}: {e}")

    print("Done! All images saved to:", output_dir)

if __name__ == '__main__':
    # ================= 配置区域 =================
    # 输入文件夹路径
    input_folder = '/root/Mobile-Seed/data/test/results_pascal_train_v5_1000' 
    
    # 输出文件夹路径
    output_folder = '/root/Mobile-Seed/data/test/results_pascal_train_v5_1000'
    # ===========================================

    splice_images_1x4(input_folder, output_folder)
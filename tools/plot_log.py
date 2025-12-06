import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ================= 配置区域 =================
# 定义需要对比的模型信息
MODELS_CONFIG = [
    {
        "name": "tuned model",
        "path": "/root/Mobile-Seed/work_dirs/MS_tiny_pascal_transfer/v5_dense/v5_dense_log.json",
        "color": "#d62728",  # 红色系
        "style": "-"
    },
    {
        "name": "original model",
        "path": "/root/Mobile-Seed/work_dirs/MS_tiny_pascal_context/v0_dense/v0_dense_log.json",
        "color": "#1f77b4",  # 蓝色系
        "style": "-"
    }
]

# 图片保存目录
OUTPUT_DIR = "./work_dirs/comparison_plots" 
# ===========================================

def parse_json_log(file_path):
    """
    解析单个日志文件
    """
    train_data = {'iter': [], 'loss': [], 'acc_seg': []}
    val_data = {'iter': [], 'mIoU': [], 'aAcc': []}
    
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}，跳过该模型。")
        return None, None

    print(f"正在读取日志: {file_path} ...")
    last_train_iter = 0 

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                log_entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            mode = log_entry.get('mode', '')

            if mode == 'train':
                if 'iter' in log_entry and 'loss' in log_entry:
                    current_iter = log_entry['iter']
                    train_data['iter'].append(current_iter)
                    train_data['loss'].append(log_entry['loss'])
                    last_train_iter = current_iter 
                    if 'acc_seg' in log_entry:
                        train_data['acc_seg'].append(log_entry['acc_seg'])

            elif mode == 'val':
                if 'mIoU' in log_entry:
                    # 如果日志里val没有iter，就用最近一次train的iter
                    val_iter = log_entry.get('iter', last_train_iter)
                    val_data['iter'].append(val_iter)
                    val_data['mIoU'].append(log_entry['mIoU'])
                    if 'aAcc' in log_entry:
                        val_data['aAcc'].append(log_entry['aAcc'])

    return train_data, val_data

def set_global_y_axis_scale(all_data_lists, padding_factor=2.0):
    """
    升级版辅助函数：
    接收多个数据列表（例如模型A的loss和模型B的loss），
    计算全局最大最小值，然后设置更宽的Y轴范围，使曲线更平滑。
    """
    # 展平所有数据以寻找全局极值
    flat_data = []
    for sublist in all_data_lists:
        if sublist: # 确保子列表不为空
            flat_data.extend(sublist)
        
    if not flat_data:
        return
    
    y_min = min(flat_data)
    y_max = max(flat_data)
    diff = y_max - y_min
    
    # 如果数据是一条直线
    if diff == 0:
        diff = 0.1 * abs(y_max) if y_max != 0 else 1.0
        
    # 上下各扩展 diff * padding_factor 的距离
    new_min = y_min - (diff * padding_factor)
    new_max = y_max + (diff * padding_factor)
    
    plt.ylim(new_min, new_max)

def plot_comparison(parsed_models, output_dir):
    """
    绘制对比图
    parsed_models: 字典列表，包含配置信息和解析后的数据
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    # ==========================================
    # 1. 绘制 Training Loss 对比
    # ==========================================
    plt.figure(figsize=(12, 6))
    all_loss_data = []
    
    has_data = False
    for model in parsed_models:
        t_data = model['train_data']
        if t_data and t_data['loss']:
            plt.plot(t_data['iter'], t_data['loss'], 
                     label=f"{model['name']} Loss", 
                     color=model['color'], 
                     linestyle='-', linewidth=1.5)
            all_loss_data.append(t_data['loss'])
            has_data = True

    if has_data:
        plt.title('Training Loss Comparison', fontsize=14)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
        
        # Loss 波动大，padding 给大一点 (2.0)
        set_global_y_axis_scale(all_loss_data, padding_factor=2.0)
        
        save_path = os.path.join(output_dir, 'compare_train_loss.png')
        plt.savefig(save_path)
        print(f"生成图片: {save_path}")
    plt.close()

    # ==========================================
    # 2. 绘制 Training Accuracy 对比
    # ==========================================
    plt.figure(figsize=(12, 6))
    all_acc_data = []
    
    has_data = False
    for model in parsed_models:
        t_data = model['train_data']
        if t_data and t_data['acc_seg']:
            min_len = min(len(t_data['iter']), len(t_data['acc_seg']))
            plt.plot(t_data['iter'][:min_len], t_data['acc_seg'][:min_len], 
                     label=f"{model['name']} Pixel Acc", 
                     color=model['color'], 
                     linestyle='-', alpha=0.8, linewidth=1.5)
            all_acc_data.append(t_data['acc_seg'][:min_len])
            has_data = True

    if has_data:
        plt.title('Training Pixel Accuracy Comparison', fontsize=14)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
        
        # Acc 波动也可能较大，padding 给 2.0
        set_global_y_axis_scale(all_acc_data, padding_factor=2.0)
        
        save_path = os.path.join(output_dir, 'compare_train_acc.png')
        plt.savefig(save_path)
        print(f"生成图片: {save_path}")
    plt.close()

    # ==========================================
    # 3. 绘制 Validation Metrics 对比 (已修改)
    # ==========================================
    plt.figure(figsize=(12, 6))
    has_data = False
    all_val_data_for_scale = [] # 用于收集所有验证集数据来计算Y轴范围
    
    for model in parsed_models:
        v_data = model['val_data']
        if v_data and v_data['mIoU']:
            # 绘制 mIoU (实线)
            plt.plot(v_data['iter'], v_data['mIoU'], 
                     label=f"{model['name']} mIoU", 
                     color=model['color'], 
                     marker='o', markersize=4, linestyle='-', linewidth=2)
            all_val_data_for_scale.append(v_data['mIoU'])
            
            # 绘制 aAcc (虚线)
            if v_data['aAcc']:
                plt.plot(v_data['iter'], v_data['aAcc'], 
                         label=f"{model['name']} aAcc", 
                         color=model['color'], 
                         marker='', linestyle='--', alpha=0.6, linewidth=1.5)
                all_val_data_for_scale.append(v_data['aAcc'])
            
            has_data = True

    if has_data:
        plt.title('Validation Metrics Comparison (Solid: mIoU, Dashed: aAcc)', fontsize=14)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
        plt.xticks(rotation=45)

        # --- 修改开始 ---
        # 不再强制使用 ylim(0, 1.0)
        # 使用 padding_factor=0.2，这意味着如果数据范围是 0.1，上下各留 0.02 的空间
        # 这样可以“放大”曲线的波动细节
        set_global_y_axis_scale(all_val_data_for_scale, padding_factor=0.2)
        # --- 修改结束 ---
        
        save_path = os.path.join(output_dir, 'compare_val_metrics.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"生成图片: {save_path}")
    plt.close()

def main():
    # 1. 解析所有模型数据
    parsed_models = []
    for config in MODELS_CONFIG:
        train_data, val_data = parse_json_log(config['path'])
        if train_data:
            # 将数据合并回配置字典中
            model_info = config.copy()
            model_info['train_data'] = train_data
            model_info['val_data'] = val_data
            parsed_models.append(model_info)
    
    if not parsed_models:
        print("没有解析到任何有效数据，程序结束。")
        return

    # 2. 绘图
    print("-" * 30)
    print(f"开始绘图，输出目录: {OUTPUT_DIR}")
    plot_comparison(parsed_models, OUTPUT_DIR)
    print("所有绘图任务完成。")

if __name__ == "__main__":
    main()



'''import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ================= 配置区域 =================
LOG_FILE_PATH = '/root/Mobile-Seed/work_dirs/MS_tiny_pascal_context/20251206_004938/20251206_004939.log.json'
# ===========================================

def parse_json_log(file_path):
    """
    解析日志
    """
    train_data = {'iter': [], 'loss': [], 'acc_seg': []}
    val_data = {'iter': [], 'mIoU': [], 'aAcc': []}
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return None, None

    print(f"正在读取日志: {file_path} ...")
    last_train_iter = 0 

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                log_entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            mode = log_entry.get('mode', '')

            if mode == 'train':
                if 'iter' in log_entry and 'loss' in log_entry:
                    current_iter = log_entry['iter']
                    train_data['iter'].append(current_iter)
                    train_data['loss'].append(log_entry['loss'])
                    last_train_iter = current_iter 
                    if 'acc_seg' in log_entry:
                        train_data['acc_seg'].append(log_entry['acc_seg'])

            elif mode == 'val':
                if 'mIoU' in log_entry:
                    val_iter = last_train_iter
                    val_data['iter'].append(val_iter)
                    val_data['mIoU'].append(log_entry['mIoU'])
                    if 'aAcc' in log_entry:
                        val_data['aAcc'].append(log_entry['aAcc'])

    return train_data, val_data

def set_y_axis_scale(data_list, padding_factor=2.0):
    """
    辅助函数：计算更宽的 Y 轴范围，使曲线看起来更平滑
    padding_factor: 越大，留白越多，曲线看起来越平（波动越小）
    """
    if not data_list:
        return
    
    y_min = min(data_list)
    y_max = max(data_list)
    diff = y_max - y_min
    
    # 如果数据是一条直线（diff=0），默认给一个范围
    if diff == 0:
        diff = 0.1 * y_max if y_max != 0 else 1.0
        
    # 上下各扩展 diff * padding_factor 的距离
    new_min = y_min - (diff * padding_factor)
    new_max = y_max + (diff * padding_factor)
    
    plt.ylim(new_min, new_max)

def plot_metrics(train_data, val_data, output_dir):
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    # ==========================================
    # 1. 绘制 Training Loss
    # ==========================================
    if train_data['iter'] and train_data['loss']:
        plt.figure(figsize=(12, 6))
        plt.plot(train_data['iter'], train_data['loss'], label='Train Loss', color='#d62728', linewidth=1.5)
        plt.title('Training Loss Curve', fontsize=14)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # X轴设置：防止重叠
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
        
        # === Y轴修改：扩大范围，减少视觉波动 ===
        # padding_factor=2.0 表示上下留白是数据波动幅度的2倍，曲线会被压缩在中间1/5的区域
        set_y_axis_scale(train_data['loss'], padding_factor=2.0)
        
        save_path = os.path.join(output_dir, 'plot_train_loss.png')
        plt.savefig(save_path)
        print(f"生成图片: {save_path}")
        plt.close()

    # ==========================================
    # 2. 绘制 Training Accuracy
    # ==========================================
    if train_data['iter'] and train_data['acc_seg']:
        min_len = min(len(train_data['iter']), len(train_data['acc_seg']))
        plt.figure(figsize=(12, 6))
        plt.plot(train_data['iter'][:min_len], train_data['acc_seg'][:min_len], label='Train Pixel Acc', color='#1f77b4', alpha=0.8)
        plt.title('Training Pixel Accuracy', fontsize=14)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # X轴设置：防止重叠
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
        
        # === Y轴修改：扩大范围，减少视觉波动 ===
        # Accuracy 也可以用同样的逻辑，或者你可以手动设置 plt.ylim(0, 100)
        set_y_axis_scale(train_data['acc_seg'], padding_factor=2.0)
        
        save_path = os.path.join(output_dir, 'plot_train_acc.png')
        plt.savefig(save_path)
        print(f"生成图片: {save_path}")
        plt.close()

    # ==========================================
    # 3. 绘制 Validation Metrics
    # ==========================================
    if val_data['iter'] and val_data['mIoU']:
        plt.figure(figsize=(12, 6))
        
        plt.plot(val_data['iter'], val_data['mIoU'], label='Val mIoU', color='#2ca02c', marker='o', linewidth=2)
        if val_data['aAcc']:
            plt.plot(val_data['iter'], val_data['aAcc'], label='Val aAcc', color='#ff7f0e', marker='s', linestyle='--', linewidth=2)

        plt.title('Validation Metrics (mIoU & aAcc)', fontsize=14)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Score (0-1)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # X轴设置：防止重叠，并旋转45度
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
        plt.xticks(rotation=45)

        # === Y轴修改：验证集通常在 0-1 之间，或者根据数据自动调整 ===
        # 如果你想让验证集曲线也平滑一点，可以取消下面这行的注释：
        # set_y_axis_scale(val_data['mIoU'] + val_data['aAcc'], padding_factor=1.5)
        # 或者手动设置一个较宽的范围，例如 0 到 1
        plt.ylim(0, 1.0)  # 强制设置为 0-1，这样波动看起来会很真实且平滑
        
        save_path = os.path.join(output_dir, 'plot_val_metrics.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"生成图片: {save_path}")
        plt.close()
    else:
        print("提示: 未检测到验证集数据 (mIoU)。")

def main():
    train_data, val_data = parse_json_log(LOG_FILE_PATH)
    if not train_data: return
    output_dir = os.path.dirname(LOG_FILE_PATH)
    plot_metrics(train_data, val_data, output_dir)
    print("所有绘图任务完成。")

if __name__ == "__main__":
    main()'''
_base_ = [
    '../_base_/models/Mobile_Seed.py', 
    '../_base_/datasets/pascal_context_boundary.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_80k.py'
]

# ===========================================================
# 1. 加载预训练权重
# ===========================================================
# load_from = '/root/Mobile-Seed/ckpt/MS_tiny_pascal_context.pth'
load_from = '/root/Mobile-Seed/work_dirs/MS_tiny_pascal_tuning_loss/v2/latest.pth'

# ===========================================================
# 2. 模型设置
# ===========================================================
# 0号类为背景(权重1.0)，1-59号类为前景(权重5.0)
# class_weights = [1.0] + [5.0] * 59 
class_weights = [1.0] + [2.0] * 59  # 从 5.0 降为 2.0

model = dict(
    pretrained=None, 
    backbone=dict(
        type='AFFormer_for_MS_tiny'),
    decode_head=[
        dict(
            type="BoundaryHead",
            bound_channels = [16,16,32,32],
            bound_ratio = 2,
            in_channels = [16,64,216,216], 
            in_index = [1,3,5,6],
            channels= 16 + 16 + 32 + 32,
            num_classes=1,
            # 策略A：边缘Loss权重 10.0 ——> 从 10.0 降为 3.0
            loss_decode= dict(type='ML_BCELoss', use_sigmoid=True, loss_weight=3.0, loss_name = "loss_be")),
        dict(
            type="RefineHead",
            fuse_channel = 96,
            in_channels=[216],
            in_index=[-1],
            channels=256,
            num_classes=60, 
            # 策略B：类别加权
            loss_decode=dict(
                type='CrossEntropyLoss', 
                use_sigmoid=False, 
                loss_weight=1.0, 
                loss_name = "loss_ce",
                class_weight=class_weights 
            )),
    ],
)

# ===========================================================
# 3. 优化器与学习率 (已修复梯度裁剪配置)
# ===========================================================
# 修复点1：optimizer 中只保留 PyTorch 优化器本身的参数
optimizer = dict(
    _delete_=True, 
    type='AdamW', 
    # lr=0.0004, 
    # 原来是 0.0004，现在改为 0.00004
    lr=0.00004,     
    betas=(0.9, 0.999), 
    weight_decay=0.01
)

# 修复点2：grad_clip 必须放在 optimizer_config 中
optimizer_config = dict(
    _delete_=True,  # 覆盖 base 中的配置
    grad_clip=dict(max_norm=35, norm_type=2)
)

lr_config = dict(
    _delete_=True, 
    policy='poly',
    warmup='linear',
    # warmup_iters=200,  # 策略C：快速预热
    warmup_iters=100, # 只有1000 iters的时候改成100
    warmup_ratio=1e-6,
    power=1.0, 
    min_lr=0.0, 
    by_epoch=False
)

# ===========================================================
# 4. 运行环境
# ===========================================================
# 显存不足请改为 8 或 4
data = dict(samples_per_gpu=16, workers_per_gpu=4)

# 多任务网络必须开启此项，否则DDP模式会报错
find_unused_parameters = True

# ===========================================================
# 5. 训练流程控制
# ===========================================================
runner = dict(type='IterBasedRunner', max_iters=1000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=500, metric='mIoU', pre_eval=True)
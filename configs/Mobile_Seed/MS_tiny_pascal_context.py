_base_ = [
    '../_base_/models/Mobile_Seed.py', '../_base_/datasets/pascal_context_boundary.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
# ===========================================================
# 【修改点 1：加载预训练权重】
# ===========================================================
# load_from 用于微调。它会加载你下载的 .pth 文件的参数。
load_from = '/root/Mobile-Seed/ckpt/MS_tiny_pascal_context.pth'

# ===========================================================
# 【修改点 2：模型设置】
# ===========================================================
model = dict(
    # 将 pretrained 设为 None。
    # 因为我们已经用 load_from 加载了全量权重，不需要再单独加载 backbone 的 ImageNet 权重了。
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
            loss_decode= dict(type='ML_BCELoss', use_sigmoid=True, loss_weight=1.0,loss_name = "loss_be")),
        dict(
            type="RefineHead",
            fuse_channel = 96,
            in_channels=[216],
            in_index=[-1],
            channels=256,
            num_classes=60, 
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,loss_name = "loss_ce")),
                ],
    )

# ===========================================================
# 【修改点 3：微调的学习率设置】
# ===========================================================
# 微调时，学习率通常要比从头训练小一点，或者保持原样。
# 这里保持原样 0.0004，如果发现 loss 不降反升，可以尝试改成 0.0001
optimizer = dict(_delete_=True, type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.01)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0 , by_epoch=False)

# ===========================================================
# 【修改点 4：显存与Batch Size】
# ===========================================================
# 如果你的显卡显存较小（比如只有 6G 或 8G），samples_per_gpu=16 可能会报 OOM (Out of Memory)。
# 如果报错，请把下面的 16 改成 8 或 4。
data = dict(samples_per_gpu=16, workers_per_gpu=4)

find_unused_parameters = True

# ===========================================================
# 【修改点 5：控制训练时长与保存频率 (关键修改)】
# ===========================================================
# 这里的设置会覆盖 _base_ 里 schedule_80k.py 的设置。
# 80k (8万次) 太久了，我们先跑 2000 次验证一下能不能跑通。

# 1. 总共跑 2000 次迭代 (大约几十分钟到一小时，取决于显卡)
runner = dict(type='IterBasedRunner', max_iters=2000)

# 2. 设置保存模型和验证的频率
# 必须 <= max_iters。这里设置每 500 次保存一次权重，并验证一次精度。
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=50, metric='mIoU', pre_eval=True)



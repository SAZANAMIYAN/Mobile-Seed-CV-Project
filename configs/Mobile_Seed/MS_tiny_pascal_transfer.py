_base_ = [
    '../_base_/models/Mobile_Seed.py', '../_base_/datasets/pascal_context_boundary.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

# 【重要】：加载你那个效果尚可的 ckpt
load_from = '/root/Mobile-Seed/ckpt/MS_tiny_pascal_context.pth' 

model = dict(
    pretrained=None,
    backbone=dict(
        type='AFFormer_for_MS_tiny'),
    decode_head=[
        # -------------------------------------------------
        # Head 1: BoundaryHead (边界预测头)
        # -------------------------------------------------
        dict(
            type="BoundaryHead",
            bound_channels = [16,16,32,32],
            bound_ratio = 2,
            in_channels = [16,64,216,216], 
            in_index = [1,3,5,6],
            channels= 16 + 16 + 32 + 32,
            num_classes=1,
            loss_decode=[
                # -------------------------------------------------------
                # 策略 B：Class Balanced / 加权召回 -> 使用 Focal Loss
                # -------------------------------------------------------
                dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, 
                     loss_weight=1.0, loss_name="loss_focal"),
                
                # -------------------------------------------------------
                # 策略 A：引入 Dice Loss -> 解决断线
                # -------------------------------------------------------
                # Dice Loss 不在乎单个像素，只在乎整体形状的重合度。
                # 权重设为 3.0，强迫模型把断开的线连起来。
                dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0, 
                     loss_name="loss_dice")
            ]
        ),
        # -------------------------------------------------
        # Head 2: RefineHead (语义分割头)
        # -------------------------------------------------
        dict(
            type="RefineHead",
            fuse_channel = 96,
            in_channels=[216],
            in_index=[-1],
            channels=256,
            num_classes=60,
            # -------------------------------------------------------
            # 策略 C：加强语义分割辅助
            # -------------------------------------------------------
            # 将权重从 1.0 提升至 3.0。
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=3.0, 
                loss_name="loss_ce")
        ),
    ],
    # test_cfg 保持注释状态或按需开启
    # test_cfg = dict(mode='slide',crop_size=(512, 512), stride=(384, 384))
)

# -------------------------------------------------
# 【温和微调策略】：解决报错并保护权重
# -------------------------------------------------
optimizer = dict(
    _delete_=True, 
    type='AdamW', 
    lr=0.00005, 
    betas=(0.9, 0.999), 
    weight_decay=0.01
)

# Warmup 依然保留，防止刚开始 Loss 激增导致梯度爆炸
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0 , by_epoch=False)

data=dict(samples_per_gpu=16, workers_per_gpu=4)
find_unused_parameters=True

# -------------------------------------------------
# 运行设置
# -------------------------------------------------
runner = dict(type='IterBasedRunner', max_iters=2000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=500, metric='mIoU')
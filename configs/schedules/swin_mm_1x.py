from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop
from torch.optim.adamw import AdamW

# Default 1x schedule with AdamW as used by Swin (mmdetection)
#
# TLDR: AdamW, 1000 steps LR warmup, LR milestones [8, 11], epochs 12
#
# NOTE: Swin used amp training by default which is not enabled in this config, run train.py --amp if you want this.
#
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py
#         https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/tree/master/configs/swin
#         The original repository uses old train.py/runner/etc. so config is a little different from mmdetection.

# training schedule for 1x
schedule_name = 'swin_mm_1x'  # used for wandb logging and run_name
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=12, val_interval=1)
# See config/evaluators for val_cfg
# See config/evaluators for test_cfg

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optimizer_name='AdamW'
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=None  #  update dynamically in train.py depending on model, see below
    )
    # Example: swin transformer requires the following paramwise_cfg which put in the model config file
    # paramwise_cfg=dict(
    #     custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
    #                  'relative_position_bias_table': dict(decay_mult=0.),
    #                  'norm': dict(decay_mult=0.)})

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

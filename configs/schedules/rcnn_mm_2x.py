from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop
from torch.optim.sgd import SGD

# Default 2x schedule with SGD for rcnn like models in mmdetection
#
# TLDR: SGD, 500 steps LR warmup, LR milestones [16, 22], epochs 24
#
# Source: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/configs/_base_/schedules/schedule_2x.py

# training schedule for 2x
schedule_name = 'rcnn_mm_2x'  # used for wandb logging and run_name
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=24, val_interval=1)
# See config/evaluators for val_cfg
# See config/evaluators for test_cfg

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# optimizer
optimizer_name='SGD'
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

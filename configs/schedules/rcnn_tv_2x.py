from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop
from torch.optim.sgd import SGD

# Default schedule with SGD for rcnn like models as used by the torchvision reference implementation
#
# TLDR: SGD, 1000 steps LR warmup, LR milestones [16, 22], epochs 26
#
# NOTE: The schedule *follows* the Mask R-CNN paper schedule which trained for 160k iterations and
#       decreased the learning rate by factor 10 at 120k iterations. Assuming 7330 steps per epoch
#       (117266 images with annotations / batch_size 16) this translates to:
#
#       - 120000/7330=16,3710777626 -> 16
#       - 160000/7330=21,8281036835 -> 22
#
#       The reference implementation uses these as milestones and trains for a total of 26 epochs.
#       Since a 2x schedule in mmdet and d2 is 24 epochs, we classify this config as 2x. A linear
#       warmup schedule is applied for the first 1000 steps.
#
# Source: https://github.com/pytorch/vision/blob/main/references/detection/train.py#L264
#         https://github.com/pytorch/vision/blob/main/references/detection/train.py#L71
#         https://github.com/pytorch/vision/blob/main/references/detection/engine.py#L12

# training schedule for 2x
schedule_name = 'rcnn_tv_2x'  # used for wandb logging and run_name
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=26, val_interval=1)
# See config/evaluators for val_cfg
# See config/evaluators for test_cfg

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=26,
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

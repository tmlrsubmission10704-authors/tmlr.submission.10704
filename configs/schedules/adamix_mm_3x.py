from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop
from torch.optim.adamw import AdamW

# 3x schedule as used by AdaMixer
# NOTE: we found the model to be highly sensitive to these settings
#       using a different learning rate will diverge AdaMixer
#       using LR step 27 is too late as the model get's worse after 24
#
# Source: https://github.com/MCG-NJU/AdaMixer/blob/a50e33d766c68e0df9ac592913ed40a528914fab/configs/adamixer/adamixer_r50_1x_coco.py#L163
#         https://github.com/MCG-NJU/AdaMixer/blob/a50e33d766c68e0df9ac592913ed40a528914fab/configs/adamixer/adamixer_r50_300_query_crop_mstrain_480-800_3x_coco.py#L54

# training schedule for 3x
schedule_name = 'adamix_mm_3x'  # used for wandb logging and run_name
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=36, val_interval=1)
# See config/evaluators for val_cfg
# See config/evaluators for test_cfg

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 33],
        gamma=0.1)
]

# optimizer
optimizer_name='AdamW'
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=0.000025, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=None  #  update dynamically in train.py depending on model, see below
    )

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

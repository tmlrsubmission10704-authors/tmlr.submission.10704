from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop
from torch.optim.sgd import SGD

# Default 3x schedule with SGD for rcnn like models as used in detectron 2
#
# TLDR: SGD, 1000 steps LR warmup, LR milestones [29, 34], epochs 37
#
# NOTE: Detectron 2 uses steps, i.e. iters instead of epochs for it's LR milestones.
#
#       The default 3x schedule for RCNN models uses: STEPS: (210000, 250000) MAX_ITER: 270000
#       It assumes an coco epoch to have 7500 iterations (120000/16). This translates to:
#
#       - 210000/7500=28 -> 28
#       - 250000/7500=33,3333333333 -> 33
#       - 270000/7500=36 -> 36
#
#       In reality, coco train split has (118287 images and) 117266 images with annotations
#       which is (7393) 7330 steps per epoch and translates to:
#
#       - 210000/7330=28,6493860846 -> 29
#       - 250000/7330=34,1064120055 -> 34
#       - 270000/7330=36,8349249659 -> 37
#
# Source: https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
#         https://github.com/facebookresearch/detectron2/blob/main/configs/common/coco_schedule.py
#
#
#         The 'LinearLR' warmup in detectron 2 is calculated in relation to the training length using
#         a CompositeParamScheduler. It will use the warmup schedule for e.g. 1% of the training and
#         and then switch to the MultiStep schedule for the other 99%. The exact relation for the
#         default schedules is defined as:
#
#         - warmup_length=1000/total_steps_16bs (e.g. 1000/270000)
#
#         The warmup will hence end after 1000 steps, independent of total training length.
#
# Source: https://github.com/facebookresearch/detectron2/blob/main/configs/common/coco_schedule.py
#         https://github.com/facebookresearch/detectron2/blob/main/detectron2/solver/lr_scheduler.py#L22
#         https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/param_scheduler.py#L353 

# training schedule for 3x
schedule_name = 'rcnn_d2_3x'  # used for wandb logging and run_name
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=37, val_interval=1)
# See config/evaluators for val_cfg
# See config/evaluators for test_cfg

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=37,
        by_epoch=True,
        milestones=[29, 34],
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

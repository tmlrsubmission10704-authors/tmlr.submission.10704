from mmengine.config import read_base

from torchvision.ops.misc import FrozenBatchNorm2d

with read_base():
    from .mask_rcnn_r50_fpn import *

model_backbone_name = 'r50_fbn'  # short name used for wandb logging and run_name
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        norm_cfg=dict(type=FrozenBatchNorm2d, requires_grad=False)
    )
)

from mmengine.config import read_base

with read_base():
    from .mask_rcnn_r152_fpn import *

model_backbone_name = 'r152_sbn'  # short name used for wandb logging and run_name
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        norm_eval=False  # SyncBN!
    )
)

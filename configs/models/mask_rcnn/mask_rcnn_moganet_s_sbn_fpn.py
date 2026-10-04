from mmengine.config import read_base

with read_base():
    from .mask_rcnn_moganet_s_fpn import *

model_backbone_name = 'moganet_s_sbn'  # short name used for wandb logging and run_name
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        batch_norm_eval=False  # SyncBN!
    )
)

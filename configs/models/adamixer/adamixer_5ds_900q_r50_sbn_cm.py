from mmengine.config import read_base

with read_base():
    from .adamixer_900q_r50_cm import *

model_framework_name = 'adamixer_5ds_900q'
model_backbone_name = 'r50_sbn'  # short name used for wandb logging and run_name
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        out_indices=(1, 2, 3),
        norm_eval=False  # SyncBN!
    ),
    neck=dict(in_channels=[512, 1024, 2048], num_outs=5),
    roi_head=dict(featmap_strides=[8, 16, 32, 64, 128])
)

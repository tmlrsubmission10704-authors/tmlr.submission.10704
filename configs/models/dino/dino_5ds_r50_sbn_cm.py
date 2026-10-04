from mmengine.config import read_base

with read_base():
    from .dino_r50_cm import *

model_framework_name = 'dino_5ds'
model_backbone_name = 'r50_sbn'  # short name used for wandb logging and run_name
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        norm_eval=False  # SyncBN!
    ),
    neck=dict(in_channels=[512, 1024, 2048], num_outs=5),
    num_feature_levels=5,
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=5))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=5))),
)

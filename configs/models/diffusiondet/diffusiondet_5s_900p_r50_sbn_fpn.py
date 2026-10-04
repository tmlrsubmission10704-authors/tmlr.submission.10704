from mmengine.config import read_base

with read_base():
    from .diffusiondet_900p_r50_fpn import *

model_framework_name = 'diffusiondet_5s_900p'
model_backbone_name = 'r50_sbn'  # short name used for wandb logging and run_name
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(norm_eval=False),
    neck=dict(num_outs=5),
    bbox_head=dict(roi_extractor=dict(featmap_strides=[4, 8, 16, 32, 64]))
)

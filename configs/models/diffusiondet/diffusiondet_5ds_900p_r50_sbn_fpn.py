from mmengine.config import read_base

with read_base():
    from .diffusiondet_900p_r50_fpn import *

model_framework_name = 'diffusiondet_5ds_900p'
model_backbone_name = 'r50_sbn'  # short name used for wandb logging and run_name
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        out_indices=(1, 2, 3),
        norm_eval=False),
    neck=dict(in_channels=[512, 1024, 2048], num_outs=5, add_extra_convs='on_input'),
    bbox_head=dict(roi_extractor=dict(featmap_strides=[8, 16, 32, 64, 128]))
)

from mmengine.config import read_base

with read_base():
    from .mask_rcnn_r50_fpn import *

# https://github.com/open-mmlab/mmdetection/blob/main/configs/dcnv2/mask-rcnn_r50-mdconv-c3-c5_fpn_1x_coco.py
model_backbone_name = 'r50_dcnv2'  # short name used for wandb logging and run_name
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    )
)

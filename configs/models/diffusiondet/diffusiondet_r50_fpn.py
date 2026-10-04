from mmengine.config import read_base

from torch.nn import BatchNorm2d
from mmdet.models.backbones.resnet import ResNet
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._diffusiondet_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'r50'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.resnet50'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type=BatchNorm2d, requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(resolve='mmdet.resnet50.tv1_in1k')),
    neck=dict(in_channels=[256, 512, 1024, 2048])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
# https://github.com/open-mmlab/mmdetection/blob/main/projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py#L159
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

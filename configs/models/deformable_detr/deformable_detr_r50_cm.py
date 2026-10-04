from mmengine.config import read_base

from torch.nn import BatchNorm2d
from mmdet.models.backbones.resnet import ResNet
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._deformable_detr_cm import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'r50'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.resnet50'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config

model.update(
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type=BatchNorm2d, requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(resolve='mmdet.resnet50.tv1_in1k')),
    neck=dict(in_channels=[512, 1024, 2048])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
# https://github.com/open-mmlab/mmdetection/blob/main/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py#L125
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))
        ),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

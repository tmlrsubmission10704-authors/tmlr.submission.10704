from mmengine.config import read_base

from torch.nn import BatchNorm2d
from mmdet.models.backbones.resnet import ResNet
#from mmengine.model.weight_init import PretrainedInit
from mmdet.models.layers import FrozenBatchNorm2d

with read_base():
    from ._align_detr_cm import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'r50'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.resnet50'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config

# NOTE: The original AlignDETR config only freeze stem (0) and uses FrozenBatchNorm2d 
#       but we keep these consistent with the other models!
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
# https://github.com/open-mmlab/mmdetection/blob/main/projects/AlignDETR/configs/align_detr-4scale_r50_8xb2-12e_coco.py#L145
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(backbone=dict(lr_mult=0.1), norm_decay_mult=0.0)  # AlignDETR: No norm decay.
        ),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

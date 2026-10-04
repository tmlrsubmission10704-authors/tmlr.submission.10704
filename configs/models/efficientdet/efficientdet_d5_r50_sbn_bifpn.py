from mmengine.config import read_base
from torch.nn import BatchNorm2d
from mmdet.models.backbones.resnet import ResNet

with read_base():
    from ._efficientdet_d5 import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'r50_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.resnet50'
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type=BatchNorm2d, requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(resolve='mmdet.resnet50.tv2_in1k'),
    ),
    neck=dict(in_channels=[512, 1024, 2048])
)

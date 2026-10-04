from mmengine.config import read_base

from torch.nn import BatchNorm2d
from mmdet.models.backbones.resnet import ResNet
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

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
        #init_cfg=dict(type=PretrainedInit, checkpoint="torchvision://resnet50.imagenet1k_v2")),
        #init_cfg=dict(type=PretrainedInit, checkpoint="https://download.pytorch.org/models/resnet50-11ad3fa6.pth")),
    neck=dict(in_channels=[256, 512, 1024, 2048])
)

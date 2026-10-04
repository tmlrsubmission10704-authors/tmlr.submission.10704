from mmengine.config import read_base

from torch.nn import BatchNorm2d
from mmdet.models.backbones.resnet import ResNet
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_scoring_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'r50'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.resnet50'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    # use caffe img_norm only when evaluating exiting checkpoints from mmdet
    #data_preprocessor=dict(
    #    mean=[103.530, 116.280, 123.675],
    #    std=[1.0, 1.0, 1.0],
    #    bgr_to_rgb=False),
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type=BatchNorm2d, requires_grad=True),
        norm_eval=True,
        style='pytorch',
        #style='caffe',
        init_cfg=dict(resolve='mmdet.resnet50.tv1_in1k')),
    neck=dict(in_channels=[256, 512, 1024, 2048])
)

from mmengine.config import read_base

from mmdet_one_piece.models.detectors.deta.backbone import MM_backbone_wrapper_DETA
from torch.nn import BatchNorm2d
from mmdet.models.backbones.resnet import ResNet
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._deta_cm import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'r50'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.resnet50'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config

model.update(
    backbone=dict(
        type=MM_backbone_wrapper_DETA,
        frozen_stages=1,
        init_cfg=dict(resolve='mmdet.resnet50.tv1_in1k'),
        backbone=dict(
            type=ResNet,
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            #frozen_stages=,  # will be set by MM_backbone_wrapper_DETA
            norm_cfg=dict(type=BatchNorm2d, requires_grad=True),
            norm_eval=True,
            style='pytorch',
            #init_cfg=  # will be set by MM_backbone_wrapper_DETA
            ),
        strides = [8, 16, 32],
        num_channels = [512, 1024, 2048])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        bypass_duplicate=True, # they clone model.class_embed & model.bbox_embed ...
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))
        ),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

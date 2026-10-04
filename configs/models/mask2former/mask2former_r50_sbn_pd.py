from mmengine.config import read_base

from torch.nn import BatchNorm2d
from mmdet.models.backbones.resnet import ResNet

with read_base():
    from ._mask2former_pd import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'r50_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.resnet50'
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type=BatchNorm2d, requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(resolve='mmdet.resnet50.tv2_in1k')),
)


# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
# https://github.com/open-mmlab/mmdetection/blob/main/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py#L195C1-L213C48
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            backbone=dict(lr_mult=0.1, decay_mult=1.0),
            query_embed=dict(lr_mult=1.0, decay_mult=0.0),
            query_feat=dict(lr_mult=1.0, decay_mult=0.0),
            level_embed=dict(lr_mult=1.0, decay_mult=0.0)),
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2)
)

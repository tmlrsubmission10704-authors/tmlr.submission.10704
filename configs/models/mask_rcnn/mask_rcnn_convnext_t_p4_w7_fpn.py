from mmengine.config import read_base

from mmpretrain.models import ConvNeXt
#from mmengine.model.weight_init import PretrainedInit
from mmdet.engine.optimizers import LearningRateDecayOptimizerConstructor

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'convnext_t_p4_w7'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmpre.convnext_t_p4_w7'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/open-mmlab/mmdetection/blob/main/configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py
# https://github.com/facebookresearch/ConvNeXt/blob/main/object_detection/configs/convnext/mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k.py
# https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/backbones/convnext.py#L130
model.update(
    backbone=dict(
        type=ConvNeXt,
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        frozen_stages=0,
        gap_before_final_norm=False,
        init_cfg=dict(resolve='mmpre.convnext_t_p4_w7.mm_in1k', prefix='backbone.')),
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone.')),
    neck=dict(in_channels=[96, 192, 384, 768])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    constructor=LearningRateDecayOptimizerConstructor,
    paramwise_cfg=dict(
        _delete_=True,
        decay_rate=0.95,
        decay_type='layer_wise',
        num_layers=6)
)

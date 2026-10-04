from mmengine.config import read_base

from torch import nn
from mmdet_one_piece.models.backbones.mpvit import mpvit_small
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'mpvit_s_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.mpvit_s'
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/youngwanLEE/MPViT/blob/main/detectron2/configs/Base-RCNN-FPN.yaml
# https://github.com/youngwanLEE/MPViT/blob/main/detectron2/configs/maskrcnn/mask_rcnn_mpvit_small_ms_3x.yaml
# https://github.com/youngwanLEE/MPViT/blob/main/detectron2/mpvit/config.py
# https://github.com/youngwanLEE/MPViT/blob/main/detectron2/mpvit/backbone.py
# https://github.com/youngwanLEE/MPViT/blob/main/detectron2/mpvit/mpvit.py#L748
#
# NOTE: they use clip gradients see: Base-RCNN-FPN.yaml
#
model.update(
    backbone=dict(
        type=mpvit_small,
        norm='BN',  # d2 reference, they actually use SyncBN but we convert BN globally
        drop_path_rate=0.2,
        out_features=["stage2", "stage3", "stage4", "stage5"],
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.mpvit_s.ek_in1k')),
    neck=dict(in_channels=[128, 216, 288, 288])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
#
# NOTE: they use convolutional (relative) positional encoding. Should this be excluded from weight_decay as well?
# NOTE: some layers are referenced more than once which will put them in the same parameter group when using default
#       paramwise_cfg.
#optim_wrapper = dict(
    #paramwise_cfg=dict(
    #    _delete_=True,
    #    custom_keys=dict(
    #        norm=dict(decay_mult=0.0))
    #)
#)

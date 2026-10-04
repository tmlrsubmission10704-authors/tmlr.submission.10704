from mmengine.config import read_base

from torch import nn
from mmdet_one_piece.models.backbones.mvit import MViT
from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'mvit_v2_l'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'd2.mvit_v2_l'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/detectron2/modeling/backbone/mvit.py#L271
# https://github.com/facebookresearch/detectron2/blob/main/projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_l_in21k_lsj_50ep.py
model.update(
    backbone=dict(
        type=MViT,
        embed_dim=144,
        depth=48,
        num_heads=2,
        last_block_indexes=(1, 7, 43, 47),
        residual_pooling=True,
        drop_path_rate=0.5,
        norm_layer=dict(type=nn.LayerNorm, eps=1e-6),
        out_features=("scale2", "scale3", "scale4", "scale5"),
        frozen_stages=-1,
        init_cfg=dict(resolve='d2.mvit_v2_l.fb_in21k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint='https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/mvitv2/MViTv2_T_in1k.pyth')),
        #init_cfg=None),
    neck=dict(in_channels=[144, 288, 576, 1152]),
    roi_head=dict(
        bbox_roi_extractor=dict(roi_layer=dict(use_torchvision=True)),
        mask_roi_extractor=dict(roi_layer=dict(use_torchvision=True))
    )
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
                pos_embed=dict(decay_mult=0.0),
                rel_pos_h=dict(decay_mult=0.0),
                rel_pos_w=dict(decay_mult=0.0),
                norm=dict(decay_mult=0.0))
    )
)

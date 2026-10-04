from mmengine.config import read_base

from torch import nn
from mmdet_one_piece.models.backbones.focalnet import FocalNet
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'focalnet_t_p4_lrf'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.focalnet_t_p4_lrf'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/microsoft/FocalNet/blob/main/detection/configs/focalnet/mask_rcnn_focalnet_tiny_patch4_mstrain_480-800_adamw_3x_coco_lrf.py
# https://github.com/microsoft/FocalNet/blob/main/detection/mmdet/models/backbones/focalnet.py#L317
model.update(
    backbone=dict(
        type=FocalNet,
        # from cfg file
        embed_dim=96,
        depths=[2, 2, 6, 2],
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        focal_windows=[9,9,9,9],
        focal_levels=[3,3,3,3],
        use_conv_embed=False,
        # default args
        patch_size=4,
        mlp_ratio=4.,
        drop_rate=0.,
        norm_layer=nn.LayerNorm,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_layerscale=False,
        # new args
        init_cfg=dict(resolve='mmdet.focalnet_t_p4_lrf.ms_in1k')),
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(in_channels=[96, 192, 384, 768])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.),
            relative_position_bias_table=dict(decay_mult=0.),
            norm=dict(decay_mult=0.))
    )
)

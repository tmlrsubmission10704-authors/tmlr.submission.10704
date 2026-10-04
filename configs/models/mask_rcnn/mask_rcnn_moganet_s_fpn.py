from mmengine.config import read_base

from torch import nn
from mmdet_one_piece.models.backbones.moganet import MogaNet
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'moganet_s'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.moganet_s'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/Westlake-AI/MogaNet/blob/main/detection/configs/moganet/mask_rcnn_moganet_s_fpn_mstrain_480-800_3x_coco.py
# https://github.com/Westlake-AI/MogaNet/blob/main/models/moganet.py#L484
model.update(
    backbone=dict(
        type=MogaNet,
        fork_feat=True,
        arch='small',
        drop_path_rate=0.1,
        # from here, default args of constructor. We put them here so they get logged.
        drop_rate=0.,
        init_value=1e-5,
        head_init_scale=1.,
        patch_sizes=[3, 3, 3, 3],
        stem_norm_type='BN',
        conv_norm_type='BN',
        patchembed_types=['ConvEmbed', 'Conv', 'Conv', 'Conv',],
        attn_dw_dilation=[1, 2, 3],
        attn_channel_split=[1, 3, 4],
        attn_act_type='SiLU',
        attn_final_dilation=True,
        attn_force_fp32=False,
        frozen_stages=-1,
        batch_norm_eval=False,  # we tried semi-frozen BN similar to ResNet Default in mmdet but it diverges...
        batch_norm_grad=True,
        init_cfg=dict(resolve='mmdet.moganet_s.wl_in1k')),
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(in_channels=[64, 128, 320, 512])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            layer_scale=dict(decay_mult=0.),
            scale=dict(decay_mult=0.),
            norm=dict(decay_mult=0.))
    )
)

from mmengine.config import read_base

from mmdet_one_piece.models.backbones.pvt import PyramidVisionTransformerV2
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'pvt_v2_b2'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.pvt_v2_b2'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/open-mmlab/mmdetection/blob/main/configs/pvt/retinanet_pvtv2-b2_fpn_1x_coco.py
# https://github.com/whai362/PVT/blob/57e2dfaa5a46f9050d76f306a4fcd9a7c061f520/detection/configs/mask_rcnn_pvt_v2_b2_fpn_3x_mstrain.py
# https://github.com/whai362/PVT/blob/57e2dfaa5a46f9050d76f306a4fcd9a7c061f520/detection/pvt_v2.py#L360
model.update(
    backbone=dict(
        type=PyramidVisionTransformerV2,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        #patch_sizes=[7, 3, 3, 3], -> hardcoded in mmdet pvt2 implementations
        strides=[4, 2, 2, 2],
        #paddings=[3, 1, 1, 1],    -> hardcoded in mmdet pvt2 implementations
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        #use_abs_pos_embed=False,  -> hardcoded in mmdet pvt2 implementations
        #norm_after_stage=True,    -> hardcoded in mmdet pvt2 implementations
        #use_conv_ffn=True,        -> hardcoded in mmdet pvt2 implementations
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', eps=1e-6),
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.pvt_v2_b2.pvt_in1k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint="...."))
    neck=dict(in_channels=[64, 128, 320, 512])
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

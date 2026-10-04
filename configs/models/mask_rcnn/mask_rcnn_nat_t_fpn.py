from mmengine.config import read_base

from mmdet_one_piece.models.backbones.nat import NAT
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'nat_t'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.nat_t'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/detection/configs/nat/mask_rcnn_nat_tiny_3x_coco.py
# https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/detection/configs/_base_/models/mask_rcnn_nat.py
# https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/detection/nat.py#L218
model.update(
    backbone=dict(
        type=NAT,
        embed_dim=64,
        mlp_ratio=3.0,
        depths=[3, 4, 18, 5],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.2,
        kernel_size=7,
        # ARGS from _base_ cfg
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        in_patch_size=4,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.nat_t.sl_in1k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint="...")),
    neck=dict(in_channels=[64, 128, 256, 512])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        norm_decay_mult=0.,
        custom_keys=dict(
            rpb=dict(decay_mult=0.),
            norm=dict(decay_mult=0.)
            )
    )
)

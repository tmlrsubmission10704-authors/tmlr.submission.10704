from mmengine.config import read_base

from mmdet_one_piece.models.backbones.fan import fan_small_12_p4_hybrid
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'fan_small_12_p4_hybrid_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.fan_small_12_p4_hybrid'
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/NVlabs/FAN/blob/master/detection/configs/cascade_mask_fan_small_fpn_3x_mstrain_fp16.py

model.update(
    backbone=dict(
        type=fan_small_12_p4_hybrid,
        use_fused_sdpa=True,
        use_checkpoint=False,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.fan_small_12_p4_hybrid.nv_in1k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint="...."))
    neck=dict(in_channels=[128, 256, 384, 768]),
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
            absolute_pos_embed=dict(decay_mult=0.),
            relative_position_bias_table=dict(decay_mult=0.),
            norm=dict(decay_mult=0.))
    )
)

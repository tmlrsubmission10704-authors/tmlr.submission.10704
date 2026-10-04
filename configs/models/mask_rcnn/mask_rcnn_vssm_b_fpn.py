from mmengine.config import read_base

from mmdet_one_piece.models.backbones.vmamba import vmamba

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'vssm_b'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.vssm_b'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/MzeroMiko/VMamba/blob/main/detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_base.py
# https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py#L1574
# https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py#L1243
model.update(
    backbone=dict(
        type=vmamba,
        out_indices=(0, 1, 2, 3),
        dims=128,
        depths=(2, 2, 15, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz", # v3_noz
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.6,
        norm_layer="ln2d",
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.vssm_b.mm_in1k')),
    neck=dict(in_channels=[128, 256, 512, 1024])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            norm=dict(decay_mult=0.))
    )
)

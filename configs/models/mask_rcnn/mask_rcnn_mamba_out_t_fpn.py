from mmengine.config import read_base

from mmdet_one_piece.models.backbones.timm_backbone import TIMMBackbone
# import not required, but useful for quick access via IDEs
from timm.models.mambaout import mambaout_tiny

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'mamba_out_t'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'timm.mamba_out_t'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# 
# 
model.update(
    backbone=dict(
        type=TIMMBackbone,
        model_name='mambaout_tiny',
        drop_path_rate=0.1,
        frozen_stages=-1,
        freeze_stages_fn = None,
        feature_out_norm_cfg=dict(type='LN2d', eps=1e-6),
        out_indices=(0, 1, 2, 3),
        nhwc_to_nchw=True,
        init_cfg=dict(resolve='timm.mamba_out_t.yu_in1k')),
    neck=dict(in_channels=[96, 192, 384, 576])
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            norm=dict(decay_mult=0.))
    )
)

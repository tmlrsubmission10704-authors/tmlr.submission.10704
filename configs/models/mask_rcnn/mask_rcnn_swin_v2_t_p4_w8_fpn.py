from mmengine.config import read_base

from mmdet_one_piece.models.backbones.timm_backbone import TIMMBackbone
# import not required, but useful for quick access via IDEs
from timm.models.swin_transformer_v2 import swinv2_tiny_window8_256

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'swin_v2_t_p4_w8'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'timm.swin_v2_t_p4_w8'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# See Appendix A1.2. https://arxiv.org/pdf/2111.09883
# NOTE: does not work with amp!
model.update(
    backbone=dict(
        type=TIMMBackbone,
        model_name='swinv2_tiny_window8_256',
        drop_path_rate=0.1,  # not reported in paper but we use 0.1 by default
        frozen_stages=-1,
        freeze_stages_fn = None,
        feature_out_norm_cfg=dict(type='LN2d', eps=1e-6),
        out_indices=(0, 1, 2, 3),
        strict_img_size=False,
        nhwc_to_nchw=True,
        init_cfg=dict(resolve='timm.swin_v2_t_p4_w8.ms_in1k')),
    neck=dict(in_channels=[96, 192, 384, 768])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            cpb=dict(decay_mult=0.),
            norm=dict(decay_mult=0.))
    )
)

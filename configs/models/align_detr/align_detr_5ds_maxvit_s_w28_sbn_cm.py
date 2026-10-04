from mmengine.config import read_base

from mmdet_one_piece.models.backbones.timm_backbone import TIMMBackbone
# import not required, but useful for quick access via IDEs
from timm.models.maxxvit import maxvit_small_tf_224

with read_base():
    from ._align_detr_cm import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_framework_name = 'align_detr_5ds'
model_backbone_name = 'maxvit_s_w28_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'timm.maxvit_s_tf_224'
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config


model.update(
    data_preprocessor=dict(pad_size_divisor=896),  # https://github.com/google-research/maxvit/issues/10
        backbone=dict(
            type=TIMMBackbone,
            model_name='maxvit_small_tf_224',
            img_size=896,
            drop_path_rate=0.2,     # we use smaller dpr for smaller batch sizes! e.g. bs16
            batch_norm_eval=False,
            batch_norm_grad=True,
            frozen_stages=-1,
            freeze_stages_fn = None,
            feature_out_norm_cfg=dict(type='LN2d', eps=1e-6),
            out_indices=(2, 3, 4),
            use_fused_sdpa=True,
            init_cfg=dict(resolve='timm.maxvit_s_tf_224.tf_in1k')),
    neck=dict(in_channels=[192, 384, 768], num_outs=5),
    num_feature_levels=5,
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=5))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=5))),
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(backbone=dict(lr_mult=0.1), norm_decay_mult=0.0)  # AlignDETR: No norm decay.
        ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=2
)

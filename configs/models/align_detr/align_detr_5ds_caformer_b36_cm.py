from mmengine.config import read_base

#from mmengine.model.weight_init import PretrainedInit
from mmdet_one_piece.models.backbones.metaformer_baselines import caformer_b36

with read_base():
    from ._align_detr_cm import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_framework_name = 'align_detr_5ds'
model_backbone_name = 'caformer_b36'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.caformer_b36'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config

model.update(
    backbone=dict(
        type=caformer_b36,
        drop_path_rate=0.3,
        frozen_stages=-1,
        use_fused_sdpa=True,
        out_indices=(1, 2, 3),
        init_cfg=dict(resolve='mmdet.caformer_b36.sa_in1k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint="...."))
    neck=dict(in_channels=[256, 512, 768],  num_outs=5),
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
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

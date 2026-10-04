from mmengine.config import read_base

from mmdet_one_piece.models.detectors.deta.backbone import MM_backbone_wrapper_DETA
from mmdet_one_piece.models.backbones.metaformer_baselines import caformer_b36
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._deta_cm import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'caformer_b36'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.caformer_b36'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config

model.update(
    backbone=dict(
        type=MM_backbone_wrapper_DETA,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.caformer_b36.sa_in1k'),
        backbone=dict(
            type=caformer_b36,
            drop_path_rate=0.3,
            frozen_stages=-1,
            use_fused_sdpa=True,
            out_indices=(1, 2, 3),
            #frozen_stages=,  # will be set by MM_backbone_wrapper_DETA
            #init_cfg=  # will be set by MM_backbone_wrapper_DETA
        ),
        strides = [8, 16, 32],
        num_channels = [256, 512, 768])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        bypass_duplicate=True, # they clone model.class_embed & model.bbox_embed ...
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))
        ),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

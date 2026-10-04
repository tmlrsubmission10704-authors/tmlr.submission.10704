from mmengine.config import read_base

from mmdet_one_piece.models.detectors.deta.backbone import MM_backbone_wrapper_DETA
from mmdet_one_piece.models.backbones.transnext import TransNeXt
import torch.nn as nn
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._deta_cm import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'transnext_b'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.transnext_b'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config

model.update(
    backbone=dict(
        type=MM_backbone_wrapper_DETA,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.transnext_b.tx_in1k'),
        backbone=dict(
            type= TransNeXt,
            #type='transnext_base',  # we use put all args here instead of calling transnext_base()
            window_size=[3, 3, 3, None],
            patch_size=4,
            embed_dims=[96, 192, 384, 768],
            num_heads=[4, 8, 16, 32],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=dict(type=nn.LayerNorm, eps=1e-6),  # instead of partial(nn.LayerNorm, eps=1e-6)
            depths=[5, 5, 23, 5],
            sr_ratios=[8, 4, 2, 1],  # original
            #sr_ratios=[16, 8, 4, 1],  # way faster with small performance hit
            drop_rate=0.0,
            drop_path_rate=0.7,
            img_size=800,
            pretrain_size=224,
            is_extrapolation=True,  # faster
            out_indices=(1, 2, 3),
            #frozen_stages=,  # will be set by MM_backbone_wrapper_DETA
            #init_cfg=  # will be set by MM_backbone_wrapper_DETA
        ),
        strides = [8, 16, 32],
        num_channels = [192, 384, 768])
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

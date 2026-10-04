from mmengine.config import read_base

import torch.nn as nn
from mmdet_one_piece.models.backbones.cswin_transformer import CSWin
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._ddq_detr_5s_cm import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_framework_name = 'ddq_detr_5ds'
model_backbone_name = 'cswin_b'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.cswin_b'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config


model.update(
    backbone=dict(
            type=CSWin,
            embed_dim=96,
            patch_size=4,
            depth=[2,4,32,2],
            num_heads=[2,4,8,16],
            split_size=[1,2,7,7],
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.6,
            hybrid_backbone=None,
            norm_layer=nn.LayerNorm,
            use_chk=False,
            frozen_stages=-1,
            out_indices=(1, 2, 3),
            init_cfg=dict(resolve='mmdet.cswin_b.ms_in1k')),
    neck=dict(in_channels=[192, 384, 768], num_outs=5)
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(backbone=dict(lr_mult=0.1))
        ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=2
)

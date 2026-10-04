from mmengine.config import read_base

import torch.nn as nn
from mmdet_one_piece.models.backbones.biformer import BiFormer
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._diffusiondet_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'biformer_s_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.biformer_s'
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        type=BiFormer,
        num_classes=80,
        depth=[4, 4, 18, 4],
        embed_dim=[64, 128, 256, 512],
        mlp_ratios=[3, 3, 3, 3],
        n_win=16,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False,
        diff_routing=False,
        soft_routing=False,
        pre_norm=True,
        pe=None,
        auto_pad=True,
        #use_checkpoint_stages=[],  # they use a third party lib which we don't support!
        drop_path_rate=0.2,
        norm_eval=False,            # SyncBN!
        disable_bn_grad=False,
        use_fused_sdpa=True,
        frozen_stages=-1,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(resolve='mmdet.biformer_s.bi_in1k')),
    neck=dict(in_channels=[64, 128, 256, 512])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
# https://github.com/open-mmlab/mmdetection/blob/main/projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py#L159
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=2
)

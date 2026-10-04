from mmengine.config import read_base

from torch import nn
from mmdet_one_piece.models.backbones.vitdet import ViTDet
from mmdet_one_piece.models.necks.feature_pyramids import FeatureMapResamplingWrapper, _SimpleFeaturePyramid
from mmdet_one_piece.engine.optimizer import LearningRateDecayOptimizerConstructor_ViTDet

with read_base():
    from ._mask_rcnn_v2_sbn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_neck_name = 'sfp'  # SimpleFeaturePyramid
model_backbone_name = 'vitdet_b'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'd2.vitdet_b'
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# 
# https://github.com/facebookresearch/detectron2/blob/main/configs/common/models/mask_rcnn_vitdet.py
# https://github.com/facebookresearch/detectron2/blob/main/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py#L232
#
# VitDet did not train a small model
# NOTE: we find that layer-wise lr decay is not helpful for last_map sfp model with a 3x schedule
#       (even if it looks promising at first)
#
model.update(
    #data_preprocessor=dict(pad_size_divisor=1024),
    backbone=dict(
        type=ViTDet,
        # vit/deit small cfg
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        # vitdet cfg args
        window_size=14,
        qkv_bias=True,
        norm_layer=dict(type=nn.LayerNorm, eps=1e-6),
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],  # 2, 5, 8 11 are global attention
        residual_block_indexes=[],
        use_rel_pos=True,
        # vitdet constructor args
        img_size=1024,
        use_abs_pos=True,
        rel_pos_zero_init=True,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        # our cfg
        drop_path_rate=0.1,
        out_indices=(11,),   # only 'last_map' for SimpleFeaturePyramid
        frozen_stages=-1,
        init_cfg=dict(resolve='d2.vitdet_b.fb_mae'))  # NOTE: d2 weights!
)
# hard overwrite of any previous neck cfg
model.neck=dict(
        type=FeatureMapResamplingWrapper,
        backbone_channels = 768,             # 'last_map' resampling
        scale_factors=[4.0, 2.0, 1.0, 0.5],  # strides [ 16,  16,  16,  16] -> [    4,     8,  16,  32]
        scale_dims=True,                     # dims    [768, 768, 768, 768] -> [768/4, 768/2, 768, 768]
        out_norm_cfg=dict(type='LN2d', eps=1e-6),  # NEW, was not used in original paper
        neck=dict(
            type=_SimpleFeaturePyramid,
            in_channels=[192, 384, 768, 768],
            out_channels=256,
            num_outs=5,
        )
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    constructor=LearningRateDecayOptimizerConstructor_ViTDet,
    paramwise_cfg=dict(
        _delete_=True,
        num_layers=12,
        decay_rate=0.7,
        decay_type='layer_wise',
        no_wd_list=['pos_embed', 'norm', 'rel_pos_h', 'rel_pos_w'],
        #
        # NOTE: with layer_wise, decay_mult=0. will be handled by the constructor using no_wd_list instead of
        #       custom_keys. Without custom LearningRateDecayOptimizerConstructor, you can use the following cfg to
        #       do the same:
        #
        # custom_keys=dict(
        #     pos_embed=dict(decay_mult=0.),
        #     rel_pos_h=dict(decay_mult=0.0),
        #     rel_pos_w=dict(decay_mult=0.0),
        #     norm=dict(decay_mult=0.))
    )
)

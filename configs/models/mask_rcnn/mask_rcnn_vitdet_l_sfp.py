from mmengine.config import read_base

from torch import nn
from mmdet_one_piece.models.backbones.vitdet import ViTDet
from mmdet_one_piece.models.necks.feature_pyramids import FeatureMapResamplingWrapper, _SimpleFeaturePyramid
from mmdet_one_piece.engine.optimizer import LearningRateDecayOptimizerConstructor_ViTDet

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_neck_name = 'sfp'  # SimpleFeaturePyramid
model_backbone_name = 'vitdet_l'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'd2.vitdet_l'
sync_bn = False
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# 
# https://github.com/facebookresearch/detectron2/blob/main/configs/common/models/mask_rcnn_vitdet.py
# https://github.com/facebookresearch/detectron2/blob/main/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_l_100ep.py
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py#L232
#
model.update(
    #data_preprocessor=dict(pad_size_divisor=1024),
    backbone=dict(
        type=ViTDet,
        # vit/deit small cfg
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        # vitdet cfg args
        window_size=14,
        qkv_bias=True,
        norm_layer=dict(type=nn.LayerNorm, eps=1e-6),
        window_block_indexes=[0,1,2,3,4, 6,7,8,9,10, 12,13,14,15,16, 18,19,20,21,22],  # 5, 11, 17, 23 are global attention
        residual_block_indexes=[],
        use_rel_pos=True,
        # vitdet constructor args
        img_size=1024,
        use_abs_pos=True,
        rel_pos_zero_init=True,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        # our cfg
        drop_path_rate=0.4,
        out_indices=(23,),   # only 'last_map' for SimpleFeaturePyramid
        frozen_stages=-1,
        init_cfg=dict(resolve='d2.vitdet_l.fb_mae'))  # NOTE: d2 weights!
)
# hard overwrite of any previous neck cfg
model.neck=dict(
        type=FeatureMapResamplingWrapper,
        backbone_channels = 1024,             # 'last_map' resampling
        scale_factors=[4.0, 2.0, 1.0, 0.5],  # strides [ 16,  16,  16,  16] -> [    4,     8,  16,  32]
        scale_dims=True,                     # dims    [1024, 1024, 1024, 1024] -> [1024/4, 1024/2, 1024, 1024]
        out_norm_cfg=dict(type='LN2d', eps=1e-6),  # NEW, was not used in original paper
        neck=dict(
            type=_SimpleFeaturePyramid,
            in_channels=[256, 512, 1024, 1024],
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
        num_layers=24,
        decay_rate=0.8,
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

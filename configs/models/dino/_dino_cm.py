# NOTE: Copy of mmdet/configs/dino/dino-4scale_r50_8xb2-12e_coco.py but without 'backbone' and 'neck.in_channels'
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/dino/dino-4scale_r50_8xb2-12e_coco.py
#
# NOTE: we use the better Hyperparameter Settings from:
# https://github.com/open-mmlab/mmdetection/blob/main/configs/dino/dino-4scale_r50_improved_8xb2-12e_coco.py
#
# NOTE: We set max_per_img=100 which will reduce AP by ~0.23 (49.87 instead of 50.1 when testing the better ckpt)

from mmdet_one_piece.models.detectors import DINO_OP
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.necks import ChannelMapper
from mmdet.models.dense_heads import DINOHead
from mmdet.models.losses import GIoULoss, L1Loss, FocalLoss
from mmdet.models.task_modules.assigners import HungarianAssigner, FocalLossCost, BBoxL1Cost, IoUCost

# used for wandb logging and run_name
model_framework_name = 'dino'
model_backbone_name = None  # set in specific config
model_neck_name = 'cm'

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True)
test_pipeline_LA=dict(with_bbox=True)

model = dict(
    type=DINO_OP,
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=None,         #  NOTE: add in specific config
    neck=dict(
        type=ChannelMapper,
        in_channels=None,  #  NOTE: add in specific config, depends on backbone!
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=-0.5,  # -0.5 for DeformDETR  # NOTE: improved HP (0.0 originally)
        temperature=10000),  # 10000 for DeformDETR  # NOTE: improved HP (20 originally)
    bbox_head=dict(
        type=DINOHead,
        num_classes=80,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),  # 2.0 in DeformDETR  # NOTE: improved HP (1.0 originally)
        loss_bbox=dict(type=L1Loss, loss_weight=5.0),
        loss_iou=dict(type=GIoULoss, loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=300)),  # TODO: half num_dn_queries  # NOTE: improved HP (100 originally)
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=FocalLossCost, weight=2.0),
                dict(type=BBoxL1Cost, weight=5.0, box_format='xywh'),
                dict(type=IoUCost, iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))  # 100 for DeformDETR NOTE: we use 100 as any other model does..

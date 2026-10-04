# NOTE: Copy of projects but without 'backbone' and 'neck.in_channels'
# Source: https://github.com/open-mmlab/mmdetection/blob/main/projects/AlignDETR/configs/align_detr-4scale_r50_8xb2-12e_coco.py
#
# NOTE: We set max_per_img=100 which will reduce AP by ~0.23 (51.15 instead of 51.38 (51.4) when testing the better ckpt)

from mmdet_one_piece.models.detectors import DINO_OP
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.necks import ChannelMapper
from mmdet.models.losses import GIoULoss, L1Loss, CrossEntropyLoss
from mmdet.models.task_modules.assigners import FocalLossCost, BBoxL1Cost, IoUCost

from mmdet_one_piece.models.task_modules.assigners import MixedHungarianAssigner
from mmdet_one_piece.models.dense_heads import AlignDETRHead

# used for wandb logging and run_name
model_framework_name = 'align_detr'
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
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        # AlignDETR: Add conv bias.
        bias=True,
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
        # AlignDETR: Set offset and temperature the same as DeformDETR.
        offset=-0.5,  # -0.5 for DeformDETR
        temperature=10000),  # 10000 for DeformDETR
    bbox_head=dict(
        type=AlignDETRHead,
        # AlignDETR: First 6 elements of `all_layers_num_gt_repeat` are for
        #   decoder layers' outputs. The last element is for encoder layer.
        all_layers_num_gt_repeat=[2, 2, 2, 2, 2, 1, 2],
        alpha=0.25,
        gamma=2.0,
        tau=1.5,
        num_classes=80,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type=CrossEntropyLoss, use_sigmoid=True,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type=L1Loss, loss_weight=5.0),
        loss_iou=dict(type=GIoULoss, loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=MixedHungarianAssigner,
            match_costs=[
                dict(type=FocalLossCost, weight=2.0),
                dict(type=BBoxL1Cost, weight=5.0, box_format='xywh'),
                dict(type=IoUCost, iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))  # 100 for DeformDETR NOTE: we use 100 as any other model does..

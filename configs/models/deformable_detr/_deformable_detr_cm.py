# NOTE: Copy of mmdet/configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py but without 'backbone' and 'neck.in_channels'
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py

from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor

from mmdet.models.necks import ChannelMapper
from mmdet_one_piece.models.detectors import DeformableDETR_OP
from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.models.losses import FocalLoss, GIoULoss, L1Loss
from mmdet.models.task_modules.assigners import HungarianAssigner, FocalLossCost, BBoxL1Cost, IoUCost

# used for wandb logging and run_name
model_framework_name = 'deformable_detr'
model_backbone_name = None  # set in specific config
model_neck_name = 'cm'

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True)
test_pipeline_LA=dict(with_bbox=True)

model = dict(
    type=DeformableDETR_OP,
    num_queries=300,
    num_feature_levels=4,
    with_box_refine=False,   # Tricks (see  + models)
    as_two_stage=False,      # Tricks (see ++ models)
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
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1))),
    decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type=DeformableDETRHead,
        num_classes=80,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type=L1Loss, loss_weight=5.0),
        loss_iou=dict(type=GIoULoss, loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=FocalLossCost, weight=2.0),
                dict(type=BBoxL1Cost, weight=5.0, box_format='xywh'),
                dict(type=IoUCost, iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))

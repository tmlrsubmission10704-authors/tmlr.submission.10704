# NOTE: Copy of mmdet/configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py

from mmcv.ops import RoIAlign
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.detectors import SparseRCNN
from mmdet.models.roi_heads import DIIHead
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import \
    SingleRoIExtractor
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import \
    DeltaXYWHBBoxCoder
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.task_modules.assigners import HungarianAssigner, FocalLossCost, BBoxL1Cost, IoUCost
from mmdet.models.losses import GIoULoss, FocalLoss, L1Loss
from mmdet.models.necks.fpn import FPN
from mmdet.models.dense_heads import EmbeddingRPNHead
from mmdet.models.layers import DynamicConv

from mmdet_one_piece.models.roi_heads import SparseRoIHead_OP

# used for wandb logging and run_name
model_framework_name = 'sparse_rcnn'
model_backbone_name = None  # set in specific config
model_neck_name = 'fpn'

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True)
test_pipeline_LA=dict(with_bbox=True)

# model settings
num_stages = 6
num_proposals = 100
model = dict(
    type=SparseRCNN,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=None,         #  NOTE: add in specific config
    neck=dict(
        type=FPN,
        in_channels=None,  #  NOTE: add in specific config, depends on backbone!
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',  # NOTE: This is part of the original cfg but has no effect with 4in-4out
        num_outs=4),
    rpn_head=dict(
        type=EmbeddingRPNHead,
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type=SparseRoIHead_OP,
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type=SingleRoIExtractor,
            roi_layer=dict(type=RoIAlign, output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type=DIIHead,
                num_classes=80,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type=DynamicConv,
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type=L1Loss, loss_weight=5.0),
                loss_iou=dict(type=GIoULoss, loss_weight=2.0),
                loss_cls=dict(
                    type=FocalLoss,
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type=DeltaXYWHBBoxCoder,
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type=HungarianAssigner,
                    match_costs=[
                        dict(type=FocalLossCost, weight=2.0),
                        dict(type=BBoxL1Cost, weight=5.0, box_format='xyxy'),
                        dict(type=IoUCost, iou_mode='giou', weight=2.0)
                    ]),
                sampler=dict(type=PseudoSampler),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=100)))  # max_per_img is usually set to num_proposals but we enforce 100 to match other models!

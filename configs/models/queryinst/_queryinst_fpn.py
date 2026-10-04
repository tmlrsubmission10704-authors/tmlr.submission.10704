# NOTE: Copy of mmdet/configs/queryinst/* but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
#       we also change to pytorch style (RGB)
#
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/queryinst/queryinst_r50_fpn_1x_coco.py

from mmcv.ops import RoIAlign
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.detectors import QueryInst
from mmdet.models.roi_heads import DIIHead
from mmdet.models.roi_heads.mask_heads import DynamicMaskHead
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import \
    SingleRoIExtractor
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import \
    DeltaXYWHBBoxCoder
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.task_modules.assigners import HungarianAssigner, FocalLossCost, BBoxL1Cost, IoUCost
from mmdet.models.losses import GIoULoss, FocalLoss, L1Loss, DiceLoss
from mmdet.models.necks.fpn import FPN
from mmdet.models.dense_heads import EmbeddingRPNHead
from mmdet.models.layers import DynamicConv

from mmdet_one_piece.models.roi_heads import SparseRoIHead_OP

# used for wandb logging and run_name
model_framework_name = 'queryinst'
model_backbone_name = None  # set in specific config
model_neck_name = 'fpn'

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True, with_mask=True)
test_pipeline_LA=dict(with_bbox=True, with_mask=True)

# model settings
num_stages = 6
num_proposals = 300  # NOTE: we use this for intermediate but enforce 100 for testing
model = dict(
    type=QueryInst,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
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
        mask_roi_extractor=dict(
            type=SingleRoIExtractor,
            roi_layer=dict(type=RoIAlign, output_size=14, sampling_ratio=2),
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
        ],
        mask_head=[
            dict(
                type=DynamicMaskHead,
                dynamic_conv_cfg=dict(
                    type=DynamicConv,
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=14,
                    with_proj=False,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                num_convs=4,
                num_classes=80,
                roi_feat_size=14,
                in_channels=256,
                conv_kernel_size=3,
                conv_out_channels=256,
                class_agnostic=False,
                norm_cfg=dict(type='BN'),
                upsample_cfg=dict(type='deconv', scale_factor=2),
                loss_mask=dict(
                    type=DiceLoss,
                    loss_weight=8.0,
                    use_sigmoid=True,
                    activate=False,
                    eps=1e-5)) for _ in range(num_stages)
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
                pos_weight=1,
                mask_size=28,
            ) for _ in range(num_stages)
        ]),
    test_cfg=dict(
        rpn=None,
        rcnn=dict(max_per_img=num_proposals,  # because of DynamicMaskHead, this MUST be the same as num_proposals!
                  mask_thr_binary=0.5,
                  max_per_img_post=100)  # we reduce detections in post-processing instead
    )
)

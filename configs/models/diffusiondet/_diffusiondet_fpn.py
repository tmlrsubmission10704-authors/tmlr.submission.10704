# NOTE: Copy of projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
# Source: https://github.com/open-mmlab/mmdetection/blob/main/projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py

from mmcv.ops import RoIAlign, nms
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.losses import GIoULoss, FocalLoss, L1Loss
from mmdet.models.necks.fpn import FPN
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import \
    SingleRoIExtractor
from mmdet.models.task_modules import FocalLossCost, BBoxL1Cost, IoUCost

from mmdet_one_piece.models.detectors import DiffusionDet
from mmdet_one_piece.models.dense_heads import DynamicDiffusionDetHead, SingleDiffusionDetHead
from mmdet_one_piece.models.losses import DiffusionDetCriterion, DiffusionDetMatcher

# used for wandb logging and run_name
model_framework_name = 'diffusiondet'
model_backbone_name = None  # set in specific config
model_neck_name = 'fpn'

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True)
test_pipeline_LA=dict(with_bbox=True)

# model settings
model = dict(
    type=DiffusionDet,
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
        num_outs=4),
    bbox_head=dict(
        type=DynamicDiffusionDetHead,
        num_classes=80,
        feat_channels=256,
        num_proposals=500,
        num_heads=6,
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=4,
        ddim_sampling_eta=1.0,
        single_head=dict(
            type=SingleDiffusionDetHead,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
        roi_extractor=dict(
            type=SingleRoIExtractor,
            roi_layer=dict(type=RoIAlign, output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        # criterion
        criterion=dict(
            type=DiffusionDetCriterion,
            num_classes=80,
            assigner=dict(
                type=DiffusionDetMatcher,
                match_costs=[
                    dict(
                        # NOTE: this MUST be a string because DiffusionDetMatcher implements a str comparison...
                        #       also when using 'FedLoss'!
                        type='FocalLossCost',
                        alpha=0.25,
                        gamma=2.0,
                        weight=2.0,
                        eps=1e-8),
                    dict(type=BBoxL1Cost, weight=5.0, box_format='xyxy'),
                    dict(type=IoUCost, iou_mode='giou', weight=2.0)
                ],
                center_radius=2.5,
                candidate_topk=5),
            loss_cls=dict(
                type=FocalLoss,
                use_sigmoid=True,
                alpha=0.25,
                gamma=2.0,
                reduction='sum',
                loss_weight=2.0),
            loss_bbox=dict(type=L1Loss, reduction='sum', loss_weight=5.0),
            loss_giou=dict(type=GIoULoss, reduction='sum',
                           loss_weight=2.0))),
    test_cfg=dict(
        use_nms=True,
        score_thr=0.5,
        min_bbox_size=0,
        nms=dict(type=nms, iou_threshold=0.5, max_num=100),
    ))

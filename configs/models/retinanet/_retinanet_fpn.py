# NOTE: Copy of mmdet/configs/__base__/models/retinanet_r50_fpn.py but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/models/retinanet_r50_fpn.py

from mmcv.ops import nms
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.detectors import RetinaNet
from mmdet.models.dense_heads import RetinaHead
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.task_modules.prior_generators.anchor_generator import \
    AnchorGenerator
from mmdet.models.task_modules.assigners.max_iou_assigner import MaxIoUAssigner
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import \
    DeltaXYWHBBoxCoder
from mmdet.models.losses import FocalLoss, L1Loss
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.necks.fpn import FPN

# used for wandb logging and run_name
model_framework_name = 'retinanet'
model_backbone_name = None  # set in specific config
model_neck_name = 'fpn'

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True)
test_pipeline_LA=dict(with_bbox=True)

# model settings
model = dict(
    type=RetinaNet,
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
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type=RetinaHead,
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type=AnchorGenerator,
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type=DeltaXYWHBBoxCoder,
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type=L1Loss, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=MaxIoUAssigner,
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type=PseudoSampler),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type=nms, iou_threshold=0.5),
        max_per_img=100))

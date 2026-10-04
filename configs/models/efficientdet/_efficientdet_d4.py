# NOTE: Copy of mmdet/projects/EfficientDet/configs/* but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
#
# Source: https://github.com/open-mmlab/mmdetection/blob/main/projects/EfficientDet/configs/
#  Paper: https://arxiv.org/pdf/1911.09070
#
#     D4: 21M  55B, phi4 -> 1024 img_size, B4, BiFPN: 224 7 BoxLayers: 4
#     D5: 34M 135B, phi5 -> 1280 img_size, B5, BiFPN: 288 7 BoxLayers: 4
#
# NOTE: We leave img_size and backbone to the user but pre-configure BiFPN and BBox head
#
from mmcv.ops import soft_nms
from mmdet_one_piece.models.detectors import EfficientDet
from mmdet_one_piece.models.necks import BiFPN
from mmdet_one_piece.models.dense_heads import EfficientDetSepBNHead
from mmdet_one_piece.models.losses import HuberLoss
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import \
    DeltaXYWHBBoxCoder
from mmdet.models.task_modules.prior_generators.anchor_generator import \
    AnchorGenerator
from mmdet.models.task_modules.assigners.max_iou_assigner import MaxIoUAssigner
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.losses import FocalLoss

# used for wandb logging and run_name
model_framework_name = 'efficientdet_d4'
model_backbone_name = None  # set in specific config
model_neck_name = 'bifpn'

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True)
test_pipeline_LA=dict(with_bbox=True)

norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)
model = dict(
    type=EfficientDet,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=896),  # NOTE: everything else will break BiFPN, not 100% sure why, could be fixed but I have not time right now
    backbone=None,         #  NOTE: add in specific config
    neck=dict(
        type=BiFPN,
        num_stages=7,
        in_channels=None,  #  NOTE: add in specific config, depends on backbone!
        out_channels=224,
        start_level=0,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type=EfficientDetSepBNHead,
        num_classes=80,
        num_ins=5,
        in_channels=224,
        feat_channels=224,
        stacked_convs=4,
        norm_cfg=norm_cfg,
        anchor_generator=dict(
            type=AnchorGenerator,
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
        bbox_coder=dict(
            type=DeltaXYWHBBoxCoder,
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=1.5,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type=HuberLoss, beta=0.1, loss_weight=50)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=MaxIoUAssigner,
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
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
        nms=dict(
            type=soft_nms,
            iou_threshold=0.3,
            sigma=0.5,
            min_score=1e-3,
            method='gaussian'),
        max_per_img=100))

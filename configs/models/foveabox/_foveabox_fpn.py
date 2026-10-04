# NOTE: Copy of mmdet/configs/foveabox/* but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
#       we also change to pytorch style (RGB)
#
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/foveabox/fovea_r50_fpn_4xb4-1x_coco.py
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/foveabox/fovea_r50_fpn_gn-head-align_ms-640-800-4xb4-2x_coco.py

from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.detectors import FOVEA
from mmdet.models.dense_heads import FoveaHead
from mmcv.ops import nms
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.losses import FocalLoss, SmoothL1Loss
from mmdet.models.necks.fpn import FPN

# used for wandb logging and run_name
model_framework_name = 'foveabox'
model_backbone_name = None  # set in specific config
model_neck_name = 'fpn'

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True)
test_pipeline_LA=dict(with_bbox=True)

# model settings
model = dict(
    type=FOVEA,
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
        num_outs=5,
        add_extra_convs='on_input'),
    bbox_head=dict(
        type=FoveaHead,
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        base_edge_list=[16, 32, 64, 128, 256],
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        sigma=0.4,
        with_deform=True,  # from second config
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),  # from second config
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=1.50,
            alpha=0.4,
            loss_weight=1.0),
        loss_bbox=dict(type=SmoothL1Loss, beta=0.11, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=1000,
        score_thr=0.05,
        nms=dict(type=nms, iou_threshold=0.5),
        max_per_img=100))

# NOTE: Copy of mmdet/configs/solov2/solov2_r50_fpn_1x_coco.py but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/solov2/solov2_r50_fpn_1x_coco.py

from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.detectors import SOLOv2
from mmdet.models.dense_heads import SOLOV2Head
from mmdet.models.losses import DiceLoss, FocalLoss

# used for wandb logging and run_name
model_framework_name = 'solo_v2'
model_backbone_name = None  # set in specific config
model_neck_name = 'fpn'

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True, with_mask=True)
test_pipeline_LA=dict(with_bbox=True, with_mask=True)

# This will override val_evaluator and test_evaluator in train.py
derive_bbox_from_mask = True

# NOTE: we find solo will hang without this, not 100% sure what the cause is but the official repro did add it as well...
find_unused_parameters=True

# model settings
model = dict(
    type=SOLOv2,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=None,        #  NOTE: add in specific config
    neck=dict(
        type='FPN',
        in_channels=None, #  NOTE: add in specific config, depends on backbone!
        out_channels=256,
        start_level=0,
        num_outs=5),
    mask_head=dict(
        type=SOLOV2Head,
        num_classes=80,
        in_channels=256,
        feat_channels=512,
        stacked_convs=4,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        pos_scale=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        mask_feature_head=dict(
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=256,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        loss_mask=dict(type=DiceLoss, use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    # model training and testing settings
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=100))

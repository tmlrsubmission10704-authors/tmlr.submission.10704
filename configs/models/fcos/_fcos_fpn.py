# NOTE: Copy of mmdet/configs/fcos/* but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
#       we also change to pytorch style (RGB)
#
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/fcos/fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py

from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.detectors import FCOS
from mmdet.models.dense_heads import FCOSHead
from mmcv.ops import nms
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses import GIoULoss, FocalLoss, CrossEntropyLoss
from mmdet.models.necks.fpn import FPN

# used for wandb logging and run_name
model_framework_name = 'fcos'
model_backbone_name = None  # set in specific config
model_neck_name = 'fpn'

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True)
test_pipeline_LA=dict(with_bbox=True)

# model settings
model = dict(
    type=FCOS,
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
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type=FCOSHead,
        norm_on_bbox=True,       # 'Tricks'
        centerness_on_reg=True,  # 'Tricks'
        dcn_on_last_conv=False,  # 'Tricks'
        center_sampling=True,    # 'Tricks'
        conv_bias=True,          # 'Tricks'
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type=GIoULoss, loss_weight=1.0),
        loss_centerness=dict(
            type=CrossEntropyLoss, use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type=nms, iou_threshold=0.6),  # changed from 0.5 in fcos_r50-caffe_fpn_gn-head_1x.py
        max_per_img=100))

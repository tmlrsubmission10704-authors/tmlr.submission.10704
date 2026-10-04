# NOTE: Copy of mmdet/configs/vfnet/vfnet_r50_fpn_1x_coco.py but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/vfnet/vfnet_r50_fpn_1x_coco.py

from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmcv.ops import nms
from mmdet.models.detectors import VFNet
from mmdet.models.dense_heads import VFNetHead
from mmdet.models.losses import GIoULoss, VarifocalLoss
from mmdet.models.task_modules.assigners import ATSSAssigner
from mmdet.models.necks.fpn import FPN

# used for wandb logging and run_name
model_framework_name = 'vfnet'
model_backbone_name = None  # set in specific config
model_neck_name = 'fpn'

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True)
test_pipeline_LA=dict(with_bbox=True)

# model settings
model = dict(
    type=VFNet,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=None,        #  NOTE: add in specific config
    neck=dict(
        type=FPN,
        in_channels=None, #  NOTE: add in specific config, depends on backbone!
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type=VFNetHead,
        num_classes=80,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type=VarifocalLoss,
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type=GIoULoss, loss_weight=1.5),
        loss_bbox_refine=dict(type=GIoULoss, loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type=ATSSAssigner, topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type=nms, iou_threshold=0.6),
        max_per_img=100))

# NOTE: Copy of mmdet/configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
# Source: https://github.com/open-mmlab/mmdetection/tree/main/configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py

from mmengine.config import read_base
from mmdet.models.roi_heads import DynamicRoIHead
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import \
    Shared2FCBBoxHead
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import \
    DeltaXYWHBBoxCoder
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss

with read_base():
    from ..faster_rcnn._faster_rcnn_fpn import model, train_pipeline_LA, test_pipeline_LA

# used for wandb logging and run_name
model_framework_name = 'dynamic_rcnn'
model_backbone_name = None  # set in specific config
model_neck_name = 'fpn'

model.update(
    roi_head=dict(
        type=DynamicRoIHead,
        bbox_head=dict(
            type=Shared2FCBBoxHead,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type=DeltaXYWHBBoxCoder,
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type=SmoothL1Loss, beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn_proposal=dict(nms=dict(iou_threshold=0.85)),
        rcnn=dict(
            dynamic_rcnn=dict(
                iou_topk=75,
                beta_topk=10,
                update_iter_interval=100,
                initial_iou=0.4,
                initial_beta=1.0))),
    test_cfg=dict(rpn=dict(nms=dict(iou_threshold=0.85)))
)

# NOTE: Copy of mmdet/configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py but without 'backbone' and 'neck.in_channels'
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py

from mmdet_one_piece.models.detectors import ConditionalDETR_OP
from mmdet.models.dense_heads import ConditionalDETRHead
from mmdet.models.losses import FocalLoss
from mmdet.models.task_modules.assigners import HungarianAssigner, FocalLossCost, BBoxL1Cost, IoUCost

from mmengine.config import read_base
with read_base():
    from ..detr._detr_cm import model, train_pipeline_LA, test_pipeline_LA

# used for wandb logging and run_name
model_framework_name = 'conditional_detr'
model_backbone_name = None  # set in specific config
model_neck_name = 'cm'

# NOTE: we have to use merge here so that roi_layer and mask_head get replaced properly (_delete_)!
model.merge(dict(
    type=ConditionalDETR_OP,
    num_queries=300,
    backbone=None,         #  NOTE: add in specific config
    neck=dict(
        in_channels=None,  #  NOTE: add in specific config, depends on backbone!
    ),
    decoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                _delete_=True,
                embed_dims=256,
                num_heads=8,
                attn_drop=0.1,
                cross_attn=False),
            cross_attn_cfg=dict(
                _delete_=True,
                embed_dims=256,
                num_heads=8,
                attn_drop=0.1,
                cross_attn=True))),
    bbox_head=dict(
        type=ConditionalDETRHead,
        loss_cls=dict(
            _delete_=True,
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=FocalLossCost, weight=2.0),
                dict(type=BBoxL1Cost, weight=5.0, box_format='xywh'),
                dict(type=IoUCost, iou_mode='giou', weight=2.0)
            ])))
)

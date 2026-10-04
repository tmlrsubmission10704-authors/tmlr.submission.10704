# NOTE: Copy of mmdet/configs/point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
# Source: https://github.com/open-mmlab/mmdetection/tree/main/configs/point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py
from mmengine.config import read_base

from mmdet.models.detectors import PointRend
from mmdet.models.roi_heads import PointRendRoIHead, GenericRoIExtractor, CoarseMaskHead, MaskPointHead
from mmcv.ops.point_sample import SimpleRoIAlign
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss

from mmdet_one_piece.models.roi_heads import MaskPointHead_OP, PointRendRoIHead_OP

with read_base():
    from ..mask_rcnn._mask_rcnn_fpn import model, train_pipeline_LA, test_pipeline_LA

# used for wandb logging and run_name
model_framework_name = 'point_rend'
model_backbone_name = None  # set in specific config
model_neck_name = 'fpn'

# NOTE: we have to use merge here so that roi_layer and mask_head get replaced properly (_delete_)!
model.merge(dict(
    type=PointRend,
    roi_head=dict(
        #type=PointRendRoIHead,
        type=PointRendRoIHead_OP,
        mask_roi_extractor=dict(
            type=GenericRoIExtractor,
            aggregation='concat',
            roi_layer=dict(
                _delete_=True, type=SimpleRoIAlign, output_size=14),
            out_channels=256,
            featmap_strides=[4]),
        mask_head=dict(
            _delete_=True,
            type=CoarseMaskHead,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=80,
            loss_mask=dict(
                type=CrossEntropyLoss, use_mask=True, loss_weight=1.0)),
        point_head=dict(
            #type=MaskPointHead,
            type=MaskPointHead_OP,
            num_fcs=3,
            in_channels=256,
            fc_channels=256,
            num_classes=80,
            coarse_pred_each_layer=True,
            loss_point=dict(
                type=CrossEntropyLoss, use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            mask_size=7,
            num_points=14 * 14,
            oversample_ratio=3,
            importance_sample_ratio=0.75)),
    test_cfg=dict(
        rcnn=dict(
            subdivision_steps=5,
            subdivision_num_points=28 * 28,
            scale_factor=2))
))

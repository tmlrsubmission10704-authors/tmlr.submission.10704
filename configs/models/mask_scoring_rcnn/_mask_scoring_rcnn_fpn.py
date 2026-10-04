# NOTE: Copy of mmdet/configs/ms_rcnn/ms-rcnn_r50_fpn_1x_coco.py but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
# Source: https://github.com/open-mmlab/mmdetection/tree/main/configs/ms_rcnn/ms-rcnn_r50_fpn_1x_coco.py
from mmengine.config import read_base

from mmdet.models.detectors import MaskScoringRCNN
from mmdet.models.roi_heads import MaskScoringRoIHead, MaskIoUHead

with read_base():
    from ..mask_rcnn._mask_rcnn_fpn import model, train_pipeline_LA, test_pipeline_LA

# used for wandb logging and run_name
model_framework_name = 'mask_scoring_rcnn'
model_backbone_name = None  # set in specific config
model_neck_name = 'fpn'

model.update(
    type=MaskScoringRCNN,
    roi_head=dict(
        type=MaskScoringRoIHead,
        mask_iou_head=dict(
            type=MaskIoUHead,
            num_convs=4,
            num_fcs=2,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=80)),
    # model training and testing settings
    train_cfg=dict(rcnn=dict(mask_thr_binary=0.5))
)

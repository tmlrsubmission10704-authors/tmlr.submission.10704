# NOTE: V2 LN  - This version is similar to mask_rcnn_v2 but applied to cascade_mask_rcnn
#                This version does NOT use giou! The LN upgrade was introduced in VitDet paper!
#
# Improved Mask R-CNN model as introduced in https://arxiv.org/pdf/2111.11429 (section 2.2 Upgraded Modules)
#
# - (1) following the convolutions in FPN with batch normalization (BN) [23],
# - (2) using two convolutional layers in the region proposal network (RPN) [33] instead of one,
# - (3) using four convolutional layers with BN followed by one linear layer for the region-of-interest (RoI)
#       classification and box regression head [39] instead of a two-layer MLP without normalization,
# - (4) and following the convolutions in the standard mask head with BN. Wherever BN is applied, we use 
#       synchronous BN across all GPUs
#
# tv: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py#L519
# d2: https://github.com/facebookresearch/detectron2/blob/main/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py
# mm: https://github.com/open-mmlab/mmdetection/blob/main/configs/strong_baselines/mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-100e_coco.py
#
from mmengine.config import read_base

from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead
# same as ConvFCBBoxHead but already configured, just here for reference to check if you want
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared4Conv1FCBBoxHead
from mmpretrain.models.utils import LayerNorm2d

with read_base():
    from ..cascade_mask_rcnn._cascade_mask_rcnn_fpn import *

model_framework_name = 'cascade_mask_rcnn_v2_ln'
norm_cfg = dict(type=LayerNorm2d, requires_grad=True)  # NOTE: only applies to neck, rpn, roi_heads, not the backbone!
model.update(
    neck=dict(norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=[
            dict(
                type=ConvFCBBoxHead,
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type=DeltaXYWHBBoxCoder,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                reg_decoded_bbox=False,
                norm_cfg=norm_cfg,
                loss_cls=dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type=SmoothL1Loss, beta=1.0, loss_weight=1.0)),
            dict(
                type=ConvFCBBoxHead,
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type=DeltaXYWHBBoxCoder,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                reg_decoded_bbox=False,
                norm_cfg=norm_cfg,
                loss_cls=dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type=SmoothL1Loss, beta=1.0, loss_weight=1.0)),
            dict(
                type=ConvFCBBoxHead,
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type=DeltaXYWHBBoxCoder,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                reg_decoded_bbox=False,
                norm_cfg=norm_cfg,
                loss_cls=dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type=SmoothL1Loss, beta=1.0, loss_weight=1.0)),
        ],
        mask_head=dict(norm_cfg=norm_cfg))
)

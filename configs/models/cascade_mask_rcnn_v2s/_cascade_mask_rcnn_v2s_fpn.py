# NOTE: V2s SBN - This Cascade Mask RCNN version was used in SWIN and every work that copied their detection code
#                 In comparison to v2, it only uses the 4C1FC bbox head with sbn and GIoU Loss instead of L1!
#
# From the improved Mask R-CNN model as introduced in https://arxiv.org/pdf/2111.11429 (section 2.2 Upgraded Modules)
# it only implements:
#
# - (3) using four convolutional layers with BN followed by one linear layer for the region-of-interest (RoI)
#       classification and box regression head [39] instead of a two-layer MLP without normalization,
#
# which was introduced in the Group Norm paper (code actually). In addition, it uses GIoU loss instead of L1
#
# Default CMRCNN: https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/_base_/models/cascade_mask_rcnn_swin_fpn.py
# Swin Upgrades:  https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py
# NOTE: reg_class_agnostic=False, (True in default CMRCNN)
# NOTE: reg_decoded_bbox=True, (because of GIoU loss)
from mmengine.config import read_base

from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead

with read_base():
    from ..cascade_mask_rcnn._cascade_mask_rcnn_fpn import *

model_framework_name = 'cascade_mask_rcnn_v2s'
norm_cfg = dict(type='SyncBN', requires_grad=True)  # this only applies to bbox_head, not the backbone
model.update(
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
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=norm_cfg,
                loss_cls=dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
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
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=norm_cfg,
                loss_cls=dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
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
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=norm_cfg,
                loss_cls=dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]
    )
)

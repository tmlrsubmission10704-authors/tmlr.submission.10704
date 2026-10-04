# Improved Mask R-CNN model as introduced in https://arxiv.org/pdf/2111.11429 (section 2.2 Upgraded Modules)
# NOTE: This version uses LayerNorm as introduced in the VitDet paper
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
# NOTE: some papers like ViT-Det use this version but may change small things, like LN instead of SyncBN
# TODO: document papers that use improved Mask RCNN
#
from mmengine.config import read_base
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared4Conv1FCBBoxHead
from mmpretrain.models.utils import LayerNorm2d

with read_base():
    from ..mask_rcnn._mask_rcnn_fpn import *

model_framework_name = 'mask_rcnn_v2_ln'
norm_cfg = dict(type=LayerNorm2d, requires_grad=True)  # NOTE: only applies to neck, rpn, roi_heads, not the backbone!
model.update(
    neck=dict(norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type=Shared4Conv1FCBBoxHead,
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg))
)

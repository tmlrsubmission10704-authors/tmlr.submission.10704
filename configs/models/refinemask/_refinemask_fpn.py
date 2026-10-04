from mmengine.config import read_base

from mmdet.models.detectors import PointRend
from mmdet.models.roi_heads import PointRendRoIHead, GenericRoIExtractor, CoarseMaskHead, MaskPointHead
from mmcv.ops.point_sample import SimpleRoIAlign
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss

from mmdet_one_piece.models.roi_heads import RefineRoIHead, RefineMaskHead, SimpleRefineRoIHead, SimpleRefineMaskHead
from mmdet_one_piece.models.losses.cross_entropy_loss import RefineCrossEntropyLoss

with read_base():
    from ..mask_rcnn._mask_rcnn_fpn import model, train_pipeline_LA, test_pipeline_LA

# used for wandb logging and run_name
model_framework_name = 'refinemask'
model_backbone_name = None  # set in specific config
model_neck_name = 'fpn'

# Source: https://github.com/zhanggang001/RefineMask/blob/main/configs/refinemask/coco/r50-refinemask-1x.py
#         https://github.com/zhanggang001/RefineMask/blob/main/configs/refinemask/coco/r50-refinemask-1x-faster-better.py

# NOTE: we have to use merge here so that mask_head gets replaced properly (_delete_)!
model.merge(dict(
    type=PointRend,
    roi_head=dict(
        type=RefineRoIHead,
        bbox_head=dict(
            loss_cls=dict(loss_weight=2.0),
            loss_bbox=dict(loss_weight=2.0)
        ),
        mask_head=dict(
            _delete_=True,
            type=RefineMaskHead,
                 num_convs_instance=2,
                 num_convs_semantic=4,
                 conv_in_channels_instance=256,
                 conv_in_channels_semantic=256,
                 conv_kernel_size_instance=3,
                 conv_kernel_size_semantic=3,
                 conv_out_channels_instance=256,
                 conv_out_channels_semantic=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 dilations=[1, 3, 5],
                 semantic_out_stride=4,
                 mask_use_sigmoid=True,
                 stage_num_classes=[80, 80, 80, 80],
                 stage_sup_size=[14, 28, 56, 112],
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                 loss_cfg=dict(
                    type=RefineCrossEntropyLoss,
                    stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                    semantic_loss_weight=1.0,
                    boundary_width=2,
                    start_stage=1)
        )
    )
))

# NOTE: Implementation differences 
# RefineROIHead:
#   - forward_train has an additional loss loss_semantic from _mask_forward_train
#   - _mask_forward_train has additional arguments and is called with these in forward_train
#   - _mask_forward is different
#   - simple_test_mask seems to be the 'predict', see simple_test in standard_roi_head.py

# mmdet commits to follow for the change
# Support Datasampler 2631e2879acf0bd20a64dfdd7039f37a8e6afbf6
# Refactor interface of two-stage detector 46430db4f965a2d7e2853ce52cad828344e84ec7
# refactor two-stage model 60dc9ae489851a45a2f954ece8eb6459475d9437 

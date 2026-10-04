# NOTE: Copy from mmdet/configs/htc/htc_r50_fpn_1x_coco.py
#       without 'backbone' and 'neck.in_channels' and adapted to lazy cfg!
#
# NOTE: HTC requires 'cocostuff' semantic segmentation for training!
#       Make sure to use ms_coco_stuffthings.py instead of ms_coco.py
#
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/htc/htc_r50_fpn_1x_coco.py

from mmengine.config import read_base
from mmcv.ops import RoIAlign
from mmdet.models.roi_heads.mask_heads import FusedSemanticHead
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import \
    SingleRoIExtractor
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss

with read_base():
    from ._htc_fpn_without_semantic import model, model_framework_name, model_neck_name

model_framework_name = 'htc'

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True, with_mask=True, with_seg=True)
test_pipeline_LA=dict(with_bbox=True, with_mask=True, with_seg=True)

model.update(
    data_preprocessor=dict(pad_seg=True),
    roi_head=dict(
        semantic_roi_extractor=dict(
            type=SingleRoIExtractor,
            roi_layer=dict(type=RoIAlign, output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type=FusedSemanticHead,
            num_ins=5,
            fusion_level=1,
            seg_scale_factor=1 / 8,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=183,
            loss_seg=dict(type=CrossEntropyLoss, ignore_index=255, loss_weight=0.2)
        )
    )
)

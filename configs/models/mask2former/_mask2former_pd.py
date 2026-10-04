# NOTE: Copy of mmdet/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py but without 'backbone' and 'neck.in_channels' and adapted to lazy cfg
# Source: https://github.com/open-mmlab/mmdetection/blob/main/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py
# Merged: https://github.com/open-mmlab/mmdetection/blob/main/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py
#
# We merged the panoptic config here! If you want to train or run the panoptic mode, note that you also have to use
# matching dataset/dataloader etc..
#
# they use a ~50 epochs schedule which is expressed as iterations though:
# 327778 / (118000/16) -> 44 epochs
# 355092 / (118000/16) -> 48 epochs
# 368750 / (118000/16) -> 50 epochs

from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor, BatchFixedSizePad
from mmdet.models.detectors import Mask2Former
from mmdet.models.dense_heads import Mask2FormerHead
from mmdet.models.seg_heads import MaskFormerFusionHead
from mmdet.models.losses import CrossEntropyLoss, DiceLoss
from mmdet.models.task_modules.assigners import HungarianAssigner, ClassificationCost, CrossEntropyLossCost, DiceCost
from mmdet.models.task_modules.samplers import MaskPseudoSampler
from mmdet.models.layers import MSDeformAttnPixelDecoder

from mmdet_one_piece.models.seg_heads import MaskFormerFusionHead_OP

# used for wandb logging and run_name
model_framework_name = 'mask2former'
model_backbone_name = None  # set in specific config
model_neck_name = 'pd'  # pixel decoder

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True, with_mask=True)
test_pipeline_LA=dict(with_bbox=True, with_mask=True)

# model settings
num_things_classes = 80
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
image_size = (1024, 1024)
batch_augments = [
    dict(
        type=BatchFixedSizePad,
        size=image_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)    # NOTE: deactivated for instance segmentation
]
data_preprocessor = dict(
    type=DetDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,        # NOTE: deactivated for instance segmentation
    batch_augments=batch_augments)  # NOTE: batch batch_augments are only used in training!

num_things_classes = 80
num_stuff_classes = 0      # NOTE: 0 for instance segmentation mode
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type=Mask2Former,
    data_preprocessor=data_preprocessor,
    backbone=None,         #  NOTE: add in specific config
    panoptic_head=dict(
        type=Mask2FormerHead,
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type=MSDeformAttnPixelDecoder,
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type=CrossEntropyLoss,
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type=DiceLoss,
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        #type=MaskFormerFusionHead,
        type=MaskFormerFusionHead_OP,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=ClassificationCost, weight=2.0),
                dict(
                    type=CrossEntropyLossCost, weight=5.0, use_sigmoid=True),
                dict(type=DiceCost, weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type=MaskPseudoSampler)),
    test_cfg=dict(
        panoptic_on=False,  # NOTE: deactivate for instance segmentation mode
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
)

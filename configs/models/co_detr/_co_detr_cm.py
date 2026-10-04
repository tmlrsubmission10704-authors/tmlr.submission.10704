# NOTE: Copy of projects but without 'backbone' and 'neck.in_channels'
# Source: https://github.com/open-mmlab/mmdetection/blob/main/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py

from mmcv.ops import RoIAlign, nms, soft_nms
from mmcv.cnn.bricks.transformer import MultiheadAttention

from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor
from mmdet.models.necks import ChannelMapper
from mmdet.models.losses import GIoULoss, L1Loss, CrossEntropyLoss, QualityFocalLoss, FocalLoss
from mmdet.models.task_modules.assigners import FocalLossCost, BBoxL1Cost, IoUCost, HungarianAssigner, ATSSAssigner
from mmdet.models.layers.positional_encoding import SinePositionalEncoding
from mmdet.models.dense_heads.rpn_head import RPNHead
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import \
    Shared2FCBBoxHead
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import \
    SingleRoIExtractor
from mmdet.models.task_modules.assigners.max_iou_assigner import MaxIoUAssigner
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import \
    DeltaXYWHBBoxCoder
from mmdet.models.task_modules.prior_generators.anchor_generator import \
    AnchorGenerator
from mmdet.models.task_modules.samplers.random_sampler import RandomSampler

from mmdet_one_piece.models.detectors import (CoDETR, CoDINOHead, CoDinoTransformer, CoDinoTransformer, 
                                              DetrTransformerEncoder, BaseTransformerLayer,
                                              MultiScaleDeformableAttention, DinoTransformerDecoder,
                                              DetrTransformerDecoderLayer, CoStandardRoIHead, CoATSSHead)


# used for wandb logging and run_name
model_framework_name = 'co_detr'
model_backbone_name = None  # set in specific config
model_neck_name = 'cm'

# This will be checked against the pipelines and merged in train.py
# NOTE: for training, we need masks but not for testing!
train_pipeline_LA=dict(with_bbox=True, with_mask=True)
test_pipeline_LA=dict(with_bbox=True)


# model settings
num_dec_layer = 6
loss_lambda = 2.0
num_classes = 80

#image_size = (1024, 1024)
#batch_augments = [
#    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
#]
model = dict(
    type=CoDETR,
    # If using the lsj augmentation,
    # it is recommended to set it to True.
    use_lsj=True,
    # detr: 52.1
    # one-stage: 49.4
    # two-stage: 47.9
    eval_module='detr',  # in ['detr', 'one-stage', 'two-stage']
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=1
        #batch_augments=batch_augments
        ),
    backbone=None,         #  NOTE: add in specific config
    neck=dict(
        type=ChannelMapper,
        in_channels=None,  #  NOTE: add in specific config, depends on backbone!
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    query_head=dict(
        type=CoDINOHead,
        num_query=900,
        num_classes=num_classes,
        in_channels=2048,
        as_two_stage=True,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=1.0,
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type=CoDinoTransformer,
            with_coord_feat=False,
            num_co_heads=2,  # ATSS Aux Head + Faster RCNN Aux Head
            num_feature_levels=5,
            encoder=dict(
                type=DetrTransformerEncoder,
                num_layers=6,
                # number of layers that use checkpoint.
                # The maximum value for the setting is num_layers.
                # FairScale must be installed for it to work.
                with_cp=4,
                transformerlayers=dict(
                    type=BaseTransformerLayer,
                    attn_cfgs=dict(
                        type=MultiScaleDeformableAttention,
                        embed_dims=256,
                        num_levels=5,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type=DinoTransformerDecoder,
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type=DetrTransformerDecoderLayer,
                    attn_cfgs=[
                        dict(
                            type=MultiheadAttention,
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type=MultiScaleDeformableAttention,
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type=SinePositionalEncoding,
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(  # Different from the DINO
            type=QualityFocalLoss,
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type=L1Loss, loss_weight=5.0),
        loss_iou=dict(type=GIoULoss, loss_weight=2.0)),
    rpn_head=dict(
        type=RPNHead,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type=AnchorGenerator,
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type=DeltaXYWHBBoxCoder,
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            loss_weight=1.0 * num_dec_layer * loss_lambda),
        loss_bbox=dict(
            type=L1Loss, loss_weight=1.0 * num_dec_layer * loss_lambda)),
    roi_head=[
        dict(
            type=CoStandardRoIHead,
            bbox_roi_extractor=dict(
                type=SingleRoIExtractor,
                roi_layer=dict(
                    type=RoIAlign, output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type=Shared2FCBBoxHead,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type=DeltaXYWHBBoxCoder,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type=CrossEntropyLoss,
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(
                    type=GIoULoss,
                    loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    bbox_head=[
        dict(
            type=CoATSSHead,
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type=AnchorGenerator,
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type=DeltaXYWHBBoxCoder,
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type=FocalLoss,
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(
                type=GIoULoss,
                loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type=CrossEntropyLoss,
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type=HungarianAssigner,
                match_costs=[
                    dict(type=FocalLossCost, weight=2.0),
                    dict(type=BBoxL1Cost, weight=5.0, box_format='xywh'),
                    dict(type=IoUCost, iou_mode='giou', weight=2.0)
                ])),
        dict(
            rpn=dict(
                assigner=dict(
                    type=MaxIoUAssigner,
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type=RandomSampler,
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                # NOTE: we add and increase split_thr from the default 10k to allow amp train
                #       batch_nms will split the operation otherwise which gives dtype errors
                # see:  https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/nms.py#L300
                nms=dict(type=nms, iou_threshold=0.7, split_thr=20000),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type=MaxIoUAssigner,
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type=RandomSampler,
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type=ATSSAssigner, topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
    ],
    test_cfg=[
        # Deferent from the DINO, we use the NMS.
        dict(
            max_per_img=100,
            # NMS can improve the mAP by 0.2.
            nms=dict(type=soft_nms, iou_threshold=0.8)),
        dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type=nms, iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                nms=dict(type=nms, iou_threshold=0.5),
                max_per_img=100)),
        dict(
            # atss bbox head:
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type=nms, iou_threshold=0.6),
            max_per_img=100),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ])


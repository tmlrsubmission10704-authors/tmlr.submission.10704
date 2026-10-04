# https://github.com/MCG-NJU/AdaMixer/blob/a50e33d766c68e0df9ac592913ed40a528914fab/configs/adamixer/adamixer_r50_1x_coco.py

from mmdet.models.detectors import SparseRCNN
from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor

from mmdet_one_piece.models.dense_heads import InitialQueryGenerator
from mmdet_one_piece.models.necks import ChannelMapping
from mmdet_one_piece.models.roi_heads import AdaMixerRoIHead
from mmdet_one_piece.models.roi_heads.bbox_heads import AdaMixerDecoderStage


# used for wandb logging and run_name
model_framework_name = 'adamixer'
model_backbone_name = None  # set in specific config
model_neck_name = 'cm'

# This will be checked against the pipelines and merged in train.py
# NOTE: for training, we need masks but not for testing!
train_pipeline_LA=dict(with_bbox=True)
test_pipeline_LA=dict(with_bbox=True)

num_stages = 6
in_points_list = [32, ] * num_stages  # P_in for spatial mixing in the paper.
out_patterns_list = [128, ] * num_stages  # P_out for spatial mixing in the paper. Also named as `out_points` in this codebase.
n_group_list = [4, ] * num_stages  # G for the mixer grouping in the paper. Please distinguishe it from num_heads in MHSA in this codebase.

num_query = 100
model = dict(
    #type='QueryBased',
    type=SparseRCNN,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=None,         #  NOTE: add in specific config
    neck=dict(
        type=ChannelMapping,  #  NOTE: This is NOT ChannelMapper, slightly different!
        in_channels=None,     #  NOTE: add in specific config, depends on backbone!
        out_channels=256,  # FEAT_DIM = 256
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4),
    rpn_head=dict(
        type=InitialQueryGenerator,
        num_query=num_query,  # num_query = 100
        content_dim=256),     # QUERY_DIM = 256
    roi_head=dict(
        type=AdaMixerRoIHead,
        featmap_strides=[4, 8, 16, 32],
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        content_dim=256,   # QUERY_DIM = 256
        bbox_head=[
            dict(
                type=AdaMixerDecoderStage,  # TODO: PORT
                num_classes=80,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=1,
                feedforward_channels=2048,  # FF_DIM = 2048
                content_dim=256,            # QUERY_DIM = 256
                feat_channels=256,          # FEAT_DIM = 256
                dropout=0.0,
                in_points=in_points_list[stage_idx],
                out_points=out_patterns_list[stage_idx],
                n_groups=n_group_list[stage_idx],
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                # NOTE: The following argument is a placeholder to hack the code. No real effects for decoding or updating bounding boxes.
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for stage_idx in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ]),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=100)))   # max_per_img is usually set to num_query but we enforce 100 to match other models!

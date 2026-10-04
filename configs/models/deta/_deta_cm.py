from mmdet.models.data_preprocessors.data_preprocessor import \
    DetDataPreprocessor

from mmdet_one_piece.models.detectors.deta.deta import DETA


# used for wandb logging and run_name
model_framework_name = 'deta'
model_backbone_name = None  # set in specific config
model_neck_name = 'cm'  # NOTE: this is implemented in their DeformableDETR

# This will be checked against the pipelines and merged in train.py
train_pipeline_LA=dict(with_bbox=True)
test_pipeline_LA=dict(with_bbox=True)

model = dict(
    type=DETA,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=None,  # NOTE: add in specific config
    # neck=None,    # NOTE: no explicit neck, the cm is implemented in their DeformableDETR!
    with_box_refine = True,
    two_stage = True,
    num_feature_levels = 5,
    num_queries = 900,
    dim_feedforward = 2048,
    dropout = 0.0,
    cls_loss_coef = 1.0,
    assign_first_stage = True,
    assign_second_stage = True,
)

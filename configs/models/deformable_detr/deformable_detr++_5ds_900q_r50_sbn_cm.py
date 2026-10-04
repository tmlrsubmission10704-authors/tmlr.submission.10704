from mmengine.config import read_base

with read_base():
    from .deformable_detr_r50_sbn_cm import *

model_framework_name = 'deformable_detr++_5ds_900'
model.update(
    num_queries=900,
    num_feature_levels=5,
    with_box_refine=True,
    as_two_stage=True,
    neck=dict(num_outs=5),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=5))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=5)))
)

from mmengine.config import read_base

with read_base():
    from .deformable_detr_r50_sbn_cm import *

model_framework_name = 'deformable_detr++_900'
model.update(
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
)

from mmengine.config import read_base

with read_base():
    from .deformable_detr_r50_cm import *

model_framework_name = 'deformable_detr+'
model.update(
    with_box_refine=True
)

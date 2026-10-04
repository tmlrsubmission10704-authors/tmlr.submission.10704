from mmengine.config import read_base

with read_base():
    from .dab_detr_r50_sbn_cm import *

model_framework_name = 'dab_detr_900'
model.update(num_queries=900)

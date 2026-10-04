from mmengine.config import read_base

with read_base():
    from ._efficientdet_d4 import *

model_framework_name = 'efficientdet_d5'
model.update(
    neck=dict(out_channels=288),
    bbox_head=dict(
        in_channels=288,
        feat_channels=288,
    )
)

from mmengine.config import read_base

with read_base():
    from ._diffusiondet_fpn import *

# https://github.com/open-mmlab/mmdetection/blob/main/configs/sparse_rcnn/sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco.py
model_framework_name = 'diffusiondet_900p'

model.update(bbox_head=dict(num_proposals=900))

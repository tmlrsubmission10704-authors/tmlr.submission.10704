from mmengine.config import read_base

with read_base():
    from ._sparse_rcnn_fpn import *

# https://github.com/open-mmlab/mmdetection/blob/main/configs/sparse_rcnn/sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco.py
model_framework_name = 'sparse_rcnn_300p'

num_proposals = 300
model.update(
    rpn_head=dict(num_proposals=num_proposals),
    test_cfg=dict(rcnn=dict(max_per_img=100))  # max_per_img is usually set to num_proposals but we enforce 100 to match other models!
)

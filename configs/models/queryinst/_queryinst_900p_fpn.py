from mmengine.config import read_base

with read_base():
    from ._queryinst_fpn import *

# https://github.com/open-mmlab/mmdetection/blob/main/configs/sparse_rcnn/sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco.py
model_framework_name = 'sparse_rcnn_900p'

num_proposals = 900
model.update(
    rpn_head=dict(num_proposals=num_proposals),
    test_cfg=dict(rcnn=dict(max_per_img=num_proposals, max_per_img_post=100))
)

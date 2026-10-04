from mmengine.config import read_base

with read_base():
    from ._adamixer_cm import *

# https://github.com/MCG-NJU/AdaMixer/blob/a50e33d766c68e0df9ac592913ed40a528914fab/configs/adamixer/adamixer_r50_300_query_crop_mstrain_480-800_3x_coco.py
model_framework_name = 'adamixer_300q'

num_query = 300
model.update(
    rpn_head=dict(num_query=num_query),
    test_cfg=dict(rcnn=dict(max_per_img=100))  # max_per_img is usually set to num_proposals but we enforce 100 to match other models!
)

from mmengine.config import read_base
from mmdet.datasets.transforms.loading import FilterAnnotations

# Large Scale Jitter (resize) training for Mask2Former, slightly different from the default lsj
# https://github.com/open-mmlab/mmdetection/blob/main/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py#L38
with read_base():
    from .train_lsj import *

assert train_pipeline[-2]['type'] == FilterAnnotations
train_pipeline[-3] = dict(type=FilterAnnotations, min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True)

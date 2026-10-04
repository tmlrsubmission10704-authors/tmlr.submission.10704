from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms import RandomResize
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import RandomFlip, Resize

# Default multiscale (resize) training with hflip as used in mmdetection
#
# NOTE: Resize will do a 'ResizeShortestEdge' with RandomResize sampling min_size as
#       np.random.randint(640, 800+1) and max_size np.random.randint(1333, 1333+1).
#       See: https://github.com/open-mmlab/mmcv/blob/main/mmcv/transforms/processing.py#L1381
#
# Source: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/configs/common/ms_3x_coco_instance.py#L49
train_pipeline_name = 'ms_mm'  # used for wandb logging and run_name
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),  # update dynamically in train.py
    dict(type=LoadAnnotations), # with_bbox=, with_mask= and with_segm= will be set in train.py as it depends on the model!
    dict(type=RandomResize, resize_type=Resize, scale=[(1333, 640), (1333, 800)], keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

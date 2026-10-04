from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms import RandomChoiceResize
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import RandomFlip, Resize

# Default multiscale (resize) training with hflip as used in detectron 2
#
# NOTE: Resize will do a 'ResizeShortestEdge' with RandomChoiceResize sampling (min_size, max_size)
#       from the list of scales. In detectron 2 these values are configured as:
#       MIN_SIZE_TRAIN: (640, 672, 704, 736, 768, 800)
#       MAX_SIZE_TRAIN = 1333
#
# Source:
#   https://github.com/facebookresearch/detectron2/blob/b7c7f4ba82192ff06f2bbb162b9f67b00ea55867/detectron2/config/defaults.py#L50
#   https://github.com/facebookresearch/detectron2/blob/b7c7f4ba82192ff06f2bbb162b9f67b00ea55867/configs/Base-RCNN-FPN.yaml#L41
#   https://github.com/facebookresearch/detectron2/blob/92ae9f0b92aba5867824b4f12aa06a22a60a45d3/detectron2/data/detection_utils.py#L629
#   https://github.com/facebookresearch/detectron2/blob/92ae9f0b92aba5867824b4f12aa06a22a60a45d3/detectron2/data/transforms/augmentation_impl.py#L82
train_pipeline_name = 'ms_d2'  # used for wandb logging and run_name
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),  # update dynamically in train.py
    dict(type=LoadAnnotations), # with_bbox=, with_mask= and with_segm= will be set in train.py as it depends on the model!
    dict(type=RandomChoiceResize, resize_type=Resize,
         scales=[(640, 1333), (672, 1333), (704, 1333),
                 (736, 1333), (768, 1333), (800, 1333)],
         keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

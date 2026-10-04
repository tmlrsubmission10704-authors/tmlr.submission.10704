from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms import RandomChoiceResize
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import RandomFlip, Resize

# Default multiscale (resize) training with hflip as used in torchvision reference implementation
#
# NOTE: Resize will do a 'ResizeShortestEdge' with RandomChoiceResize sampling (min_size, max_size)
#       from the list of scales. In the torchvision reference implementation these values are configured as:
#       min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
#       max_size=1333
#
# NOTE: I'm not entirely sure where this range originates from but most likely, from detr.
#
# Source: https://github.com/pytorch/vision/blob/main/references/detection/presets.py#L51
train_pipeline_name = 'ms_tv'  # used for wandb logging and run_name
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),  # update dynamically in train.py
    dict(type=LoadAnnotations), # with_bbox=, with_mask= and with_segm= will be set in train.py as it depends on the model!
    dict(type=RandomChoiceResize, resize_type=Resize,
         scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                 (736, 1333), (768, 1333), (800, 1333)],
         keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

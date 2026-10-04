from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import RandomFlip, Resize

# Standard image pipeline as introduced in Mask R-CNN https://arxiv.org/pdf/1703.06870 (or before)
# with shortest edge resize and hflip.
#
# NOTE: The paper also used multiscale training, see train_ms_* configs for that!
#
# Source: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/configs/_base_/datasets/coco_instance.py#L31
train_pipeline_name = 'hflip'  # used for wandb logging and run_name
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),  # update dynamically in train.py
    dict(type=LoadAnnotations), # with_bbox=, with_mask= and with_segm= will be set in train.py as it depends on the model!
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

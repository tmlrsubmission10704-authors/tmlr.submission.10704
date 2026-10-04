
from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms import RandomResize
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations, FilterAnnotations
from mmdet.datasets.transforms.transforms import RandomFlip, RandomCrop, Resize

# Large Scale Jitter (resize) training as used in detectron 2
# "New baselines using Large-Scale Jitter and Longer Training Schedule"
#
# LSJ was introduced in Simple Copy-Paste Data Augmentation https://arxiv.org/pdf/2012.07177.pdf
#
# Source:
# https://github.com/facebookresearch/detectron2/blob/main/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py#L40
# https://github.com/open-mmlab/mmdetection/blob/main/configs/common/lsj-100e_coco-instance.py
train_pipeline_name = 'lsj'  # used for wandb logging and run_name
image_size = (1024, 1024)
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),   # update dynamically in train.py
    dict(type=LoadAnnotations), # with_bbox=, with_mask= and with_segm= will be set in train.py as it depends on the model!
    dict(
        type=RandomResize,
        resize_type=Resize,
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type=RandomCrop,
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type=FilterAnnotations, min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

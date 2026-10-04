from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms import RandomChoice, RandomChoiceResize
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import RandomFlip, RandomCrop, Resize

# RandomChoice multiscale training as introduced in DETR
# - DETR: https://arxiv.org/abs/2005.12872
#   code: https://github.com/facebookresearch/detr/tree/main
#         (see d2/config and d2/detr/dataset_mapper.py)
#
# NOTE: This pipeline was also used by:
#
#       - Swin: https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/tree/master/configs/swin
#               https://github.com/open-mmlab/mmdetection/tree/main/configs/swin
#       - PVT(v2): https://github.com/whai362/PVT/tree/v2/detection
#                  https://github.com/open-mmlab/mmdetection/tree/main/configs/pvt (no cfg with ms training)
#
#       for their detection models, e.g. Mask R-CNN, Cascade Mask R-CNN, Sparse RCNN, ...
#
# Source: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/configs/detr/detr_r50_8xb2_150e_coco.py#L115
train_pipeline_name = 'ms_detr'  # used for wandb logging and run_name
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),  # update dynamically in train.py
    dict(type=LoadAnnotations), # with_bbox=, with_mask= and with_segm= will be set in train.py as it depends on the model!
    dict(type=RandomFlip, prob=0.5),
    dict(type=RandomChoice, transforms=[
        [
            dict(type=RandomChoiceResize, resize_type=Resize,
                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                         (736, 1333), (768, 1333), (800, 1333)],
                 keep_ratio=True)
        ],
        [
            dict(type=RandomChoiceResize, resize_type=Resize,
                 scales=[(400, 1333), (500, 1333), (600, 1333)],
                 keep_ratio=True),
            dict(type=RandomCrop,
                 crop_type='absolute_range',
                 crop_size=(384, 600),
                 allow_negative_crop=True),
            dict(type=RandomChoiceResize, resize_type=Resize,
                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                         (736, 1333), (768, 1333), (800, 1333)],
                 keep_ratio=True)
        ]]),
    dict(type=PackDetInputs)
]

from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import Resize

# Default test pipeline with shortest edge resize, see test_ms_* for multiscale testing!
# Source: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/configs/_base_/datasets/coco_instance.py#L38
test_pipeline_name = 'se_1280'  # used for wandb logging and run_name
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),  # update dynamically in train.py
    dict(type=Resize, scale=(2048, 1280), keep_ratio=True),
    dict(type=LoadAnnotations), # with_bbox=, with_mask= and with_segm= will be set in train.py as it depends on the model!
    dict(type=PackDetInputs,
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import Resize

# Fixed size testing pipeline as used by MaxVit and UVit
# NOTE: they don't provide code so we don't know exactly know what they did but likely similar to ViTDet
test_pipeline_name = 'fs_896'  # used for wandb logging and run_name
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),  # update dynamically in train.py
    dict(type=Resize, scale=(896, 896), keep_ratio=True),
    dict(type=LoadAnnotations), # with_bbox=, with_mask= and with_segm= will be set in train.py as it depends on the model!
    dict(type=PackDetInputs,
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

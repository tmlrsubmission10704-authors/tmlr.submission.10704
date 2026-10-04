from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import Resize

test_pipeline_name = 'test_model_complexity_1280x800'  # used for wandb logging and run_name
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),  # update dynamically in train.py
    dict(type=Resize, scale=(1280, 800), keep_ratio=False),
    dict(type=LoadAnnotations), # with_bbox=, with_mask= and with_segm= will be set in train.py as it depends on the model!
    dict(type=PackDetInputs,
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

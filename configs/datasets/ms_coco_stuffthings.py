from mmengine.config import read_base

# Some models like HTC require 'cocostuff' semantic segmentation for training!
#
# NOTE: These are not the stuff annotations from the coco website!
#
#       see: https://mmdetection.readthedocs.io/en/dev-3.x/user_guides/dataset_prepare.html
#       see: https://github.com/nightrome/cocostuff?tab=readme-ov-file#downloads
#
# NOTE: usually, 'seg' is only required for train_dataset but since we calculate loss on val_dataset,
#       we have to include this as well...
#
with read_base():
    from .ms_coco import *

train_dataset.update(data_prefix=dict(img='coco/train2017/', seg='coco/stuffthingmaps/train2017/'))
val_dataset.update(data_prefix=dict(img='coco/val2017/', seg='coco/stuffthingmaps/val2017/'))
# not required for train_val_dataset since this will be used for validation only, not training!

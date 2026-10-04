from mmengine.config import read_base

with read_base():
    from .ms_coco import *

dataset_name = 'coco-rem'
train_dataset.update(
    ann_file='coco-rem/annotations/instances_trainrem.json',
    data_prefix=dict(img='coco-rem/train2017/')
    )
val_dataset.update(
    ann_file='coco-rem/annotations/instances_valrem.json',
    data_prefix=dict(img='coco-rem/val2017/'),
)

train_val_dataset.update(
    ann_file='coco-rem/annotations/instances_trainrem_val.json',
    data_prefix=dict(img='coco-rem/train2017/'),
)

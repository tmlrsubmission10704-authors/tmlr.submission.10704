from mmengine.config import read_base

with read_base():
    from .ms_coco import *

dataset_name = 'sama-coco'
train_dataset.update(
    ann_file='sama-coco/annotations/instances_train2017.json',
    data_prefix=dict(img='sama-coco/train2017/')
    )
val_dataset.update(
    ann_file='sama-coco/annotations/instances_val2017.json',
    data_prefix=dict(img='sama-coco/val2017/'),
)
# we use this to calculate metrics on the training set
train_val_dataset.update(
    ann_file='sama-coco/annotations/instances_train2017_val.json',
    data_prefix=dict(img='sama-coco/train2017/')
)

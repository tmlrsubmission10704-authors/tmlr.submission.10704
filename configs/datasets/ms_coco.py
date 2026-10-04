from mmdet.datasets.coco import CocoDataset

dataset_type = 'CocoDataset'
dataset_name = 'ms-coco'
data_root = 'data/'
train_dataset=dict(
    type=CocoDataset,
    data_root=data_root,  # update dynamically in train.py
    ann_file='coco/annotations/instances_train2017.json',
    data_prefix=dict(img='coco/train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None,  # add dynamically in train.py
    backend_args=None)  # update dynamically in train.py

val_dataset=dict(
    type=CocoDataset,
    data_root=data_root,  # update dynamically in train.py
    ann_file='coco/annotations/instances_val2017.json',
    data_prefix=dict(img='coco/val2017/'),
    test_mode=True,
    pipeline=None,  # add dynamically in train.py
    backend_args=None)  # update dynamically in train.py

# we use this to calculate metrics on the training set
train_val_dataset=dict(
    type=CocoDataset,
    data_root=data_root,  # update dynamically in train.py
    ann_file='coco/annotations/instances_train2017_val.json',
    data_prefix=dict(img='coco/train2017/'),
    test_mode=True,
    pipeline=None,  # add dynamically in train.py
    backend_args=None)  # update dynamically in train.py

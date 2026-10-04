from mmengine.dataset.sampler import DefaultSampler
from mmdet.datasets.samplers.batch_sampler import AspectRatioBatchSampler
from mmdet_one_piece.engine import DefaultSampler_OP, worker_init_fn_OP

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(
        type=DefaultSampler_OP,
        shuffle=True,
        sampler_seed=0),  # update dynamically in train.py
    batch_sampler=dict(type=AspectRatioBatchSampler),
    worker_init_fn=dict(
        type=worker_init_fn_OP,
        data_seed=0),  # update dynamically in train.py
    dataset=None  # add dynamically in train.py
    )
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=None  # add dynamically in train.py
    )
test_dataloader = val_dataloader
train_val_dataloader = val_dataloader  # used to calculate metrics on the training set

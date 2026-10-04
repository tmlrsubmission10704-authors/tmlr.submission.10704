from mmengine.config import read_base

with read_base():
    from .atss_r50_fpn_dyhead import *

model_backbone_name = 'r50_sbn'  # short name used for wandb logging and run_name
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        norm_eval=False  # SyncBN!
    )
)

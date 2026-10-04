from mmengine.config import read_base

with read_base():
    from .queryinst_900p_r50_fpn import *

model_framework_name = 'sparse_rcnn_900p_5ds'
model_backbone_name = 'r50_sbn'  # short name used for wandb logging and run_name
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        out_indices=(1, 2, 3),
        norm_eval=False  # SyncBN!
    ),
    neck=dict(in_channels=[512, 1024, 2048], num_outs=5),
    roi_head=dict(
        bbox_roi_extractor=dict(featmap_strides=[8, 16, 32, 64, 128]),
        mask_roi_extractor=dict(featmap_strides=[8, 16, 32, 64, 128])
        )
)
# NOTE: we find queryinst with SBN, LSJ and Swin 3x schedule can hang without this
find_unused_parameters=True

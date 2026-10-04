from mmengine.config import read_base

with read_base():
    from .mask_rcnn_vitdet_s_sfp import *

model_backbone_name = 'vitdet_s_d3'  # short name used for wandb logging and run_name
model.update(
    backbone=dict(
        pretrain_use_cls_token=False,
        use_deit3_blocks=True
    )
)

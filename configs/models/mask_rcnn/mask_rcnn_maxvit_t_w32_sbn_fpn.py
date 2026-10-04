from mmengine.config import read_base

from mmdet_one_piece.models.backbones.timm_backbone import TIMMBackbone
# import not required, but useful for quick access via IDEs
from timm.models.maxxvit import maxvit_tiny_tf_224

with read_base():
    from .mask_rcnn_maxvit_t_w7_sbn_fpn import *

model_backbone_name = 'maxvit_t_w32_sbn'  # short name used for wandb logging and run_name
# NOTE: see mask_rcnn_maxvit_t_w7_sbn_fpn.py for more details
#       1024/32 -> window_size=32
model.update(
    data_preprocessor=dict(pad_size_divisor=1024),
    backbone=dict(img_size=1024)
)

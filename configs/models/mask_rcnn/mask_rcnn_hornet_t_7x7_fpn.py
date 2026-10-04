from mmengine.config import read_base

from mmdet_one_piece.models.backbones.hornet import HorNet
#from mmengine.model.weight_init import PretrainedInit
from mmdet_one_piece.models.backbones.hornet.layer_decay_optimizer_constructor import HorNet_LearningRateDecayOptimizerConstructorHorNet

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'hornet_t_7x7'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.hornet_t_7x7'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# Mask R-CNN configs but for 'small' model
# https://github.com/raoyongming/HorNet/blob/master/object_detection/configs/horfpn/fpn_mask_rcnn_hornet_small_7x7_1x_coco_in1k.py
# https://github.com/raoyongming/HorNet/blob/master/object_detection/configs/_base_/models/mask_rcnn_hornet_fpn.py
# https://github.com/raoyongming/HorNet/blob/master/object_detection/models/hornet.py#L164
# Cascade Mask R-CNN configs for 'tiny' model
# https://github.com/raoyongming/HorNet/blob/master/object_detection/configs/hornet/cascade_mask_rcnn_hornet_tiny_7x7_3x_coco_in1k.py
# https://github.com/raoyongming/HorNet/blob/master/object_detection/configs/hornet/cascade_mask_rcnn_hornet_tiny_gf_3x_coco_in1k.py
model.update(
    backbone=dict(
        type=HorNet,
        depths=[2, 3, 18, 2],
        base_dim=64,
        gnconv=[
            'partial(gnconv, order=2, s=1/3)',
            'partial(gnconv, order=3, s=1/3)',
            'partial(gnconv, order=4, s=1/3)',
            'partial(gnconv, order=5, s=1/3)',
        ],
        drop_path_rate=0.4,
        out_indices=[0, 1, 2, 3],
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.hornet_t_7x7.tc_in1k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint="...."))
    neck=dict(in_channels=[64, 128, 256, 512])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
# NOTE: we use cfg of mask-rcnn models here!
optim_wrapper = dict(
    constructor=HorNet_LearningRateDecayOptimizerConstructorHorNet,
    paramwise_cfg=dict(
        _delete_=True,
        decay_rate=0.95,
        decay_type='layer_wise',
        num_layers=6
    )
)

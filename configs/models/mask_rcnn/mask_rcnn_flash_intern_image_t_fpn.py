from mmengine.config import read_base

from mmdet_one_piece.models.backbones.flash_intern_image import FlashInternImage
#from mmengine.model.weight_init import PretrainedInit
from mmdet_one_piece.models.backbones.flash_intern_image import FPN_vitdet, FlashInternImage_CustomLayerDecayOptimizerConstructor

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'flash_intern_image_t'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.flash_intern_image_t'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/OpenGVLab/DCNv4/blob/main/detection/configs/coco/mask_rcnn_flash_intern_image_t_fpn_3x_coco.py
# https://github.com/OpenGVLab/DCNv4/blob/main/detection/mmdet_custom/models/backbones/flash_intern_image.py#L545
model.update(
    backbone=dict(
        type=FlashInternImage,
        core_op='DCNv4',
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.flash_intern_image_t.hf_in1k')),
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(type=FPN_vitdet, in_channels=[64, 128, 256, 512], norm_cfg=dict(type='LN', requires_grad=True))
    # NOTE: 1d LN only works here because of FPN_vitdet, check for details!!!
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    constructor=FlashInternImage_CustomLayerDecayOptimizerConstructor,
    paramwise_cfg=dict(
        _delete_=True,
        num_layers=30,
        layer_decay_rate=1.0,
        depths=[4, 4, 18, 4])
)

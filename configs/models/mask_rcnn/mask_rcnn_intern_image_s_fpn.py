from mmengine.config import read_base

from mmdet_one_piece.models.backbones.intern_image import InternImage
#from mmengine.model.weight_init import PretrainedInit
from mmdet_one_piece.models.backbones.intern_image import InternImage_CustomLayerDecayOptimizerConstructor

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'intern_image_s'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.intern_image_s'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/OpenGVLab/InternImage/blob/master/detection/configs/coco/mask_rcnn_internimage_s_fpn_3x_coco.py
# https://github.com/OpenGVLab/InternImage/blob/master/detection/mmdet_custom/models/backbones/intern_image.py#L532
model.update(
    backbone=dict(
        type=InternImage,
        core_op='DCNv3',
        channels=80,
        depths=[4, 4, 21, 4],
        groups=[5, 10, 20, 40],
        mlp_ratio=4.,
        drop_path_rate=0.3,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.intern_image_s.hf_in1k')),
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(in_channels=[80, 160, 320, 640])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    constructor=InternImage_CustomLayerDecayOptimizerConstructor,
    paramwise_cfg=dict(
        _delete_=True,
        num_layers=33,
        layer_decay_rate=1.0,
        depths=[4, 4, 21, 4])
)

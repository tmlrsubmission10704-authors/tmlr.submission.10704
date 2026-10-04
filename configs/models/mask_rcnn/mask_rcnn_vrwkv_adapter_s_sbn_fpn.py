from mmengine.config import read_base

from mmdet_one_piece.models.backbones.vision_rwkv import VRWKV_Adapter
from mmdet_one_piece.models.backbones.vision_rwkv.layer_decay_optimizer_constructor import VRWKV_LayerDecayOptimizerConstructor

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'vrwkv_adapter_s_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.vrwkv_adapter_s'
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/OpenGVLab/Vision-RWKV/blob/master/detection/configs/_base_/models/mask_rcnn_r50_fpn.py

#pretrained = 'pretrained/vrwkv_s_in1k_224.pth'
# https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_s_in1k_224.pth

model.update(
    backbone=dict(
        type=VRWKV_Adapter,
        img_size=224,
        patch_size=16,
        embed_dims=384,
        depth=12,
        #pretrained=pretrained,
        init_values=1e-5,
        post_norm=True,
        with_cp=False,
        # adapter param
        drop_path_rate=0.2,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.vrwkv_adapter_s.ogvl_in1k', prefix='backbone.')),
    neck=dict(
        in_channels=[384, 384, 384, 384])
)

optim_wrapper = dict(
    constructor=VRWKV_LayerDecayOptimizerConstructor,
    paramwise_cfg=dict(
        _delete_=True,
        num_layers=12,
        layer_decay_rate=0.85)
)

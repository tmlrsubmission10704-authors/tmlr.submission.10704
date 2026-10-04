from mmengine.config import read_base

from mmdet_one_piece.models.backbones.transnext import TransNeXt
#from mmengine.model.weight_init import PretrainedInit
import torch.nn as nn

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'transnext_s_o'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.transnext_s'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/DaiShiResearch/TransNeXt/blob/main/detection/maskrcnn/configs/mask_rcnn_transnext_small_fpn_1x_coco.py
# https://github.com/DaiShiResearch/TransNeXt/blob/main/detection/maskrcnn/configs/_base_/models/mask_rcnn_transnext_fpn.py
# https://github.com/DaiShiResearch/TransNeXt/blob/main/detection/maskrcnn/transnext_native.py#L494
# https://github.com/DaiShiResearch/TransNeXt/blob/main/detection/maskrcnn/transnext_cuda.py#L535
model.update(
    backbone=dict(
        type= TransNeXt,
        #type='transnext_small',  # we use put all args here instead of calling transnext_small()
        window_size=[3, 3, 3, None],
        patch_size=4,
        embed_dims=[72, 144, 288, 576],
        num_heads=[3, 6, 12, 24],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=dict(type=nn.LayerNorm, eps=1e-6),  # instead of partial(nn.LayerNorm, eps=1e-6)
        depths=[5, 5, 22, 5],
        sr_ratios=[8, 4, 2, 1],  # original
        #sr_ratios=[16, 8, 4, 1],  # way faster with small performance hit
        drop_rate=0.0,
        drop_path_rate=0.6,
        img_size=800,
        pretrain_size=224,
        is_extrapolation=True,  # faster
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.transnext_s.tx_in1k')),
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(in_channels=[72, 144, 288, 576])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            query_embedding=dict(decay_mult=0.),
            relative_pos_bias_local=dict(decay_mult=0.),
            cpb=dict(decay_mult=0.),
            temperature=dict(decay_mult=0.),
            norm=dict(decay_mult=0.))
    )
)

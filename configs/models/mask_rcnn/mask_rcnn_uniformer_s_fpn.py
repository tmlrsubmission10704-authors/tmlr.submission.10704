from mmengine.config import read_base

from torch import nn
from mmdet_one_piece.models.backbones.uniformer import UniFormer
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'uniformer_s'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.uniformer_s'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/Sense-X/UniFormer/blob/main/object_detection/mmdet/models/backbones/uniformer.py#L248
# https://github.com/Sense-X/UniFormer/blob/main/object_detection/configs/_base_/models/mask_rcnn_uniformer_fpn.py
# https://github.com/Sense-X/UniFormer/blob/main/object_detection/exp/mask_rcnn_3x_ms_hybrid_small/config.py
model.update(
    backbone=dict(
        type=UniFormer,
        layers=[3, 4, 8, 3],
        num_classes=80,
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,             # small model cfg
        norm_layer=dict(type=nn.LayerNorm, eps=1e-6),
        #use_checkpoint=True,           # small model cfg, enable when OOM
        #checkpoint_num=[0, 0, 8, 0],   # small model cfg, enable when OOM
        windows=False,
        hybrid=True,                    # small model cfg
        window_size=14,
        batch_norm_eval=True,           # semi-frozen BN, similar to ResNet Default in mmdet
        batch_norm_grad=True,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.uniformer_s.uf_in1k')),
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(in_channels=[64, 128, 320, 512])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(                                       # the original cfg had these so we keep them, but:
            absolute_pos_embed=dict(decay_mult=0.),             # the model has no absolute_pos_embed keys
            relative_position_bias_table=dict(decay_mult=0.),   # the model has no relative biases table keys
            norm=dict(decay_mult=0.))
    )
)

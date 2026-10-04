from mmengine.config import read_base

from torch import nn
from mmdet_one_piece.models.backbones.biformer import BiFormer
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'biformer_s'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.biformer_s'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/rayleizhu/BiFormer/blob/public_release/object_detection/configs/coco/maskrcnn.1x.biformer_small.py
# https://github.com/rayleizhu/BiFormer/blob/public_release/object_detection/models_mm/biformer_mm.py#L12
# https://github.com/rayleizhu/BiFormer/blob/public_release/models/biformer.py#L136
model.update(
    backbone=dict(
        type=BiFormer,
        num_classes=80,
        depth=[4, 4, 18, 4],
        embed_dim=[64, 128, 256, 512],
        mlp_ratios=[3, 3, 3, 3],
        n_win=16,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False,
        diff_routing=False,
        soft_routing=False,
        pre_norm=True,
        pe=None,
        auto_pad=True,
        #use_checkpoint_stages=[],  # they use a third party lib which we don't support!
        drop_path_rate=0.2,
        norm_eval=True,             # semi-frozen BN, similar to ResNet Default in mmdet
        disable_bn_grad=False,
        use_fused_sdpa=True,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.biformer_s.bi_in1k')),
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(in_channels=[64, 128, 256, 512])
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

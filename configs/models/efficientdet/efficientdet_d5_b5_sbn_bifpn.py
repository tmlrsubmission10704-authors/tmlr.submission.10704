from mmengine.config import read_base
from mmdet_one_piece.models.backbones.timm_backbone import TIMMBackbone
from mmengine.hooks.ema_hook import EMAHook
from mmdet.models.layers.ema import ExpMomentumEMA

with read_base():
    from ._efficientdet_d5 import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA, norm_cfg

model_backbone_name = 'b5_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'timm.tf_efficientnet_b5'
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        type=TIMMBackbone,
        model_name='tf_efficientnet_b5',
        drop_rate=0.0,
        drop_path_rate=0.3,
        bn_momentum=0.01,  # from efficient det norm_cfg
        bn_eps=1e-3,       # from efficient det norm_cfg
        frozen_stages=-1,
        freeze_stages_fn = None,
        feature_out_norm_cfg=dict(type='LN2d', eps=1e-6),  # TODO: eps similar to backbone?
        out_indices=(2, 3, 4),
        init_cfg=dict(resolve='timm.tf_efficientnet_b5.aa_in1k')
    ),
    neck=dict(in_channels=[64, 176, 512])
)

# NOTE: they use batch_size 128 (16 per GPU), SGD and EMA training, see original cfg for details
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        #custom_keys=dict(norm=dict(decay_mult=0.), bn=dict(decay_mult=0.),
    ),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# This will be appended to custom_hooks in train.py (we have a bad check in train.py so it needs to be appended...)
# TODO: _custom_hooks -> custom_hooks after check in train.py is resolved
_custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=ExpMomentumEMA,
        momentum=0.0002,
        update_buffers=True,
        priority='NORMAL')
]

from mmengine.config import read_base

from mmdet_one_piece.models.backbones.timm_backbone import TIMMBackbone
# import not required, but useful for quick access via IDEs
from timm.models.maxxvit import maxvit_tiny_tf_224

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'maxvit_t_w7_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'timm.maxvit_t_tf_224'
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# NOTE: it's not entirely clear how MaxVit models are configured for detection
#
#       From https://arxiv.org/pdf/2204.01697 Appendix, B.2 we know that the authors used:
#       batch_size 256, AdamW and 1e-3 lr, 0.8 drop_path_rate for MaxVit-Tiny
#
#       Regarding augmentations, they report Multiscale Training with inputs resized to 896x896
#       We are not 100% sure what they did there or where this choice came from but we highly
#       suspect the parallel UViT work, see: https://arxiv.org/abs/2112.09747
#
#       The metaformer paper also includes some comparisons to MaxVit, see:
#       https://arxiv.org/pdf/2210.13452
#
# NOTE: The input must always be > and divisible by img_size!
#       Maxvit has a max. downscaling factor of 32 and by default, uses img_size=224
#       ->  224/32 -> window_size=7   (pre-trained checkpoints)
#       ->  896/32 -> window_size=28  (used for detection)
#       -> 1024/32 -> window_size=32
#
#       Since 896/224=4, this size could also be used as input without changing the window size!
#
model.update(
    data_preprocessor=dict(pad_size_divisor=224),  # https://github.com/google-research/maxvit/issues/10
    backbone=dict(
        type=TIMMBackbone,
        model_name='maxvit_tiny_tf_224',
        img_size=224,
        drop_path_rate=0.1,     # we use smaller dpr for smaller batch sizes! e.g. bs16
        batch_norm_eval=False,
        batch_norm_grad=True,
        frozen_stages=-1,
        freeze_stages_fn = None,
        feature_out_norm_cfg=dict(type='LN2d', eps=1e-6),
        out_indices=(1, 2, 3, 4),
        use_fused_sdpa=True,
        init_cfg=dict(resolve='timm.maxvit_t_tf_224.tf_in1k')),
    neck=dict(in_channels=[64, 128, 256, 512])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            relative_position_bias_table=dict(decay_mult=0.),
            norm=dict(decay_mult=0.))
    )
)

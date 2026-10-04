from mmengine.config import read_base

# import not required, but useful for quick access via IDEs
#from timm.models.hieradet_sam2 import sam2_hiera_tiny
from mmdet_one_piece.models.backbones.hieradet import hieradet_tiny

from mmdet_one_piece.models.backbones.timm_backbone import TIMMBackbone

from mmdet_one_piece.engine.optimizer import LearningRateDecayOptimizerConstructor_HieraDet

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'hieradet_t'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'timm.hieradet_t'
sync_bn = False
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# NOTE: HieraDet was introduced in https://arxiv.org/pdf/2311.05613 not in Sam2 paper.
# Appendix A3 includes some details on detection training but not for T or S models.
#
#   - drop_path 0.2 (B)
#   - layer-wise decay 0.85 (B)
#   - we found dp 0.1 and lw-decay 0.7 work well for the tiny model size
#
model.update(
    #data_preprocessor=dict(pad_size_divisor=1024),
    backbone=dict(
        type=TIMMBackbone,
        #model_name='sam2_hiera_tiny',
        model_name='hieradet_tiny',
        drop_path_rate=0.1,
        #init_values=1e-5,  # unused so far, but timm has this in it's hieradet_small cfg (untrained)
        frozen_stages=-1,
        freeze_stages_fn = None,
        feature_out_norm_cfg=dict(type='LN2d', eps=1e-6),
        out_indices=(0, 1, 2, 3),
        recursive_absolut_win=True,
        input_size=(1024, 1024),
        global_rel_pos=True,
        use_fused_sdpa=True,
        init_cfg=dict(resolve='timm.hieradet_t.fb_mae_in1k')),
    neck=dict(in_channels=[96, 192, 384, 768])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    constructor=LearningRateDecayOptimizerConstructor_HieraDet,
    paramwise_cfg=dict(
        _delete_=True,
        #num_layers=12, this is implemented dynamically in the constructor!
        decay_rate=0.7,
        decay_type='layer_wise',
        no_wd_list=['pos_embed', 'pos_embed_window', 'norm', 'rel_pos_h', 'rel_pos_w'],
        #
        # NOTE: with layer_wise, decay_mult=0. will be handled by the constructor using
        #       no_wd_list instead of custom_keys. Without custom LearningRateDecayOptimizerConstructor, you can
        #       use the following cfg to do the same:
        #
        #custom_keys=dict(
        #    pos_embed=dict(decay_mult=0.),
        #    pos_embed_window=dict(decay_mult=0.),
        #    rel_pos_h=dict(decay_mult=0.0),
        #    rel_pos_w=dict(decay_mult=0.0),
        #    norm=dict(decay_mult=0.))
    )
)
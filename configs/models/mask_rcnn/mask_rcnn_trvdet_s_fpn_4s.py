from mmengine.config import read_base

from torch import nn
from mmdet_one_piece.models.backbones.transform_vision import TrVDet
from mmdet_one_piece.models.necks.feature_pyramids import FeatureMapResamplingWrapper
from mmdet_one_piece.engine.optimizer import LearningRateDecayOptimizerConstructor_TrVDet
from mmdet.models.necks.fpn import FPN

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_neck_name = 'fpn_4s'  # SimpleFeaturePyramid
model_backbone_name = 'trvdet_s'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.trvdet_s'
sync_bn = False
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config

# Config source, different or same as VitDet
#
#  NEW: https://github.com/baaivision/EVA/blob/master/EVA-02/det/detectron2/modeling/backbone/vit.py#L290
# same: https://github.com/baaivision/EVA/blob/master/EVA-02/det/configs/common/models/mask_rcnn_vitdet.py
# same: https://github.com/baaivision/EVA/blob/master/EVA-02/det/projects/ViTDet/configs/eva2_mim_to_coco/mask_rcnn_vitdet_b_100ep.py
# same: https://github.com/baaivision/EVA/blob/master/EVA-02/det/projects/ViTDet/configs/eva2_mim_to_coco/cascade_mask_rcnn_vitdet_b_100ep.py
#  NEW: https://github.com/baaivision/EVA/blob/master/EVA-02/det/projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7.py
#       https://github.com/baaivision/EVA/blob/5623d08bb08a8268c7bc18e1d9b8b4bd651fe4d9/EVA-02/asuka/modeling_pretrain.py#L272
#
# NOTE: EVA02 did not train a small model for detection:
#
#       Since they follow the VitDet settings, we use the same adaptation and defaults as we do for ViTDet!
#       (e.g. no layer-wise lr decay, ...)
#
# NOTE: Rope only has buffers, any errors when loading a checkpoint can be ignored!
#
model.update(
    backbone=dict(
        type=TrVDet,
        # vitdet cfg
        img_size=1024,
        patch_size=16,       # NOTE: the pre-trained checkpoints use patch_size=14 and are rescaled to 16 for detection!
        embed_dim=384,       # small cfg, similar to deit
        depth=12,
        num_heads=6,         # small cfg, similar to deit
        drop_path_rate=0.1,
        #window_size=14,     # different to VitDet - see below
        #mlp_ratio=4,        # different to VitDet - see below
        qkv_bias=True,
        norm_layer=dict(type=nn.LayerNorm, eps=1e-6),
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],  # 2, 5, 8 11 are global attention
        residual_block_indexes=[],
        use_rel_pos=True,    # NOTE: set but unused in TrVDet, they use rope - see below
        # trvdet cfg (only if new or different in vitdet cfg)
        window_size=16,
        mlp_ratio=4*2/3,
        # trvdet constructor args (only if missing in vitdet and trvdet cfg)
        use_abs_pos=True,
        rope=True,             # different to VitDet NOTE: arg not used in code (will always use rope no matter what)
        pt_hw_seq_len=16,      # different to VitDet
        intp_freq=True,        # different to VitDet
        use_fused_sdpa=True,   # NOTE: they used xattn (xformers)
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        # our cfg (new, custom to our implementation)
        subln=False,
        interpolate_14to16=True,
        unpack_swiglu_weights=True,
        pop_rope_buffers=True,
        out_indices=(2, 5, 8, 11),
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.trvdet_s.ba_eva02_mae_in21k'))
)
# hard overwrite of any previous neck cfg
model.neck=dict(
        type=FeatureMapResamplingWrapper,
        backbone_channels = [384, 384, 384, 384],  # '4-stages' resampling
        scale_factors=[4.0, 2.0, 1.0, 0.5],        # strides [ 16,  16,  16,  16] -> [    4,     8,  16,  32]
        scale_dims=True,                           # dims    [384, 384, 384, 384] -> [384/4, 384/2, 384, 384]
        out_norm_cfg=dict(type='LN2d', eps=1e-6),  # NEW, was not used in original paper
        neck=dict(
            type=FPN,
            in_channels=[96, 192, 384, 384],
            out_channels=256,
            num_outs=5,
        )
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    #constructor=LearningRateDecayOptimizerConstructor_TrVDet,
    paramwise_cfg=dict(
        _delete_=True,
        #num_layers=12,
        #decay_rate=0.7,
        #decay_type='layer_wise',
        #no_wd_list=['pos_embed', 'norm'],
        #
        # NOTE: with layer_wise, decay_mult=0. will be handled by the constructor using no_wd_list instead of
        #       custom_keys. Without custom LearningRateDecayOptimizerConstructor, you can use the following cfg to
        #       do the same:
        #
        custom_keys=dict(
            pos_embed=dict(decay_mult=0.),
            norm=dict(decay_mult=0.))
    )
)

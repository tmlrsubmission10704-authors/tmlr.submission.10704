from mmengine.config import read_base

from mmdet_one_piece.models.backbones.vim import VisionMambaDet
from mmdet_one_piece.models.necks.feature_pyramids import FeatureMapResamplingWrapper
from mmdet_one_piece.engine.optimizer import LearningRateDecayOptimizerConstructor_Vim
from mmdet.models.necks.fpn import FPN

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_neck_name = 'fpn_4s'  # SimpleFeaturePyramid
model_backbone_name = 'vim_s'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.vim_s'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/hustvl/Vim/blob/main/det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_s_100ep.py
# https://github.com/hustvl/Vim/blob/main/det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_b_100ep.py
# https://github.com/hustvl/Vim/blob/main/det/projects/ViTDet/configs/COCO/mask_rcnn_vimdet_b_100ep.py
# https://github.com/hustvl/Vim/blob/main/det/configs/common/models/mask_rcnn_vimdet.py
# https://github.com/hustvl/Vim/blob/main/det/detectron2/modeling/backbone/vim.py#L34
# https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py#L229
#
# NOTE: we found layer-wise learning rate decay to perform worse
#
model.update(
    backbone=dict(
        type=VisionMambaDet,
        img_size=1024,
        patch_size=16,
        embed_dim=384,  # s model
        depth=24,       # s model
        drop_path_rate=0.1,
        out_indices=(5, 11, 17, 23),  # 23 for last_map, for 4-stages use (5, 11, 17, 23)
        last_layer_process="add",
        bimamba_type="v2",
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        if_abs_pos_embed=True,
        if_rope=True,
        if_rope_residual=True,
        pt_hw_seq_len=14,
        if_cls_token=True,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.vim_s.vl_in1k'))
)

# hard overwrite of any previous neck cfg
model.neck=dict(
        type=FeatureMapResamplingWrapper,
        backbone_channels = [384, 384, 384, 384], # '4-stages' resampling
        scale_factors=[4.0, 2.0, 1.0, 0.5],       # strides [ 16,  16,  16,  16] -> [    4,     8,  16,  32]
        scale_dims=True,                          # dims    [384, 384, 384, 384] -> [384/4, 384/2, 384, 384]
        out_norm_cfg=dict(type='LN2d', eps=1e-6),
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
    #constructor=LearningRateDecayOptimizerConstructor_Vim,
    paramwise_cfg=dict(
        _delete_=True,
        #num_layers=24,
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

from mmengine.config import read_base

from mmdet_one_piece.models.backbones.vision_lstm import VisionLSTM2
from mmdet_one_piece.models.necks.feature_pyramids import FeatureMapResamplingWrapper, _SimpleFeaturePyramid
from mmdet_one_piece.engine.optimizer import LearningRateDecayOptimizerConstructor_ViLDet

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_neck_name = 'sfp'  # SimpleFeaturePyramid
model_backbone_name = 'vildet_s'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.vil2_s'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config

# NOTE: the authors don't provide detection code or cfgs, we choose some reasonable defaults
model.update(
    backbone=dict(
        type=VisionLSTM2,
        dim=384,
        input_shape=(3, 1024, 1024),  # set this close to input, our implementation can handle dynamics shape though!
        patch_size=16,
        depth=12,
        drop_path_rate=0.1,
        drop_path_decay=True,
        proj_bias=True,
        norm_bias=True,
        out_indices = (11,),  # in sem-seg, they use 3, 5, 7 and 11 but we do simple feature pyramid!
        global_blocks = [3, 5, 7, 11],
        window_size = 14,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.vil2_s.nx_in1k'))
)
# hard overwrite of any previous neck cfg
model.neck=dict(
        type=FeatureMapResamplingWrapper,
        backbone_channels = 384,             # 'last_map' resampling
        scale_factors=[4.0, 2.0, 1.0, 0.5],  # strides [ 16,  16,  16,  16] -> [    4,     8,  16,  32]
        scale_dims=True,                     # dims    [384, 384, 384, 384] -> [384/4, 384/2, 384, 384]
        out_norm_cfg=dict(type='LN2d', eps=1e-6),
        neck=dict(
            type=_SimpleFeaturePyramid,
            in_channels=[96, 192, 384, 384],
            out_channels=256,
            num_outs=5,
        )
)

# We set norm and pos decay to 0, similar to other modern models
# NOTE: by default, we don't use layer-wise learning rate decay to be comparable with our ViTDet config
optim_wrapper = dict(
    #constructor=LearningRateDecayOptimizerConstructor_ViLDet,
    paramwise_cfg=dict(
        _delete_=True,
        # num_layers=12,
        # decay_rate=0.7,
        # decay_type='layer_wise',
        # no_wd_list=['pos_embed', 'norm']
        # NOTE: with layer_wise, decay_mult=0. will be handled by the constructor using no_wd_list instead of
        #       custom_keys. Without custom LearningRateDecayOptimizerConstructor, you can use the following cfg to
        #       do the same:
        #
        custom_keys=dict(
            pos_embed=dict(decay_mult=0.),
            norm=dict(decay_mult=0.))
    )
)

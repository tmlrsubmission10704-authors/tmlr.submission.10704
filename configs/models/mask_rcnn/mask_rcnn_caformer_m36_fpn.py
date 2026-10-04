from mmengine.config import read_base

from mmdet_one_piece.models.backbones.metaformer_baselines import caformer_m36
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'caformer_m36'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.caformer_m36'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# NOTE: They don't provide a cfg

model.update(
    backbone=dict(
        type=caformer_m36,
        drop_path_rate=0.2,
        frozen_stages=-1,
        use_fused_sdpa=True,
        init_cfg=dict(resolve='mmdet.caformer_m36.sa_in1k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint="...."))
    neck=dict(in_channels=[96, 192, 384, 576]),
    roi_head=dict(
        bbox_roi_extractor=dict(roi_layer=dict(use_torchvision=True)),
        mask_roi_extractor=dict(roi_layer=dict(use_torchvision=True))
    )
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            norm=dict(decay_mult=0.))  # They don't have this in their cfg but we added it
    )
)

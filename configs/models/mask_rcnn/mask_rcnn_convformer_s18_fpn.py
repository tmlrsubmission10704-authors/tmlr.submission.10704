from mmengine.config import read_base

from mmdet_one_piece.models.backbones.metaformer_baselines import convformer_s18
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'convformer_s18'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.convformer_s18'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# NOTE: They don't provide a cfg

model.update(
    backbone=dict(
        type=convformer_s18,
        drop_path_rate=0.1,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.convformer_s18.sa_in1k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint="...."))
    neck=dict(in_channels=[64, 128, 320, 512]),
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

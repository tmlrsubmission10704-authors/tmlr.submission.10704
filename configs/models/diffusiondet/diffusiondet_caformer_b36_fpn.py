from mmengine.config import read_base

from mmdet_one_piece.models.backbones.metaformer_baselines import caformer_b36
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._diffusiondet_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'caformer_b36'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.caformer_b36'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
model.update(
    backbone=dict(
        type=caformer_b36,
        drop_path_rate=0.3,
        frozen_stages=-1,
        use_fused_sdpa=True,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(resolve='mmdet.caformer_b36.sa_in1k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint="...."))
    neck=dict(in_channels=[128, 256, 512, 768])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
# https://github.com/open-mmlab/mmdetection/blob/main/projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py#L159
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

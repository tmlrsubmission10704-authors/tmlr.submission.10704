from mmengine.config import read_base

from mmdet_one_piece.models.backbones.ms_vmamba import ms_vmamba

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'ms_vssm_t'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.ms_vssm_t'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/YuHengsss/MSVMamba/blob/master/detection/configs/ms_vssm/mask_rcnn_ms_vssm_fpn_coco_tiny_3x.py
# https://github.com/YuHengsss/MSVMamba/blob/v2/classification/configs/msv2/msvmambav3_tiny_224.yaml
# see also: https://github.com/YuHengsss/MSVMamba/blob/v2/classification/models/__init__.py#L19
model.update(
    backbone=dict(
        type=ms_vmamba,
        out_indices=(0, 1, 2, 3),
        dims=96,
        depths=(2, 2, 9, 2),      # from msv2 cfg
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=1.0,            # from msv2 cfg
        mlp_ratio=4.0,
        downsample_version="v3",  # from msv2 cfg
        patchembed_version="v2",  # from msv2 cfg
        forward_type="vms_noz",   # from msv2 cfg
        norm_layer="ln2d",        # from msv2 cfg
        conv_ffn_ratio=4,         # from msv2 cfg
        sscore_type="multiscale_4scan_12",
        convffn=True,             # NOTE: lower case ffn for v2!
        add_se=True,
        ms_stage=[0, 1, 2, 3],
        ms_split=[1, 3],
        ffn_dropout=0.0,
        drop_path_rate=0.2,       # from msv2 cfg
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.ms_vssm_t.yh_in1k')),
    neck=dict(in_channels=[96, 192, 384, 768])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            norm=dict(decay_mult=0.))
    )
)

from mmengine.config import read_base

from mmdet_one_piece.models.backbones.slak import SLaK
#from mmengine.model.weight_init import PretrainedInit
from mmdet_one_piece.models.backbones.slak import SLaK_LearningRateDecayOptimizerConstructor

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA
model_backbone_name = 'slack_t_p4_w7_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.slack_t_p4_w7'
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# NOTE: They only provide configs for Cascade Mask RCNN, so we did check what ConvNeXt does for Mask RCNN
#       and use these settings instead, e.g. the paramwise_cfg for the optimizer!
# see: https://github.com/open-mmlab/mmdetection/blob/main/configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py
# https://github.com/VITA-Group/SLaK/blob/SLAKandLargeKernelKD/detection/configs/cascade_mask_rcnn_slak_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py
# https://github.com/VITA-Group/SLaK/blob/SLAKandLargeKernelKD/detection/configs/_base_/models/cascade_mask_rcnn_slak_fpn.py
# https://github.com/VITA-Group/SLaK/blob/SLAKandLargeKernelKD/detection/slak.py#L178
model.update(
    backbone=dict(
        type=SLaK,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        kernel_size=[51, 49, 47, 13, 5],
        LoRA=True,
        width_factor=1.3,
        sparse=True,
        use_cuda_kernel=False,  # We found their custom cuda kernel to be very slow so we use Conv2D instead
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.slack_t_p4_w7.vg_in1k')),
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone.')),
    neck=dict(in_channels=[124, 249, 499, 998])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    constructor=SLaK_LearningRateDecayOptimizerConstructor,
    paramwise_cfg=dict(
        _delete_=True,
        decay_rate=0.95,
        decay_type='layer_wise',
        num_layers=6)
)

from mmengine.config import read_base

from mmdet_one_piece.models.backbones.timm_backbone import TIMMBackbone
# import not required, but useful for quick access via IDEs
from mmdet_one_piece.models.backbones.robust_resnet import robust_resnet_up_inverted_dw_small
from mmdet_one_piece.models.backbones.robust_resnet import _freeze_stages_RobustResNet

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'robust_resnet_uidw_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'timm.robust_resnet_uidw'
sync_bn = 'torch'
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# NOTE: the authors don't provide detection code or cfgs, we re-use what can be found in the repo
# https://github.com/UCSC-VLAA/RobustCNN
# https://github.com/UCSC-VLAA/RobustCNN/blob/main/TRAIN.md
# https://github.com/UCSC-VLAA/RobustCNN/blob/main/train.py#L56
# https://github.com/UCSC-VLAA/RobustCNN/blob/main/timm/models/robust_resnet.py#L224
# https://github.com/UCSC-VLAA/RobustCNN/blob/main/timm/models/robust_resnet.py#L117
# NOTE: we set kwargs similar to RobustCNN authors
# --drop-path 0.1
#       see: https://github.com/UCSC-VLAA/RobustCNN/blob/main/TRAIN.md
# --layer_scale_init_value 0 (otherwise checkpoints can't be loaded in strict mode)
#       see: https://github.com/UCSC-VLAA/RobustCNN/blob/81b433682dc4ee5a95a03a014a9d76e6502bb611/train.py#L229

model.update(
    backbone=dict(
        type=TIMMBackbone,
        model_name='robust_resnet_up_inverted_dw_small',
        batch_norm_eval=False,
        batch_norm_grad=True,
        frozen_stages=-1,
        freeze_stages_fn = _freeze_stages_RobustResNet,
        feature_out_norm_cfg=dict(type='LN2d', eps=1e-6),
        init_cfg=dict(resolve='timm.robust_resnet_uidw.uv_in1k'),
        # robust_resnet kwargs from here:
        feature_pyramid=False,
        drop_path_rate=0.1,
        layer_scale_init_value=0.0,
        ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    rpn_head=dict(
        anchor_generator=dict(strides=[16, 16, 16, 32, 64], base_sizes=[4, 8, 16, 32, 64]),
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(featmap_strides=[16, 16, 16, 32,]),
        mask_roi_extractor=dict(featmap_strides=[16, 16, 16, 32,])
    )
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            norm=dict(decay_mult=0.))
    )
)

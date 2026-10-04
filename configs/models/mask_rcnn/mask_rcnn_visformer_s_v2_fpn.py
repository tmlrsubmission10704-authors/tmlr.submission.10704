from mmengine.config import read_base

from mmdet_one_piece.models.backbones.swin_visformer import SwinVisformer
from mmdet_one_piece.models.backbones.swin_visformer import BatchNorm
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'visformer_s_v2'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.visformer_s_v2'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/danczs/Swin-Visformer-Object-Detection/blob/master/configs/swin_visformer/mask_rcnn_swin_visformer_small_v2_mstrain_480-800_adamw_3x_coco.py
# https://github.com/danczs/Swin-Visformer-Object-Detection/blob/master/configs/_base_/models/mask_rcnn_swin_visformer_fpn.py
# https://github.com/danczs/Swin-Visformer-Object-Detection/blob/master/mmdet/models/backbones/swin_visformer.py#L292
# same files in cls project: https://github.com/danczs/Visformer/tree/main/ObjectDetection
model.update(
    backbone=dict(
        type=SwinVisformer,
        # s_v2 config
        embed_dim=256,
        depth=[1, 10, 14, 3],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.2,
        use_checkpoint=False,
        # base config
        init_channels=32,
        mlp_ratio=4.,
        group=8,
        attn_stage='0011',
        spatial_conv='1100',
        conv_init=True,
        qkv_bias=False,
        qk_scale=-0.5,
        drop_rate=0.,
        attn_drop_rate=0.,
        num_classes=80,
        out_indices=(0, 1, 2, 3),
        # remaining default args
        norm_layer=BatchNorm,
        pool=True,
        embedding_norm=BatchNorm,
        frozen_stages=-1,
        # new args
        batch_norm_eval=False,  # we tried semi-frozen BN similar to ResNet Default in mmdet but it diverges...
        batch_norm_grad=True,
        init_cfg=dict(resolve='mmdet.visformer_s_v2.vf_in1k')),
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(in_channels=[64, 128, 256, 512]),
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
            absolute_pos_embed=dict(decay_mult=0.),
            relative_position_bias_table=dict(decay_mult=0.),
            norm=dict(decay_mult=0.))
    )
)

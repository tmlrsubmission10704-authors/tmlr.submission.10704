from mmengine.config import read_base

from mmdet_one_piece.models.backbones.pvt import PyramidVisionTransformer
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'pvt_s'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.pvt_s'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/open-mmlab/mmdetection/blob/main/configs/pvt/retinanet_pvt-s_fpn_1x_coco.py
# https://github.com/whai362/PVT/blob/v2/detection/configs/mask_rcnn_pvt_s_fpn_1x_coco.py
# https://github.com/whai362/PVT/blob/57e2dfaa5a46f9050d76f306a4fcd9a7c061f520/detection/pvt.py#L245
model.update(
    backbone=dict(
        type=PyramidVisionTransformer,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[4, 2, 2, 2],
        strides=[4, 2, 2, 2],
        paddings=[0, 0, 0, 0],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_embed=True,
        norm_after_stage=False,
        use_conv_ffn=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', eps=1e-6),
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.pvt_s.pvt_in1k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint="...."))
    neck=dict(in_channels=[64, 128, 320, 512])
)

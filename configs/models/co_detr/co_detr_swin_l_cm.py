from mmengine.config import read_base

from torch.nn import BatchNorm2d
from mmdet.models.backbones.swin import SwinTransformer
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._co_detr_cm import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'swin_l_p4_w12_384'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.swin_l_p4_w12_384'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config

model.update(
    backbone=dict(
        type=SwinTransformer,
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.swin_l_p4_w12_384.ms_in22k')),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6)))
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
# https://github.com/open-mmlab/mmdetection/blob/main/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py#L325
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(backbone=dict(lr_mult=0.1))
        ),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

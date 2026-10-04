from mmengine.config import read_base

from mmdet_one_piece.models.backbones.vit_adapter import ViTAdapter
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'deit_adapter_s_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.deit_adapter_s'
sync_bn = 'torch'  # NOTE: ViTAdapter and the SpatialPriorModule use SyncBatchNorm by default
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/czczup/ViT-Adapter/blob/main/detection/configs/mask_rcnn/mask_rcnn_deit_adapter_small_fpn_3x_coco.py
# ViTAdapter: https://github.com/czczup/ViT-Adapter/blob/main/detection/mmdet_custom/models/backbones/vit_adapter.py#L20
# ViT: https://github.com/czczup/ViT-Adapter/blob/main/detection/mmdet_custom/models/backbones/base/vit.py#L354
model.update(
    backbone=dict(
        type=ViTAdapter,
        patch_size=16,      # ViT arg
        embed_dim=384,      # ViT arg
        depth=12,           # ViT arg
        num_heads=6,        # ViT arg but also used in ViTAdapter
        mlp_ratio=4,        # ViT arg
        drop_path_rate=0.2, # ViT arg
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[True, True, False, True, True, False,      # ViT arg
                     True, True, False, True, True, False],
        window_size=[14, 14, None, 14, 14, None,                # ViT arg
                     14, 14, None, 14, 14, None],
        #pretrained=pretrained,                                 # ViT arg (we use init_cfg instead)
        frozen_stages=-1,
        use_fused_sdpa=True,
        init_cfg=dict(resolve='mmdet.deit_adapter_s.fb_in1k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint="...."))
    neck=dict(in_channels=[384, 384, 384, 384])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(
            level_embed=dict(decay_mult=0.),
            pos_embed=dict(decay_mult=0.),
            norm=dict(decay_mult=0.),
            bias=dict(decay_mult=0.))
    )
)

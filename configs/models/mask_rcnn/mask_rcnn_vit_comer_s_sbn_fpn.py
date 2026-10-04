from mmengine.config import read_base

from mmdet_one_piece.models.backbones.vit_comer import ViTCoMer
#from mmengine.model.weight_init import PretrainedInit
from mmdet_one_piece.models.backbones.vit_comer.layer_decay_optimizer_constructor import Vit_Comer_LayerDecayOptimizerConstructor

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'vit_comer_s_sbn'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.vit_comer_s'
sync_bn = 'torch'  # NOTE: ViTCoMer uses SyncBatchNorm by default
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/Traffic-X/ViT-CoMer/blob/main/detection/configs/mask_rcnn/dinov2/mask_rcnn_dinov2_comer_small_fpn_3x_coco.py
# ViTCoMer: https://github.com/Traffic-X/ViT-CoMer/blob/main/detection/mmdet_custom/models/backbones/vit_comer.py#L19
# ViT: https://github.com/Traffic-X/ViT-CoMer/blob/main/detection/mmdet_custom/models/backbones/base/vit.py#L354
model.update(
    backbone=dict(
        type=ViTCoMer,
        #pretrain_size=592,  # when using dinov2, default is 224
        #img_size=592,       # when using dinov2, default is 224 (ViT arg)
        patch_size=16,       # ViT arg
        embed_dim=384,       # ViT arg
        depth=12,            # ViT arg
        num_heads=6,         # ViT arg but also used in VitCoMer
        mlp_ratio=4,         # ViT arg
        drop_path_rate=0.2,  # ViT arg but also used in VitCoMer
        conv_inplane=64,     # ViT arg
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        use_CTI_toV=[True, True, True, True],
        use_CTI_toC=[True, True, True, True],
        cnn_feature_interaction=[True, True, True, True],
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[True, True, False, True, True, False,      # ViT arg
                     True, True, False, True, True, False],
        window_size=[14, 14, None, 14, 14, None,                # ViT arg
                     14, 14, None, 14, 14, None],
        #pretrained=pretrained,                                 # ViT arg (we use init_cfg instead)
        frozen_stages=-1,
        use_fused_sdpa=True,
        init_cfg=dict(resolve='mmdet.vit_comer_s.fb_in1k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint="...."))
    neck=dict(in_channels=[384, 384, 384, 384])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    constructor=Vit_Comer_LayerDecayOptimizerConstructor,
    paramwise_cfg=dict(
        _delete_=True,
        num_layers=12, layer_decay_rate=0.70,
        #
        # NOTE: In their custom ViTDet deit comparison cfg, they use the following explicit cfg instead of
        #       the custom LayerDecayOptimizerConstructor:
        #       https://github.com/Traffic-X/ViT-CoMer/blob/main/detection/configs/mask_rcnn/mask_rcnn_deit_small_fpn_3x_coco.py
        #
        #custom_keys=dict(
        #    level_embed=dict(decay_mult=0.),
        #    pos_embed=dict(decay_mult=0.),
        #    norm=dict(decay_mult=0.),
        #    bias=dict(decay_mult=0.))
    )
)
# fp16 = dict(loss_scale=dict(init_scale=512))  # they also set this, we don't though

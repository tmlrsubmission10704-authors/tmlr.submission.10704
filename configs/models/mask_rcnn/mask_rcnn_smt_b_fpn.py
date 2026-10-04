from mmengine.config import read_base

from mmdet_one_piece.models.backbones.smt import SMT
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'smt_b'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.smt_b'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/AFeng-x/SMT/blob/main/detection/configs/mask_rcnn_smt_b_fpn.py
# https://github.com/AFeng-x/SMT/blob/main/detection/mmdet/models/backbones/smt.py#L234
# optim: https://github.com/AFeng-x/SMT/blob/main/detection/configs/mask_rcnn_smt_b_fpn_mstrain-poly_3x_coco.py
model.update(
    backbone=dict(
        type=SMT,
        embed_dims=[64, 128, 256, 512], 
        ca_num_heads=[4, 4, 4, -1], 
        sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[8, 6, 4, 2], 
        qkv_bias=True,
        drop_path_rate=0.2,
        depths=[4, 6, 28, 2],
        ca_attentions=[1, 1, 1, 0],
        num_stages=4,
        head_conv=7,
        expand_ratio=2,
        use_fused_sdpa=True,
        frozen_stages=-1,
        batch_norm_eval=True,  # semi-frozen BN, similar to ResNet Default in mmdet
        batch_norm_grad=True,
        init_cfg=dict(resolve='mmdet.smt_b.smt_in1k')),
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(in_channels=[64, 128, 256, 512])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict(                                       # the original cfg had these so we keep them, but:
            absolute_pos_embed=dict(decay_mult=0.),             # the model has no absolute_pos_embed keys
            relative_position_bias_table=dict(decay_mult=0.),   # the model has no relative biases table keys
            norm=dict(decay_mult=0.))
    )
)

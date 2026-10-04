from mmengine.config import read_base

from mmdet_one_piece.models.backbones.dat import DAT
#from mmengine.model.weight_init import PretrainedInit

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'dat++_s'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.dat++_s'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# https://github.com/LeapLabTHU/DAT-Detection/blob/main/configs/dat/mrcn_small_3x_4n_dp05_lr4.py
# https://github.com/LeapLabTHU/DAT-Detection/blob/main/configs/_base_/models/mask_rcnn_dat_fpn.py
# https://github.com/LeapLabTHU/DAT-Detection/blob/main/models/backbones/dat.py#L203
model.update(
    backbone=dict(
        type=DAT,
        dim_stem=96,
        dims=[96, 192, 384, 768],
        depths=[2, 4, 18, 2],
        stage_spec=[
            ["N", "D"], 
            ["N", "D", "N", "D"], 
            ["N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D"], 
            ["D", "D"]],
        heads=[3, 6, 12, 24],
        groups=[1, 2, 3, 6],
        use_pes=[True, True, True, True],
        strides=[8, 4, 2, 1],
        offset_range_factor=[-1, -1, -1, -1],
        use_dwc_mlps=[True, True, True, True],
        use_lpus=[True, True, True, True],
        use_conv_patches=True,
        ksizes=[9, 7, 5, 3],
        nat_ksizes=[7, 7, 7, 7],
        drop_path_rate=0.5,
        # ARGS from _base_ cfg (same es default args of constructor)
        img_size=224, patch_size=4,
        expansion=4,
        heads_q=[6, 12, 24, 48],
        window_sizes=[7, 7, 7, 7],
        drop_rate=0.0, attn_drop_rate=0.0,
        local_orf=[-1, -1, -1, -1],
        local_kv_sizes=[-1, -1, -1, -1],
        dwc_pes=[False, False, False, False],
        sr_ratios=[8, 4, 2, 1], 
        lower_lr_kvs={},
        fixed_pes=[False, False, False, False],
        no_offs=[False, False, False, False],
        ns_per_pts=[4, 4, 4, 4],
        ksize_qnas=[3, 3, 3, 3],
        nqs=[2, 2, 2, 2],
        qna_activation='exp',
        deform_groups=[0, 0, 0, 0],
        layer_scale_values=[-1,-1,-1,-1],
        use_cmt_mlps=[False, False, False, False],
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,  # set to True if OOM occurs
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.dat++_s.tc_in1k')),
        #init_cfg=dict(type=PretrainedInit, checkpoint="...")),
    neck=dict(in_channels=[96, 192, 384, 768])
)

# This model requires a different optimizer wrapper/settings.
# Config will be merged with/will override, the selected schedule config in train.py
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        norm_decay_mult=0.,
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.),
            relative_position_bias_table=dict(decay_mult=0.),
            rpe_table=dict(decay_mult=0.),
            norm=dict(decay_mult=0.)
            )
    )
)

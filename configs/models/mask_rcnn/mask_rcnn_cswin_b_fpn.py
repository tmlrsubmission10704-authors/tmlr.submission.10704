from mmengine.config import read_base

from mmdet_one_piece.models.backbones.cswin_transformer import CSWin
#from mmengine.model.weight_init import PretrainedInit
import torch.nn as nn

with read_base():
    from ._mask_rcnn_fpn import model, model_framework_name, model_neck_name, train_pipeline_LA, test_pipeline_LA

model_backbone_name = 'cswin_b'  # short name used for wandb logging and run_name
compatible_backbone_weights = 'mmdet.cswin_b'
sync_bn = None
assert_sync_bn = sync_bn  # protect against overrides somewhere else in the config
# NOTE: we use the cfg from segmentation which, according to the paper, is the same as used for detection
#       see Appendix: Experiment Details https://arxiv.org/pdf/2107.00652
# https://github.com/microsoft/CSWin-Transformer/blob/main/segmentation/configs/cswin/upernet_cswin_base.py
# https://github.com/microsoft/CSWin-Transformer/blob/main/segmentation/configs/_base/upernet_cswin.py
# https://github.com/microsoft/CSWin-Transformer/blob/main/segmentation/backbone/cswin_transformer.py#L269
model.update(
    backbone=dict(
        type=CSWin,
        embed_dim=96,
        patch_size=4,
        depth=[2,4,32,2],
        num_heads=[2,4,8,16],
        split_size=[1,2,7,7],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.6,
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
        use_chk=False,
        frozen_stages=-1,
        init_cfg=dict(resolve='mmdet.cswin_b.ms_in1k')),
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(in_channels=[96,192,384,768])
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

# pre-trained backbone weights, sorted by implementation

# detectron2 url resolution (so we can simply copy entries from their configs)
D2_PREFIX = "detectron2://"
S3_DETECTRON2_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

BACKBONE_WEIGHTS = {
    # mmdetection
    'mmdet.resnet50.tv1_in1k': "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    'mmdet.resnet50.tv2_in1k': "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    'mmdet.resnet101.tv1_in1k': "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    'mmdet.resnet101.tv2_in1k': "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
    'mmdet.resnet152.tv1_in1k': "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    'mmdet.resnet152.tv2_in1k': "https://download.pytorch.org/models/resnet152-f82ba261.pth",
    'mmdet.swin_t_p4_w7.ms_in1k': "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    'mmdet.swin_l_p4_w12_384.ms_in22k': "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth",
    'mmdet.pvt_s.pvt_in1k': "https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth",
    'mmdet.pvt_v2_b2.pvt_in1k': "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth",
    'mmdet.uniformer_s.uf_in1k': "gdrive://1-uepH3Q3BhTmWU6HK-sGAGQC_MpfIiPD/uniformer_small_in1k.pth/",  # https://github.com/Sense-X/UniFormer/tree/main/image_classification#imagenet-1k-pretrained-224x224
    'mmdet.uniformer_b.uf_in1k': "gdrive://1-wT39QazTGELxgrQIu6J12D3qcla3hui/uniformer_base_in1k.pth/",  # https://github.com/Sense-X/UniFormer/tree/main/image_classification#imagenet-1k-pretrained-224x224
    'mmdet.fan_tiny_8_p4_hybrid.nv_in1k': "https://github.com/zhoudaquan/fully_attentional_network_ckpt/releases/download/v1.0.0/fan_hybrid_tiny.pth.tar",  # https://github.com/NVlabs/FAN/
    'mmdet.fan_small_12_p4_hybrid.nv_in1k': "https://github.com/zhoudaquan/fully_attentional_network_ckpt/releases/download/v1.0.0/fan_hybrid_small.pth.tar",  # https://github.com/NVlabs/FAN/
    'mmdet.moganet_s.wl_in1k': "https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_small_sz224_8xbs128_ep300.pth.tar",  # https://github.com/Westlake-AI/MogaNet/blob/main/README.md#3-imagenet-1k-trained-models
    'mmdet.focalnet_t_p4_srf.ms_in1k': "https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_tiny_srf.pth",  # https://github.com/microsoft/FocalNet/blob/main/README.md#image-classification-on-imagenet-1k
    'mmdet.focalnet_t_p4_lrf.ms_in1k': "https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_tiny_lrf.pth",  # https://github.com/microsoft/FocalNet/blob/main/README.md#image-classification-on-imagenet-1k
    'mmdet.visformer_s_v2.vf_in1k': "https://github.com/danczs/Visformer/releases/download/v1.0.0/swin_visformer_small_v2.pth", # https://github.com/danczs/Visformer
    'mmdet.deit_adapter_s.fb_in1k': "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth", # https://github.com/czczup/ViT-Adapter/blob/main/detection/README.md
    'mmdet.smt_s.smt_in1k': "https://github.com/AFeng-x/SMT/releases/download/v1.0.0/smt_small.pth",  #  https://github.com/AFeng-x/SMT/blob/main/README.md#main-results-on-imagenet-with-pretrained-models
    'mmdet.smt_b.smt_in1k': "https://github.com/AFeng-x/SMT/releases/download/v1.0.0/smt_base.pth",  #  https://github.com/AFeng-x/SMT/blob/main/README.md#main-results-on-imagenet-with-pretrained-models
    'mmdet.smt_l.smt_in22k_in1k': "https://github.com/AFeng-x/SMT/releases/download/v1.0.0/smt_large_22k_224_ft.pth",  #  https://github.com/AFeng-x/SMT/blob/main/README.md#main-results-on-imagenet-with-pretrained-models
    'mmdet.intern_image_t.hf_in1k': "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth",  # https://github.com/OpenGVLab/InternImage?tab=readme-ov-file#released-models
    'mmdet.intern_image_s.hf_in1k': "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_s_1k_224.pth",  # https://github.com/OpenGVLab/InternImage?tab=readme-ov-file#released-models
    'mmdet.intern_image_b.hf_in1k': "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth",  # https://github.com/OpenGVLab/InternImage?tab=readme-ov-file#released-models
    'mmdet.flash_intern_image_t.hf_in1k': "https://huggingface.co/OpenGVLab/DCNv4/resolve/main/flash_intern_image_t_1k_224.pth",  # https://github.com/OpenGVLab/DCNv4/tree/main?tab=readme-ov-file#released-models
    'mmdet.cswin_t.ms_in1k': "https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_tiny_224.pth",  # https://github.com/microsoft/CSWin-Transformer/blob/main/README.md#main-results-on-imagenet
    'mmdet.cswin_s.ms_in1k': "https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_small_224.pth",  # https://github.com/microsoft/CSWin-Transformer/blob/main/README.md#main-results-on-imagenet
    'mmdet.cswin_b.ms_in1k': "https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_base_224.pth",  # https://github.com/microsoft/CSWin-Transformer/blob/main/README.md#main-results-on-imagenet
    'mmdet.biformer_s.bi_in1k': "manual://https://github.com/rayleizhu/BiFormer?tab=readme-ov-file#imagenet-1k-trained-models::onedrive/classification/biformer_small_best.pth",  # https://github.com/rayleizhu/BiFormer?tab=readme-ov-file#imagenet-1k-trained-models
    'mmdet.biformer_b.bi_in1k': "manual://https://github.com/rayleizhu/BiFormer?tab=readme-ov-file#imagenet-1k-trained-models::onedrive/classification/biformer_base_best.pth",  # https://github.com/rayleizhu/BiFormer?tab=readme-ov-file#imagenet-1k-trained-models
    'mmdet.transnext_t.tx_in1k': "https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth",  #  https://github.com/DaiShiResearch/TransNeXt/blob/main/README.md#image-classification
    'mmdet.transnext_s.tx_in1k': "https://huggingface.co/DaiShiResearch/transnext-small-224-1k/resolve/main/transnext_small_224_1k.pth",  #  https://github.com/DaiShiResearch/TransNeXt/blob/main/README.md#image-classification
    'mmdet.transnext_b.tx_in1k': "https://huggingface.co/DaiShiResearch/transnext-base-224-1k/resolve/main/transnext_base_224_1k.pth",  #  https://github.com/DaiShiResearch/TransNeXt/blob/main/README.md#image-classification
    'mmdet.slack_t_p4_w7.vg_in1k': "gdrive://1Iut2f5FMS_77jGPYoUJDQzDIXOsax1u4/SLaK_tiny_checkpoint.pth/",  # https://github.com/VITA-Group/SLaK/blob/44b4aab0611dff0023873ed6af8e0eefcea6f056/README.md#slak-with-51x51-kernels-trained-on-imagenet-1k-for-300-epochs
    'mmdet.vit_comer_s.fb_in1k': "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth", # https://github.com/Traffic-X/ViT-CoMer/blob/main/detection/README.md
    'mmdet.hornet_t_7x7.tc_in1k': "https://cloud.tsinghua.edu.cn/f/1ca970586c6043709a3f/?dl=1::hornet_tiny_7x7.pth", # https://github.com/raoyongming/HorNet/blob/master/object_detection/README.md#results-and-fine-tuned-models
    'mmdet.hornet_t_gf.tc_in1k':  "https://cloud.tsinghua.edu.cn/f/511faad0bde94dfcaa54/?dl=1::hornet_tiny_gf.pth",  # https://github.com/raoyongming/HorNet/blob/master/object_detection/README.md#results-and-fine-tuned-models
    'mmdet.dat++_t.tc_in1k': "https://cloud.tsinghua.edu.cn/f/14c5ddae10b642e68089/?dl=1::dat_pp_tiny_in1k_224.pth",  # https://github.com/LeapLabTHU/DAT?tab=readme-ov-file#evaluate-pretrained-models-on-imagenet-1k-classification
    'mmdet.dat++_s.tc_in1k': "https://cloud.tsinghua.edu.cn/f/4c2a76360c964fbd81d5/?dl=1::dat_pp_small_in1k_224.pth",  # https://github.com/LeapLabTHU/DAT?tab=readme-ov-file#evaluate-pretrained-models-on-imagenet-1k-classification
    'mmdet.dat++_b.tc_in1k': "https://cloud.tsinghua.edu.cn/f/8e30492404d348d89f25/?dl=1::dat_pp_base_in1k_224.pth",  # https://github.com/LeapLabTHU/DAT?tab=readme-ov-file#evaluate-pretrained-models-on-imagenet-1k-classification
    'mmdet.nat_t.sl_in1k': "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_tiny.pth",  # https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/NAT.md#classification
    'mmdet.dinat_t.sl_in1k': "https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_tiny_in1k_224.pth",  # https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/main/detection/DiNAT.md
    'mmdet.gc_vit_t.nv_in1k': "https://huggingface.co/nvidia/GCViT/resolve/main/gcvit_1k_tiny.pth.tar",  # https://huggingface.co/nvidia/GCViT/tree/main
    'mmdet.poolformer_s24.sa_in1k': "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar", # https://github.com/sail-sg/poolformer/blob/main/README.md#2-poolformer-models
    'mmdet.poolformer_v2_s36.sa_in1k': "https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s36.pth",  # https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py#L69
    'mmdet.identityformer_s36.sa_in1k': "https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s36.pth",  # https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py#L46
    'mmdet.caformer_s18.sa_in1k': "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth",  # https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py#L135
    'mmdet.caformer_m36.sa_in1k': "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36.pth",  # https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py#L164
    'mmdet.caformer_b36.sa_in1k': "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36.pth",  # https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py#L177
    'mmdet.convformer_s18.sa_in1k':  "https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18.pth",  # https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py#L78
    'mmdet.vil2_s.nx_in1k': "https://ml.jku.at/research/vision_lstm/download/vil2_small16_e400_in1k.th",  # https://github.com/NX-AI/vision-lstm/tree/main?tab=readme-ov-file#pre-trained-models
    'mmdet.vitdet_s.fb_in1k': "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",  # https://github.com/facebookresearch/deit/blob/main/README_deit.md#model-zoo
    'mmdet.vitdet_s.fb_in1k_d3': "https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth",  # https://github.com/facebookresearch/deit/blob/main/README_revenge.md
    'mmdet.mpvit_s.ek_in1k': "https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth",  # https://github.com/youngwanLEE/MPViT?tab=readme-ov-file#main-results-on-imagenet-1k
    'mmdet.vim_s.vl_in1k': "https://huggingface.co/hustvl/Vim-small-midclstok/resolve/main/vim_s_midclstok_80p5acc.pth", # https://github.com/hustvl/Vim?tab=readme-ov-file#model-weights
    'mmdet.vim_s.vl_ft_in1k': "https://huggingface.co/hustvl/Vim-small-midclstok/resolve/main/vim_s_midclstok_ft_81p6acc.pth", # https://github.com/hustvl/Vim?tab=readme-ov-file#model-weights
    'mmdet.vssm_t.mm_in1k': "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth",  # https://github.com/MzeroMiko/VMamba/tree/main?tab=readme-ov-file#classification-on-imagenet-1k
    'mmdet.vssm_s.mm_in1k': "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth",  # https://github.com/MzeroMiko/VMamba/tree/main?tab=readme-ov-file#classification-on-imagenet-1k
    'mmdet.vssm_b.mm_in1k': "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth",  # https://github.com/MzeroMiko/VMamba/tree/main?tab=readme-ov-file#classification-on-imagenet-1k
    'mmdet.ms_vssm_t.yh_in1k': "https://huggingface.co/YuhengSSS/MSVMamba/resolve/main/msvmambav3_tiny_224.pth",  # https://github.com/YuHengsss/MSVMamba/tree/v2?tab=readme-ov-file#classification-on-imagenet-1k
    'mmdet.vrwkv_adapter_s.ogvl_in1k': "https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_s_in1k_224.pth",  # https://github.com/YuHengsss/MSVMamba/tree/v2?tab=readme-ov-file#classification-on-imagenet-1k
    'mmdet.trvdet_s.ba_eva02_mae_in21k': "https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/pt/eva02_S_pt_in21k_p14.pt",  # https://github.com/baaivision/EVA/tree/master/EVA-02/asuka#mim-pre-trained-eva-02
    'mmdet.trvdet_b.ba_eva02_mae_in21k': "https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/pt/eva02_B_pt_in21k_p14.pt",  # https://github.com/baaivision/EVA/tree/master/EVA-02/asuka#mim-pre-trained-eva-02
    'mmdet.trvdet_b.ba_eva02_mae_in21k_p14to16': "https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/pt/eva02_B_pt_in21k_p14to16.pt",  # https://github.com/baaivision/EVA/tree/master/EVA-02/asuka#mim-pre-trained-eva-02
    'mmdet.vit7b16.dinov3': "",
    # mmpretrain
    'mmpre.convnext_t_p4_w7.mm_in1k': "https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth",  # https://mmpretrain.readthedocs.io/en/latest/papers/convnext.html also see cfg of mmdet
    'mmpre.convnext_v2_t.mm_in1k': "https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-tiny_fcmae-pre_3rdparty_in1k_20230104-471a86de.pth",  # https://mmpretrain.readthedocs.io/en/latest/papers/convnext_v2.html
    'mmpre.convnext_v2_b.mm_in1k': "https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-pre_3rdparty_in1k_20230104-00a70fa4.pth",  # https://mmpretrain.readthedocs.io/en/latest/papers/convnext_v2.html
    'mmpre.convnext_v2_l.mm_in1k': "https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-large_fcmae-pre_3rdparty_in1k_20230104-ef393013.pth",  # https://mmpretrain.readthedocs.io/en/latest/papers/convnext_v2.html
    # timm official weights (pretrained_tag)
    'timm.resnet50.a1_in1k':    'a1_in1k',
    'timm.resnet50.a1h_in1k':   'a1h_in1k',
    'timm.resnet50.a2_in1k':    'a2_in1k',
    'timm.resnet50.a3_in1k':    'a3_in1k',
    'timm.resnet50.am_in1k':    'am_in1k',
    'timm.resnet50.b1k_in1k':   'b1k_in1k',
    'timm.resnet50.b2k_in1k':   'b2k_in1k',
    'timm.resnet50.bt_in1k':    'bt_in1k',
    'timm.resnet50.c1_in1k':    'c1_in1k',
    'timm.resnet50.c2_in1k':    'c2_in1k',
    'timm.resnet50.d_in1k':     'd_in1k',
    'timm.resnet50.fb_ssl_yfcc100m_ft_in1k': 'fb_ssl_yfcc100m_ft_in1k',
    'timm.resnet50.fb_swsl_ig1b_ft_in1k':    'fb_swsl_ig1b_ft_in1k',
    'timm.resnet50.gluon_in1k': 'gluon_in1k',
    'timm.resnet50.ra_in1k':    'ra_in1k',
    'timm.resnet50.ram_in1k':   'ram_in1k',
    'timm.resnet50.tv2_in1k':   'tv2_in1k',
    'timm.resnet50.tv1_in1k':   'tv_in1k',  # timm uses tv_, not tv1_!
    'timm.maxvit_t_tf_224.tf_in1k': 'in1k',
    'timm.maxvit_s_tf_224.tf_in1k': 'in1k',
    'timm.maxvit_b_tf_224.tf_in1k': 'in1k',
    'timm.hieradet_t.fb_sam2':  'sam2_fb_r896',
    'timm.hieradet_t.fb_sam21': 'sam2_fb_r896_2pt1',
    'timm.hieradet_t.fb_mae_in1k': 'mae_in1k_ft_in1k',  # timm links this to hiera_tiny_224.mae_in1k_ft_in1k
    'timm.swin_v2_t_p4_w8.ms_in1k': 'ms_in1k',
    'timm.swin_v2_t_p4_w16.ms_in1k': 'ms_in1k',
    'timm.mamba_out_t.yu_in1k': 'in1k',
    'timm.tf_efficientnet_b4.aa_in1k': 'aa_in1k',
    'timm.tf_efficientnet_b5.aa_in1k': 'aa_in1k',
    # timm custom weights (external file that must be loaded as local file)
    'timm.robust_resnet_dw.uv_in1k':   "gdrive://1cbS3NGkkzKw2Uhq8ATMbsoGjIx8zhwgv/checkpoint-299.pth.tar/robust_resnet_dw_small_in1k.pth.tar",  # https://github.com/UCSC-VLAA/RobustCNN/blob/main/README.md#pretrained-models
    'timm.robust_resnet_idw.uv_in1k':  "gdrive://1g551TsZmVrSZ4BXje9RcT7gG_UjjFQmO/checkpoint-299.pth.tar/robust_resnet_inverted_dw_small_in1k.pth.tar",  # https://github.com/UCSC-VLAA/RobustCNN/blob/main/README.md#pretrained-models
    'timm.robust_resnet_uidw.uv_in1k': "gdrive://1lQ0zPqO6nmtXWt5r9d-M4k_GHVeW41Qy/checkpoint-299.pth.tar/robust_resnet_up_inverted_dw_small_in1k.pth.tar",  # https://github.com/UCSC-VLAA/RobustCNN/blob/main/README.md#pretrained-models
    'timm.robust_resnet_didw.uv_in1k': "gdrive://1gZVclPJXT50F6iAJHUv8Z6wG6C0ZhXds/checkpoint-299.pth.tar/robust_resnet_down_inverted_dw_small_in1k.pth.tar",  # https://github.com/UCSC-VLAA/RobustCNN/blob/main/README.md#pretrained-models
    # detectron 2
    'd2.mvit_v2_t.fb_in1k': f"detectron2://ImageNetPretrained/mvitv2/MViTv2_T_in1k.pyth".replace(D2_PREFIX, S3_DETECTRON2_PREFIX),
    'd2.mvit_v2_s.fb_in1k': f"detectron2://ImageNetPretrained/mvitv2/MViTv2_S_in1k.pyth".replace(D2_PREFIX, S3_DETECTRON2_PREFIX),
    'd2.mvit_v2_b.fb_in1k': f"detectron2://ImageNetPretrained/mvitv2/MViTv2_B_in1k.pyth".replace(D2_PREFIX, S3_DETECTRON2_PREFIX),
    'd2.mvit_v2_b.fb_in21k': f"detectron2://ImageNetPretrained/mvitv2/MViTv2_B_in21k.pyth".replace(D2_PREFIX, S3_DETECTRON2_PREFIX),
    'd2.mvit_v2_l.fb_in1k': f"detectron2://ImageNetPretrained/mvitv2/MViTv2_L_in1k.pyth".replace(D2_PREFIX, S3_DETECTRON2_PREFIX),
    'd2.mvit_v2_l.fb_in21k': f"detectron2://ImageNetPretrained/mvitv2/MViTv2_L_in21k.pyth".replace(D2_PREFIX, S3_DETECTRON2_PREFIX),
    'd2.vitdet_b.fb_mae': f"detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth".replace(D2_PREFIX, S3_DETECTRON2_PREFIX),
    'd2.vitdet_l.fb_mae': f"detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_large.pth".replace(D2_PREFIX, S3_DETECTRON2_PREFIX),
}

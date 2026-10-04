# tmlr.submission.10704

Anonymous Code Repository for TMLR Submission 10704

![Banner](wandb_banner.png)

> This repository includes lfs files. You can enable lfs with `git lfs install`

## Configs

Please note that datasets only includes "coco" datasets. The annotations (and image paths) for nao, coco-o, coco-c and ocs-coco are set dynamically from `evaluators/eval.py` and from code (not in included here). Note further that the plain ViT model configs in `models/mask_rcnn` are not yet updated with the values found in the last revision.

## Evaluation Bugs

The directory `evaluation_bugs` includes two standalone notebooks to understand and reproduce the mentioned in the submission. We provide results files for reproducing evals without installing detectron 2 and adet. Please note that we noticed that we ran the ViTDet-H model with the mmdet image loading pipeline (uses opencv by default). Detectron 2 uses pillow by default which will result in slightly different results (in the second decimal). We will re-run this experiment for the next version of the paper, with a minor change to be expected.

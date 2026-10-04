#from mmengine.config import read_base
#with read_base():
#    from mmdet.configs._base_.default_runtime import *
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
#from mmengine.runner import LogProcessor
from mmengine.visualization import LocalVisBackend

from mmdet.engine.hooks import DetVisualizationHook #, CheckInvalidLossHook
from mmdet.visualization import DetLocalVisualizer

from mmdet_one_piece.engine import Runner_OP, LoggerHook_OP, LogProcessor_OP, CheckInvalidLossHook_OP

# This is mmdet.configs._base_.default_runtime with a few small adjustments
default_scope = None

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=100, interval_exp_name=1000000),  # basically never log the exp_name
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, interval=1, save_optimizer=False, save_param_scheduler=False),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=DetVisualizationHook),
    check_loss_hook=dict(type=CheckInvalidLossHook_OP, interval=50, priority='VERY_LOW')
    )

# NOTE: 'fork' is significantly faster than 'spawn' because we have many dataloader workers
#       In case we use seq. eval, we have to increase the default timeout because its not enough.
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
    #dist_cfg=dict(backend='nccl', timeout=3600)
)

vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(
    type=DetLocalVisualizer, vis_backends=vis_backends, name='visualizer')
# custom LogProcessor_OP: will not add val/train val/val val/runtimes to log_str for the default LoogerHook
# since we have a lot of metrics this would be to much and we have custom logging anyways...
# add any test_pipeline names here to avoid large log prints
log_processor = dict(type=LogProcessor_OP, window_size=100, by_epoch=True, ignore_non_scalars=['se_800'])

#log_level = 'DEBUG'
log_level = 'INFO'
load_from = None
resume = False

# newly added
runner_type = Runner_OP
custom_hooks = [dict(type=LoggerHook_OP,
                     test_pipeline_name=None,  # must be updated in train.py to the actual pipeline name string
                     wandb_cfg=dict(
                         entity=None,  # add dynamically in train.py
                         project=None, # add dynamically in train.py
                         mode=None))]  # add dynamically in train.py

# NOTE: update the general 'seed' dynamically in train.py
#       for sampler_seed and data_seed see datasets/dataloader.py
randomness = dict(seed=0, deterministic=True, diff_rank_seed=True, cuda_matmul_allow_tf32=False)

# used to populate pipelines, datasets and val_evaluator in train.py
backend_args = None

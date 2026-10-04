from mmengine.config import read_base

with read_base():
    from ._deta_cm import *

# used for wandb logging and run_name
model_framework_name = 'deta_4s'

# update model
model.update(num_feature_levels = 4)

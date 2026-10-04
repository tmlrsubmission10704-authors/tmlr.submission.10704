from mmengine.config import read_base

# Make batch_size=2 pre GPU the default
with read_base():
    from .dataloader_2 import *

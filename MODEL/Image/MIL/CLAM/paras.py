"""
pre-defined parameters for (pretrained) transformer model loaded from https://github.com/peng-lab/HistoBistro/tree/main
"""

import attr 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

#---->
import pdb

@attr.s(auto_attribs=True)
class CLAMParas:
    """
    include all parameters to create a Transformer
    """
    #------> parameters for model
    encoder_name="pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False
    
    
    input_dim:int = 1024           # by default we'll be using uni's feature vectors which are of size 1024
    encoder_name:str = 'pre-calculated' # by default we'll be using the foundational models for feature extraction, so we avoid SSL
    task:str = 'binary'
    criterion:str = 'BCEWithLogitsLoss'
    gate:bool = True
    size_arg:str = "small"
    dropout = 0.25  # https://github.com/search?q=repo%3Amahmoodlab%2FCLAM%20drop_out&type=code
    k_sample = 8
    n_classes:int = 1
    num_classes:int = 1
    instance_loss_fn = nn.CrossEntropyLoss()
    subtyping = False

    epoch:int = 42
    lr:float = 2e-4 # https://github.com/mahmoodlab/CLAM/blob/8455dc8c4623f8881281c65a3725bbdad99fb59e/main.py
    wd:float = 1e-5
    optimizer = 'Adam' # same as https://arxiv.org/pdf/2004.09666
    lr_scheduler = 'CosineAnnealingLR'
    lr_scheduler_config:dict = {'T_max':42, 
                                'eta_min':1e-6} # assumes cosine annealing

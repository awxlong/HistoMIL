"""
pre-defined parameters for TransMIL model
"""
import torch.nn as nn
import attr 

#---->

@attr.s(auto_attribs=True)
class TransMILRegressionParas:
    """
    include all paras for create TransMIL model
    """
    #------> parameters for model
    encoder_name:str = "pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False
    feature_size:int = 512
    embed_size:int = None
    input_dim:int = 1024
    num_classes:int = 1
    norm_layer=nn.LayerNorm
    pos_enc = 'PPEG'

    ### OPTIMIZER CONFIGURATIONS
    task:str = 'regression'
    criterion:str = 'MSELoss'

    epoch:int = 32
    lr:float = 2e-5 # 1.0e-4 # same as https://arxiv.org/pdf/2106.00908
    wd:float = 1e-2
    optimizer = 'AdamW'
    lr_scheduler = 'CosineAnnealingLR'
    lr_scheduler_config:dict = {'T_max':32, 
                                'eta_min':1e-6} # assumes cosine annealing
    


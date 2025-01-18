"""
Default hyperparameters for TransMIL model with multimodal fusion
"""
import torch.nn as nn
import attr 

#---->

@attr.s(auto_attribs=True)
class TransMILMultimodalParas:
    """
    include all paras for create TransMIL model
    """
    #------> parameters for model
    encoder_name:str = "pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False
    feature_size:int = 512
    embed_size:int = 64
    input_dim:int = 1024
    num_classes:int = 1
    norm_layer=nn.LayerNorm
    pos_enc = 'PPEG'

    ### clinical features parameters
    idx_continuous = 27
    taxonomy_in = 2

    ### Multimodal configurations
    fusion_type:str = 'kron'
    scale:list = [2, 1]
    ### OPTIMIZER CONFIGURATIONS
    task:str = 'binary'
    criterion:str = 'BCEWithLogitsLoss'

    epoch:int = 32
    lr:float = 2e-5 # 1.0e-4 # same as https://arxiv.org/pdf/2106.00908
    wd:float = 1e-2
    optimizer = 'AdamW'
    lr_scheduler = 'CosineAnnealingLR'
    lr_scheduler_config:dict = {'T_max':32, 
                                'eta_min':1e-6} # assumes cosine annealing
    


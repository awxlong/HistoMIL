"""
pre-defined parameters for (pretrained) transformer model loaded from https://github.com/peng-lab/HistoBistro/tree/main
"""

import attr 
import torch
import torch.nn as nn


#---->
import pdb

# AVAILABLE_WEIGHTS = ['BRAF_CRC_model.pth', 'KRAS_CRC_model.pth', 'MSI_high_CRC_model.pth']

@attr.s(auto_attribs=True)
class TransformerRegressionParas:
    """
    include all parameters to create a Transformer
    """
    #------> parameters for model
    encoder_name="pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False
    
    num_classes:int = 1
    input_dim:int = 1024           # by default we'll be using uni's feature vectors which are of size 1024
    
    dim:int =  256 # 512 # has  to be equal to # of heads * dim_head 
    depth:int = 2
    heads:int = 8 # 8
    dim_head:int = 32 # 64
    mlp_dim:int = 256 # 512 # has  to be equal to # of heads * dim_head 
    pool:str = 'cls'
    
    dropout:float = 0.1
    emb_dropout:float = 0.1
    pos_enc:nn.Module = None
    
    encoder_name:str = 'pre-calculated' # by default we'll be using the foundational models for feature extraction, so we avoid SSL
    task:str = 'regression'
    criterion:str = 'MSELoss'
    pos_weight = torch.ones(1)
    epoch:int = 10
    lr:float = 2.0e-5 # same as https://github.com/peng-lab/HistoBistro/blob/main/config.yaml
    wd:float = 2.0e-05
    optimizer = 'AdamW'
    lr_scheduler = 'CosineAnnealingLR'
    lr_scheduler_config:dict = {'T_max':epoch, 
                                'eta_min':1e-6} # assumes cosine annealing
    
    selective_finetuning = False

    ### clinical features parameters
    idx_continuous = 3
    taxonomy_in = 2
    embed_size:int = 64

    ### Multimodal configurations
    fusion_type:str = 'kron'
    scale:list = [2, 1]

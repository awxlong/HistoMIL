"""
pre-defined parameters for (pretrained) transformer model loaded from https://github.com/peng-lab/HistoBistro/tree/main
"""

import attr 
import torch
import torch.nn as nn


#---->
import pdb

@attr.s(auto_attribs=True)
class GraphTransformerParas:
    """
    include all parameters to create a Transformer
    """
    #------> parameters for model
    encoder_name="pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False
    
    n_class:int = 1
    n_features:int = 1024           # by default we'll be using uni's feature vectors which are of size 1024
    pretrained_input_dim:int = 768 # pretrained transformer used ctranspath by default for feature encoding
    dim:int =  256 # 512 # has  to be equal to # of heads * dim_head 
    depth:int = 2
    heads:int = 8 # 8
    dim_head:int = 32 # 64
    mlp_dim:int = 256 # 512 # has  to be equal to # of heads * dim_head 
    pool:str = 'cls'
    
    dropout:float = 0.1
    emb_dropout:float = 0.1
    
    encoder_name:str = 'pre-calculated' # by default we'll be using the foundational models for feature extraction, so we avoid SSL
    task:str = 'binary'
    criterion:str = 'BCEWithLogitsLoss'
    pos_weight = torch.ones(1)
    epoch:int = 4
    lr:float = 2.0e-5 # same as https://github.com/peng-lab/HistoBistro/blob/main/config.yaml
    wd:float = 2.0e-05
    optimizer = 'Adam'
    lr_scheduler = 'CosineAnnealingLR'
    lr_scheduler_config:dict = {'T_max':epoch, 
                                'eta_min':1e-6} # assumes cosine annealing
    
DEFAULT_GRAPHTRANSFORMER_PARAS = GraphTransformerParas()
    
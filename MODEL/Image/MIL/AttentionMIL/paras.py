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

AVAILABLE_WEIGHTS = ['BRAF_CRC_model.pth', 'KRAS_CRC_model.pth', 'MSI_high_CRC_model.pth']

@attr.s(auto_attribs=True)
class AttentionMILParas:
    """
    include all parameters to create a Transformer
    """
    #------> parameters for model
    encoder_name="pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False
    
    num_classes:int = 1
    input_dim:int = 1024           # by default we'll be using uni's feature vectors which are of size 1024
    encoder_name:str = 'pre-calculated' # by default we'll be using the foundational models for feature extraction, so we avoid SSL
    task:str = 'binary'
    criterion:str = 'BCEWithLogitsLoss'

    epoch:int = 32
    lr:float = 2e-5 # 1.0e-4 # same as https://www.nature.com/articles/s41467-024-45589-1#Sec1
    wd:float = 1e-2
    max_lr:float = 1e-4 # same as https://www.cell.com/cancer-cell/fulltext/S1535-6108(23)00278-7#gr1
    optimizer = 'Adam'
    lr_scheduler = 'OneCycleLR'
    lr_scheduler_config:dict = {'max_lr':epoch, 
                                'steps_per_epoch': epoch} # should set to len(trainloader)
    
    


DEFAULT_Attention_MIL_PARAS = AttentionMILParas(input_dim=1024, task='binary', \
                                              num_classes=1, encoder_name='pre-calculated')

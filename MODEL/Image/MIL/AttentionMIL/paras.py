"""
Default hyperparameters for AttentionMIL: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
"""

import attr 

#---->
import pdb

@attr.s(auto_attribs=True)
class AttentionMILParas:
    """
    include all parameters to create a Transformer
    """
    #------> parameters for model
    encoder_name="pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False
    
    num_classes:int = 1
    input_dim:int = 1024                # by default we'll be using uni's feature vectors which are of size 1024
    encoder_name:str = 'pre-calculated' # by default we'll be using the foundational models for feature extraction, so we avoid SSL
    task:str = 'binary'
    criterion:str = 'BCEWithLogitsLoss'

    epoch:int = 32
    lr:float = 2e-5 
    wd:float = 1e-2
    max_lr:float = 1e-4 
    optimizer = 'Adam'
    lr_scheduler = 'OneCycleLR'
    lr_scheduler_config:dict = {'max_lr':epoch, 
                                'steps_per_epoch': epoch} # should set to len(trainloader)
    
    
"""
pre-defined parameters for TransMIL model
"""
import torch.nn as nn
import attr 
import torch
#---->
def get_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
@attr.s(auto_attribs=True)
class DTFDTransMILParas:
    """
    include all paras for create TransMIL model
    """
    #------> parameters for model
    encoder_name:str = "pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False
    feature_extractor_name:str = ""
    ### DTFD-MIL Params according to https://github.com/Dootmaan/DTFD-MIL.PyTorch/blob/main/train_DTFT-MIL.py#L49
    numGroup:int = 5
    input_dim:int = 1024
    mDim:int = 512
    device:str = get_available_device()
    num_cls:int = 1
    droprate:float = 0
    droprate_2:float = 0
    numLayer_Res:int = 0
    grad_clipping:int = 5
    


    ### OPTIMIZER CONFIGURATIONS
    task:str = 'binary'
    criterion:str = 'BCEWithLogitsLoss'

    epoch:int = 42
    lr:float = 2e-5 # same as https://github.com/Dootmaan/DTFD-MIL.PyTorch/blob/main/train_DTFT-MIL.py
    weight_decay:float = 1e-4
    lr_decay_ratio:float = 0.2
    



"""
Default hyperparameters for CAMIL https://github.com/olgarithmics/ICLR_CAMIL/
"""

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
class CAMILParas:
    """
    include all paras for create CAMIL model
    """
    #------> parameters for model
    encoder_name:str = "pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False
    input_shape:int = 1024
    num_classes:int = 1
    subtyping:bool = False
    ### OPTIMIZER CONFIGURATIONS
    task:str = 'binary'
    criterion:str = 'BCEWithLogitsLoss'
    device:str = get_available_device()
    epoch:int = 30
    lr:float = 0.0002 # 2e-5 # same as https://github.com/olgarithmics/ICLR_CAMIL/blob/ddd8e2e3973d234310f47c6a528ebbb2eaf369a0/args.py#L3
    wd:float = 1e-5
    optimizer = 'Adam' # 'AdamW'
    lr_scheduler = 'ReduceLROnPlateau'
    lr_scheduler_config:dict = {'factor':0.2,
                                'mode': 'min',
                                } 

def to_coo(sparse_tensor):
    return sparse_tensor.coalesce().to_sparse_coo()

def custom_camil_collate(batch):
    '''
    Custom batching for sparse tensors, working with computational histopathology 
    '''
    data_inputs = torch.stack([item[0] for item in batch])
    adj_matrices = [item[1] for item in batch]
    labels = torch.stack([torch.tensor(item[2]) for item in batch])
    return data_inputs, adj_matrices, labels

"""
pre-defined parameters for CAMIL model
"""
# import torch.nn as nn
import attr 
import torch
#---->

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

    epoch:int = 30
    lr:float = 0.0002 # 2e-5 # same as https://github.com/olgarithmics/ICLR_CAMIL/blob/ddd8e2e3973d234310f47c6a528ebbb2eaf369a0/args.py#L3
    wd:float = 1e-5
    optimizer = 'Adam' # 'AdamW'
    lr_scheduler = 'ReduceLROnPlateau'
    lr_scheduler_config:dict = {'factor':0.2,
                                'mode': 'min',
                                # 'monitor':'loss_val'
                                } 
    # lr_scheduler_config:dict = {'T_max':30, 
    #                             'eta_min':1e-6} # assumes cosine annealing



# @attr.s(auto_attribs=True)
# class TransMILParas:
#     """
#     include all paras for create TransMIL model
#     """
#     #------> parameters for model
#     encoder_name="pre-calculated"# or name of encoder",
#     encoder_pretrained:bool = True # or False
#     feature_size:int=512
#     embed_size:int=None

#     n_classes:int=2
#     norm_layer=nn.LayerNorm
    #class_nb:int=1

    #------> parameters for feature encoder
    #backbone:str="pre-calculated"
    #pretrained:bool=True

def to_coo(sparse_tensor):
    return sparse_tensor.coalesce().to_sparse_coo()

def custom_camil_collate(batch):
    '''
    custom batching for sparse tensors, working with computational histopathology 
    '''
    data_inputs = torch.stack([item[0] for item in batch])
    adj_matrices = [item[1] for item in batch]
    labels = torch.stack([torch.tensor(item[2]) for item in batch])
    return data_inputs, adj_matrices, labels

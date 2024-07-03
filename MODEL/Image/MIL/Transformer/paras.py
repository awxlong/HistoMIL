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
class TransformerParas:
    """
    include all parameters to create a Transformer
    """
    #------> parameters for model
    encoder_name="pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False
    
    num_classes:int = 1
    input_dim:int = 1024           # by default we'll be using uni's feature vectors which are of size 1024
    pretrained_input_dim:int = 768 # pretrained transformer used ctranspath by default for feature encoding
    dim:int = 512
    depth:int = 2
    heads:int = 8
    mlp_dim:int = 512
    pool:str = 'cls'
    dim_head:int = 64
    dropout:float = 0.
    emb_dropout:float = 0.
    pos_enc:nn.Module = None
    pretrained_weights_dir = ''
    pretrained_weights: str = attr.ib(default=None, validator=attr.validators.optional(attr.validators.in_(AVAILABLE_WEIGHTS)))
    encoder_name:str = 'pre-calculated' # by default we'll be using the foundational models for feature extraction, so we avoid SSL
    task:str = 'binary'
    criterion:str = 'BCEWithLogitsLoss'
    pos_weight = torch.ones(1)
    lr:float = 2.0e-05
    wd:float = 2.0e-05
    optimizer = 'AdamW'
    lr_scheduler = 'CosineAnnealingLR'
    lr_scheduler_config = {'T_max':50, 'eta_min':1e-6}

    def __attrs_post_init__(self):
        super().__init__()
        # pdb.set_trace()
        if self.pretrained_weights is not None:
            print(f"Using pretrained weights: {self.pretrained_weights}")
            # Load the pretrained weights here
        else:
            print("No pretrained weights specified. Initializing with random weights.")



def get_loss(name, **kwargs):
    # Check if the name is a valid loss name
    if name in nn.__dict__:
        # Get the loss class from the torch.nn module
        loss_class = getattr(nn, name)
        # Instantiate the loss with the reduction option
        loss = loss_class(**kwargs)
        # Return the loss
        return loss
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid loss name: {name}")
    


def get_optimizer(name, model, lr=0.01, wd=0.1):
    # Check if the name is a valid optimizer name
    if name in optim.__dict__:
        # Get the optimizer class from the torch.optim module
        optimizer_class = getattr(optim, name)
        # Instantiate the optimizer with the model parameters and the learning rate
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=wd)
        # Return the optimizer
        # pdb.set_trace()
        return optimizer
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid optimizer name: {name}")


def get_scheduler(name, optimizer, optim_config):
    # Check if the name is a valid scheduler name
    if name in lr_scheduler.__dict__:
        # Get the scheduler class from the torch.optim.lr_scheduler module
        scheduler_class = getattr(lr_scheduler, name)
        # Instantiate the scheduler with the optimizer and other keyword arguments
        scheduler = scheduler_class(optimizer, **optim_config)
        # Return the scheduler
        # pdb.set_trace()
        return scheduler
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid scheduler name: {name}")


DEFAULT_TRANSFORMER_PARAS = TransformerParas(input_dim=1024, task='binary', \
                                             pretrained_weights='MSI_high_CRC_model.pth', encoder_name='pre-calculated')

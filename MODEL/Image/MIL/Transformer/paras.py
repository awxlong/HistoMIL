"""
pre-defined parameters for (pretrained) transformer model loaded from https://github.com/peng-lab/HistoBistro/tree/main
"""

import attr 

#---->

AVAILABLE_WEIGHTS = ['BRAF_CRC_model.pth', 'KRAS_CRC_model.pth', 'MSI_high_CRC_model.pth']

@attr.s(auto_attribs=True)
class TransformerParas:
    """
    include all parameters to create a Transformer
    """
    #------> parameters for model
    encoder_name="pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False
    heads:int = 8
    dim_head:int = 64
    mlp_dim:int = 512
    input_dim:int = 768 # pretrained transformer used ctranspath by default for feature encoding
    num_classes:int = 1
    pretrained_weights: str = attr.ib(default=None, validator=attr.validators.optional(attr.validators.in_(AVAILABLE_WEIGHTS)))
    
    def __attrs_post_init__(self):
        super().__init__()
        
        if self.pretrained_weights is not None:
            print(f"Using pretrained weights: {self.pretrained_weights}")
            # Load the pretrained weights here
        else:
            print("No pretrained weights specified. Initializing with random weights.")
        
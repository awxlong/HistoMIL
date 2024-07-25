"""
Transformer with pretrained weights originally implemented from HistoBistro:
https://github.com/peng-lab/HistoBistro/tree/main
"""
import os
import sys
sys.path.append('/Users/awxlong/Desktop/my-studies/hpc_exps/')
import numpy as np
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
#-----------> external network modules 
from HistoMIL.MODEL.Image.MIL.utils import Attention, FeedForward, PreNorm, FeatureNet
from HistoMIL.MODEL.Image.MIL.Transformer.paras import TransformerParas
# from HistoMIL import logger

import pdb
class BaseAggregator(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

class TransformerBlocks(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                    ]
                )
            )

    def forward(self, x, register_hook=False):
        for attn, ff in self.layers:
            x = attn(x, register_hook=register_hook) + x
            x = ff(x) + x
        return x


class Transformer(BaseAggregator):
    def __init__(
        self,
        paras:TransformerParas
    ):
        super(BaseAggregator, self).__init__()
        assert paras.pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (class token) or mean (mean pooling)'
        
        self.transformer_paras = paras
        self.projection = nn.Sequential(nn.Linear(paras.pretrained_input_dim, paras.heads * paras.dim_head, bias=True), nn.ReLU())
        self.mlp_head = nn.Sequential(nn.LayerNorm(paras.mlp_dim), nn.Linear(paras.mlp_dim, paras.num_classes))
        self.transformer = TransformerBlocks(paras.dim, paras.depth, paras.heads, paras.dim_head, paras.mlp_dim, paras.dropout)

        self.pool = paras.pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, paras.dim))

        self.norm = nn.LayerNorm(paras.dim)
        self.dropout = nn.Dropout(paras.emb_dropout)
        
        self.pos_enc = paras.pos_enc

        if paras.pretrained_weights:
            self.load_pretrained_weights()
            print(f"Successfully loaded pretrained weights {paras.pretrained_weights}")
        
        #--------> feature encoder
        self.encoder = FeatureNet(paras.encoder_name)

        # Add a projection layer if input_dim != pretrained networks' input_dim
        if paras.input_dim != paras.pretrained_input_dim:
            self.input_projection = nn.Sequential(nn.Linear(paras.input_dim, paras.heads * paras.dim_head, bias=True), nn.ReLU()) # nn.Linear(paras.input_dim, paras.pretrained_input_dim)
        else:
            self.input_projection = nn.Identity()

        if paras.selective_finetuning:
            self.selected_layers_finetuning()

    def load_pretrained_weights(self):
        pretrained_weights_dir =  self.transformer_paras.pretrained_weights_dir# 'pretrained_weights/'
        state_dict = torch.load(f'{pretrained_weights_dir}{self.transformer_paras.pretrained_weights}')
        # Remove the 'model.' prefix from the keys
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '')
            new_state_dict[new_key] = v
        new_state_dict.pop('criterion.pos_weight', None)
        # pdb.set_trace()
        self.load_state_dict(new_state_dict, strict=True)

    def selected_layers_finetuning(self):
        for name, param in self.named_parameters():
            if not any(layer in name for layer in ['mlp_head', 'input_projection']):
                param.requires_grad = False
    def print_trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")
            else:
                print(f"Frozen: {name}")
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, coords=None, register_hook=False):
        
        b, _, _ = x.shape
        #--------> feature encoder
        # pdb.set_trace()
        x = self.encoder(x)

        # Project input if necessary
        x = self.input_projection(x)
        
        # x = self.projection(x)
        
        if self.pos_enc:
            x = x + self.pos_enc(coords)

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)
        x = self.transformer(x, register_hook=register_hook) # (#batch_size 1, #patches + 1 421, weight_dim 256)
        pdb.set_trace()
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # pdb.set_trace()
        return self.mlp_head(self.norm(x))


if __name__ == "__main__":
    # model_config = {'heads': 8, 
    #             'dim_head': 64, 
    #             'dim': 512, 
    #             'mlp_dim': 512, 
    #             'input_dim':768,
    #             'num_classes':1}
    default_paras = TransformerParas(input_dim=1024, \
                                    # pretrained_weights_dir='/Users/awxlong/Desktop/my-studies/hpc_exps/HistoMIL/MODEL/Image/MIL/Transformer/pretrained_weights/',
                                    pretrained_weights=None, \
                                    encoder_name='pre-calculated')
    default_paras.pretrained_weights_dir = ''
    default_paras.selective_finetuning = False
    
    model = Transformer(default_paras)
    rand_tensor = torch.rand((1, 420, 1024))
    y = model(rand_tensor)

    pdb.set_trace()
    
    # pdb.set_trace()
    # Load the modified state dictionary into your model
    # model.load_state_dict(new_state_dict)
    # pdb.set_trace()

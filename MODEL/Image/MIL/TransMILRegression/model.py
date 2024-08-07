"""
Implementation of abmil with/without gradient accumulation

most code copy from 
https://github.com/szc19990412/TransMIL
"""
import numpy as np
import sys
sys.path.append('/Users/awxlong/Desktop/my-studies/hpc_exps/')
import torch
import torch.nn as nn
import torch.nn.functional as F
#-----------> external network modules 
# from HistoMIL.MODEL.Image.Modules.NystromAttention import NystromAttention
from HistoMIL.MODEL.Image.MIL.utils import FeatureNet, BaseAggregator, PPEG, NystromTransformerLayer
from HistoMIL.MODEL.Image.MIL.TransMILRegression.paras import TransMILRegressionParas
from HistoMIL import logger
######################################################################################
#        pytorch module to define structure
######################################################################################

import pdb


class TransMILRegression(BaseAggregator):
    def __init__(self, paras:TransMILRegressionParas):
        super(BaseAggregator, self).__init__()
        
        self.paras = paras
        self.pos_layer = PPEG(dim=512)
        self.pos_enc = paras.pos_enc
        self.encoder = FeatureNet(model_name = paras.encoder_name)
        print(f'Using {self.pos_enc} positional encoding')
        self._fc1 = nn.Sequential(nn.Linear(paras.input_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.num_classes = paras.num_classes
        self.layer1 = NystromTransformerLayer(dim = 512)
        self.layer2 = NystromTransformerLayer(dim = 512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, paras.num_classes)
        

    def forward(self, x, coords=None):

        h = self.encoder(x) #[B, n, 1024]

        # h = x  

        h = self._fc1(h)  #[B, n, 512]

        #----> padding
        H = h.shape[1]
        if self.pos_enc == 'PPEG':
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))  # find smallest square larger than n
            add_length = _H * _W - H  # add N - n, first entries of feature vector added at the end to fill up until square number
            h = torch.cat([h, h[:, :add_length, :]], dim=1)  #[B, N, 512]
        elif self.pos_enc == 'PPEG_padded':  # only works with batch size 1 so far
            if h.shape[1] > 1:  # patient TCGA-A6-2675 has only one patch
                dimensions = coords.max(dim=1).values - coords.min(dim=1).values
                x_coords = coords[:, :, 1].unique(dim=1)  # assumes quadratic patches
                patch_size = (x_coords[:, 1:] - x_coords[:, :-1]).min(dim=-1).values
                offset = coords[:, 0, :] % patch_size
                dimensions_grid = ((dimensions - offset) / patch_size).squeeze(0) + 1
                _H, _W = dimensions_grid.int().tolist()
                base_grid = torch.zeros((h.shape[0], dimensions_grid[0].int().item(), dimensions_grid[1].int().item(), h.shape[-1]), device=h.device)
                grid_indices = (coords - offset.unsqueeze(1) - coords.min(dim=1).values.unsqueeze(1)) / patch_size
                grid_indices = grid_indices.long().cpu()
                base_grid[:, grid_indices[:, :, 0], grid_indices[:, :, 1]] = h.squeeze(0)
                h = base_grid.reshape((h.shape[0], -1, h.shape[-1]))
            else:
                _H, _W = 1, 1

        #----> cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #----> first translayer
        h = self.layer1(h)  #[B, N, 512]

        #----> ppeg
        h = self.pos_layer(h, _H, _W)  #[B, N, 512]

        #----> second translayer
        h = self.layer2(h)  #[B, N, 512]

        #----> cls_token
        h = self.norm(h)[:, 0]

        #----> predict
        logits = self._fc2(h)  #[B, n_classes]

        return logits
    
    def infer(self, x, coords=None):

        h = self.encoder(x) #[B, n, 1024]

        # h = x  

        h = self._fc1(h)  #[B, n, 512]

        #----> padding
        H = h.shape[1]
        if self.pos_enc == 'PPEG':
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))  # find smallest square larger than n
            add_length = _H * _W - H  # add N - n, first entries of feature vector added at the end to fill up until square number
            h = torch.cat([h, h[:, :add_length, :]], dim=1)  #[B, N, 512]
        elif self.pos_enc == 'PPEG_padded':  # only works with batch size 1 so far
            if h.shape[1] > 1:  # patient TCGA-A6-2675 has only one patch
                dimensions = coords.max(dim=1).values - coords.min(dim=1).values
                x_coords = coords[:, :, 1].unique(dim=1)  # assumes quadratic patches
                patch_size = (x_coords[:, 1:] - x_coords[:, :-1]).min(dim=-1).values
                offset = coords[:, 0, :] % patch_size
                dimensions_grid = ((dimensions - offset) / patch_size).squeeze(0) + 1
                _H, _W = dimensions_grid.int().tolist()
                base_grid = torch.zeros((h.shape[0], dimensions_grid[0].int().item(), dimensions_grid[1].int().item(), h.shape[-1]), device=h.device)
                grid_indices = (coords - offset.unsqueeze(1) - coords.min(dim=1).values.unsqueeze(1)) / patch_size
                grid_indices = grid_indices.long().cpu()
                base_grid[:, grid_indices[:, :, 0], grid_indices[:, :, 1]] = h.squeeze(0)
                h = base_grid.reshape((h.shape[0], -1, h.shape[-1]))
            else:
                _H, _W = 1, 1

        #----> cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #----> first translayer
        h = self.layer1(h)  #[B, N, 512]

        #----> ppeg
        h = self.pos_layer(h, _H, _W)  #[B, N, 512]

        #----> second translayer
        h = self.layer2(h)  #[B, N, 512]
        A_raw = h
        # pdb.set_trace()
        
        #----> cls_token
        h = self.norm(h)[:, 0] # [B, 512]

        #----> predict
        logits = self._fc2(h)  #[B, n_classes]
        Y_hat = (logits <= self.paras.threshold).int()# for regression, we binarize based on biological threshold
        Y_prob = F.softmax(logits, dim = 1)

        return logits, Y_prob, Y_hat, A_raw
    

if __name__ == "__main__":
    
    default_paras = TransMILRegressionParas()
    rand_tensor = torch.rand(1, 1, 1024) 
    model = TransMILRegression(default_paras)

    pdb.set_trace()

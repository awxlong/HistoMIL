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
from HistoMIL.MODEL.Image.MIL.utils import FeatureNet, BaseAggregator, PPEG, NystromTransformerLayer
from HistoMIL.MODEL.Image.MIL.DTFDTransMIL.paras import DTFDTransMILParas
from HistoMIL import logger
######################################################################################
#        pytorch module to define structure
######################################################################################

import pdb


### IMPLEMENTATION COPIED FROM https://github.com/Dootmaan/DTFD-MIL.PyTorch/blob/main/train_DTFT-MIL.py
### OR https://github.com/hrzhang1123/DTFD-MIL/blob/main/Main_DTFD_MIL.py


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x
class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N

class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred
    
class TransMIL(BaseAggregator):
    def __init__(self, paras:DTFDTransMILParas):
        super(BaseAggregator, self).__init__()
        
        self.paras = paras
        self.pos_layer = PPEG(dim=512)
        self.pos_enc = paras.pos_enc
        self.encoder = FeatureNet(model_name = paras.encoder_name)
        print(f'Using {self.pos_enc} positional encoding')
        self._fc1 = nn.Sequential(nn.Linear(paras.input_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.num_classes = paras.num_cls
        self.layer1 = NystromTransformerLayer(dim = 512)
        self.layer2 = NystromTransformerLayer(dim = 512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, paras.num_cls)
        

    def forward(self, x, coords=None):

        h = self.encoder(x) #[B, n, 1024]
        # pdb.set_trace()
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

        return h, logits

class DTFDTransMIL(BaseAggregator):
    def __init__(self, paras:DTFDTransMILParas):
        super(BaseAggregator, self).__init__()
        
        self.paras = paras
        
        ### model components
        # self.classifier = Classifier_1fc(paras.mDim, paras.num_cls, paras.droprate)# .to(paras.device)
        # self.attention = Attention_Gated(paras.mDim).to(paras.device)# .to(paras.device)
        # self.dimReduction = DimReduction(paras.input_dim, paras.mDim, numLayer_Res=paras.numLayer_Res)# .to(paras.device)
        self.attCls = Attention_with_Classifier(L=paras.mDim, num_cls=paras.num_cls, droprate=paras.droprate_2)# .to(paras.device)
        self.TransMIL = TransMIL(paras=self.paras)
        # self.attCls = self.attCls.to(paras.device)
        # self.classifier = self.classifier.to(paras.device)
        # self.attention = self.attention.to(paras.device)
        # self.dimReduction = self.dimReduction.to(paras.device)
        
    def forward(self, x):
        inputs, labels = x
        
        # inputs = inputs.to(self.paras.device)
        # labels = labels.to(self.paras.device)
        # inputs= self.encoder(x) #[B, n, 1024]

        slide_sub_preds=[]
        slide_sub_labels=[]
        slide_pseudo_feat=[]
        inputs_pseudo_bags=torch.chunk(inputs.squeeze(0), self.paras.numGroup,dim=0)
        # inputs_pseudo_bags = [chunk.to(self.paras.device) for chunk in inputs_pseudo_bags]
        # pdb.set_trace()
        
        for subFeat_tensor in inputs_pseudo_bags:
            if subFeat_tensor.dim() == 2:
                subFeat_tensor = subFeat_tensor.unsqueeze(0)
            slide_sub_labels.append(labels)
            # pdb.set_trace()
            # tmidFeat = self.dimReduction(subFeat_tensor)
            # tAA = self.attention(tmidFeat).squeeze(0)
            # tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            # tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            # tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2
            
            tattFeat_tensor, tPredict = self.TransMIL(subFeat_tensor)
            slide_sub_preds.append(tPredict)
            slide_pseudo_feat.append(tattFeat_tensor)
            

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup

        gSlidePred = self.attCls(slide_pseudo_feat)
        # pdb.set_trace()
        return slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred

if __name__ == "__main__":
    
    default_paras = DTFDTransMILParas()
    default_paras.input_dim = 1024
    rand_tensor = torch.rand(1, 42, default_paras.input_dim).to('mps')
    model = DTFDTransMIL(default_paras).to('mps')
    label = torch.tensor(1).unsqueeze(0).to('mps')
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # rand_tensor
    slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred = model([rand_tensor, label])
    criterion(slide_sub_preds, slide_sub_labels)
    # pdb.set_trace()
    
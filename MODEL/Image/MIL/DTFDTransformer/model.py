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
from HistoMIL.MODEL.Image.MIL.Transformer.model import TransformerBlocks

from HistoMIL.MODEL.Image.MIL.DTFDTransformer.paras import DTFDTransformerParas
from HistoMIL import logger
######################################################################################
#        pytorch module to define structure
######################################################################################

import pdb
from einops import repeat



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
    
class Transformer(BaseAggregator):
    def __init__(
        self,
        paras:DTFDTransformerParas
    ):
        super(BaseAggregator, self).__init__()
        assert paras.pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (class token) or mean (mean pooling)'
        
        self.transformer_paras = paras
        self.projection = nn.Sequential(nn.Linear(paras.input_dim, paras.heads * paras.dim_head, bias=True), nn.ReLU())
        self.mlp_head = nn.Sequential(nn.LayerNorm(paras.mlp_dim), nn.Linear(paras.mlp_dim, paras.num_classes))
        self.transformer = TransformerBlocks(paras.dim, paras.depth, paras.heads, paras.dim_head, paras.mlp_dim, paras.dropout)

        self.pool = paras.pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, paras.dim))

        self.norm = nn.LayerNorm(paras.dim)
        self.dropout = nn.Dropout(paras.emb_dropout)
        
        self.pos_enc = paras.pos_enc

        
        #--------> feature encoder
        self.encoder = FeatureNet(paras.encoder_name)


        if paras.selective_finetuning:
            self.selected_layers_finetuning()


    def selected_layers_finetuning(self):
        for name, param in self.named_parameters():
            if not any(layer in name for layer in ['mlp_head', 'input_projection']):
                param.requires_grad = False
    
    def forward(self, x, coords=None, register_hook=False):
        
        b, _, _ = x.shape
        #--------> feature encoder
        # pdb.set_trace()
        x = self.encoder(x)

        
        x = self.projection(x)
        
        if self.pos_enc:
            x = x + self.pos_enc(coords)

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)
        x = self.transformer(x, register_hook=register_hook) # (#batch_size 1, #patches + 1 421, weight_dim 256)
        # pdb.set_trace()
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # pdb.set_trace()
        h, logits = self.norm(x),  self.mlp_head(self.norm(x))
        return h, logits

class DTFDTransformer(BaseAggregator):
    def __init__(self, paras:DTFDTransformerParas):
        super(BaseAggregator, self).__init__()
        
        self.paras = paras
        
        ### model components
        # self.classifier = Classifier_1fc(paras.mDim, paras.num_cls, paras.droprate)# .to(paras.device)
        # self.attention = Attention_Gated(paras.mDim).to(paras.device)# .to(paras.device)
        # self.dimReduction = DimReduction(paras.input_dim, paras.mDim, numLayer_Res=paras.numLayer_Res)# .to(paras.device)
        self.attCls = Attention_with_Classifier(L=paras.mDim, num_cls=paras.num_cls, droprate=paras.droprate_2)# .to(paras.device)
        self.transformer = Transformer(paras=self.paras)
        
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
            
            tattFeat_tensor, tPredict = self.transformer(subFeat_tensor)
            
            slide_sub_preds.append(tPredict)
            slide_pseudo_feat.append(tattFeat_tensor)
            

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup

        gSlidePred = self.attCls(slide_pseudo_feat)
        # pdb.set_trace()
        return slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred

if __name__ == "__main__":
    
    default_paras = DTFDTransformerParas()
    default_paras.input_dim = 1024
    rand_tensor = torch.rand(1, 42, default_paras.input_dim).to('mps')
    model = DTFDTransformer(default_paras).to('mps')
    label = torch.tensor(1).unsqueeze(0).to('mps')
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # rand_tensor
    slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred = model([rand_tensor, label])
    criterion(slide_sub_preds, slide_sub_labels)
    # pdb.set_trace()
    
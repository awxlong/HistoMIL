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
from HistoMIL.MODEL.Image.MIL.DTFD_MIL.paras import DTFD_MILParas
from HistoMIL import logger
######################################################################################
#        pytorch module to define structure
######################################################################################

import pdb


### IMPLEMENTATION COPIED FROM https://github.com/Dootmaan/DTFD-MIL.PyTorch/blob/main/train_DTFT-MIL.py
### OR https://github.com/hrzhang1123/DTFD-MIL/blob/main/Main_DTFD_MIL.py
### TODO:

from model.network import Classifier_1fc, DimReduction
from model.Attention import Attention_Gated as Attention
from model.Attention import Attention_with_Classifier

class DTFD_MIL(BaseAggregator):
    def __init__(self, paras:DTFD_MILParas):
        super(BaseAggregator, self).__init__()
        
        self.paras = paras
        
        ### model components
        self.classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
        self.attention = Attention(params.mDim).to(params.device)
        self.dimReduction = DimReduction(1024, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
        self.attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)

    def forward(self, x):
        inputs, labels = x
        # inputs= self.encoder(x) #[B, n, 1024]

    
        # TODO: LUBA HELP
        # I THINK forward() should return slide_pseudo_feat,slide_sub_preds, slide_sub_labels,
        # LINES 149 ONWARDS
        slide_sub_preds=[]
        slide_sub_labels=[]
        slide_pseudo_feat=[]
        inputs_pseudo_bags=torch.chunk(inputs.squeeze(0), self.paras.numGroup,dim=0)

        for subFeat_tensor in inputs_pseudo_bags:

            slide_sub_labels.append(labels)
            # subFeat_tensor=subFeat_tensor.to(params.device)
            # subFeat_tensor = torch.index_select(inputs_pseudo_bags, dim=0, index=torch.LongTensor(tindex).to(params.device))
            tmidFeat = self.dimReduction(subFeat_tensor)
            tAA = self.attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)
            slide_pseudo_feat.append(tattFeat_tensor)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup

        return slide_pseudo_feat, slide_sub_preds, slide_sub_labels

if __name__ == "__main__":
    
    default_paras = DTFD_MILParas()
    rand_tensor = torch.rand(1, 1, 1024) 
    model = DTFD_MIL(default_paras)

    pdb.set_trace()
    
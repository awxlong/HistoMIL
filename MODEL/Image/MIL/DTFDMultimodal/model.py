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

class DTFD_MIL(BaseAggregator):
    def __init__(self, paras:DTFD_MILParas):
        super(BaseAggregator, self).__init__()
        
        self.paras = paras
        
        ### model components
        self.classifier = Classifier_1fc(paras.mDim, paras.num_cls, paras.droprate)# .to(paras.device)
        self.attention = Attention_Gated(paras.mDim).to(paras.device)# .to(paras.device)
        self.dimReduction = DimReduction(paras.input_dim, paras.mDim, numLayer_Res=paras.numLayer_Res)# .to(paras.device)
        self.attCls = Attention_with_Classifier(L=paras.mDim, num_cls=paras.num_cls, droprate=paras.droprate_2)# .to(paras.device)
        # self.attCls = self.attCls.to(paras.device)
        # self.classifier = self.classifier.to(paras.device)
        # self.attention = self.attention.to(paras.device)
        # self.dimReduction = self.dimReduction.to(paras.device)
        
    def forward(self, x):
        inputs, clinical_features, labels = x

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

            slide_sub_labels.append(labels)

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
        
        
        #### MULTIMODAL FUSION OCCURS HERE
        clinical_embedding = self.clinical_encoding(clinical_features)
        
        # Attention gates on each modality.
        h, clinical_embedding = self.attn_modalities(h, clinical_embedding)
        
        # Post-compressiong H&E slide embedding. # reduce dim of H&E embedding to avoid overload from kronecker delta
        # h = self.post_compression_layer_he(h)

        # Fusion, default kronecker delta fusion
        m = self.forward_fusion(h, clinical_embedding)
        
        # Post-compression of multimodal embedding.
        m = self.post_compression_layer(m)

        gSlidePred = self.attCls(m)# self.attCls(slide_pseudo_feat)

        return slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred

if __name__ == "__main__":
    
    default_paras = DTFD_MILParas()
    default_paras.input_dim = 2048
    rand_tensor = torch.rand(1, 42, default_paras.input_dim).to('mps')
    model = DTFD_MIL(default_paras)
    label = torch.tensor(1).unsqueeze(0).to('mps')
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # rand_tensor
    slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred = model([rand_tensor, label])
    criterion(slide_sub_preds, slide_sub_labels)
    pdb.set_trace()
    
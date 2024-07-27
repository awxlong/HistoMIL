"""
Implementation of abmil with/without gradient accumulation

most code copy from 
https://github.com/szc19990412/TransMIL and https://github.com/Dootmaan/DTFD-MIL.PyTorch/blob/main/train_DTFT-MIL.py
"""
import numpy as np
import sys
sys.path.append('/Users/awxlong/Desktop/my-studies/hpc_exps/')
import torch
import torch.nn as nn
import torch.nn.functional as F
#-----------> external network modules 
from HistoMIL.MODEL.Image.MIL.utils import FeatureNet, BaseAggregator, PPEG, NystromTransformerLayer
from HistoMIL.MODEL.Image.MIL.DTFDTransMILMultimodal.paras import DTFDTransMILMultimodalParas
from HistoMIL.MODEL.Image.MIL.TransMILMultimodal.model import FeatureEncoding, Attn_Modality_Gated, FC_block
from HistoMIL.MODEL.Image.MIL.DTFD_MIL.model import Attention_with_Classifier

from HistoMIL import logger
######################################################################################
#        pytorch module to define structure
######################################################################################

import pdb


### IMPLEMENTATION COPIED FROM https://github.com/Dootmaan/DTFD-MIL.PyTorch/blob/main/train_DTFT-MIL.py
### OR https://github.com/hrzhang1123/DTFD-MIL/blob/main/Main_DTFD_MIL.py
    
class TransMILMultimodal(BaseAggregator):
    def __init__(self, paras:DTFDTransMILMultimodalParas):
        super(BaseAggregator, self).__init__()
        
        self.paras = paras

        ### H&E encoding
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
        
        # self.post_compression_layer_he = FC_block(512, 512//2)
        ### Clinical feature encoding
        self.clinical_encoding = FeatureEncoding(idx_continuous=self.paras.idx_continuous,
                                                 taxonomy_in=self.paras.taxonomy_in,
                                                 embedding_dim=self.paras.embed_size)
        
        ### Gated attention with dropout for kronecker delta fusion
        self.attn_modalities = Attn_Modality_Gated(dim1_og=512, 
                                                   dim2_og=paras.embed_size // 2, # 64//2
                                                   scale=self.paras.scale) # 512 // 2, 32

        ### post-fusion compression
        if self.paras.fusion_type=='bilinear':
            head_size_in = (512 // self.paras.scale[0] + 1) * (self.paras.embed_size//2 + 1)
        elif self.paras.fusion_type=='kron':
            head_size_in = (512 // self.paras.scale[0]) * (self.paras.embed_size//2) # e.g. 256 * 32
        elif self.paras.fusion_type=='concat':
            head_size_in = 512 // self.paras.scale[0] + self.paras.embed_size//2

        self.post_compression_layer = nn.Sequential(*[FC_block(head_size_in, 1024),
                                                      FC_block(1024, 256),
                                                      FC_block(256, 128)])


        self.classifer = nn.Linear(128, paras.num_classes)
        
    def forward_fusion(self, h1, h2):

        if self.paras.fusion_type=='bilinear':
            # Append 1 to retain unimodal embeddings in the fusion
            h1 = torch.cat((h1, torch.ones(1, 1, dtype=torch.float, device=h1.device)), -1)
            h2 = torch.cat((h2, torch.ones(1, 1, dtype=torch.float, device=h2.device)), -1)
            

            return torch.kron(h1, h2)

        elif self.paras.fusion_type=='kron':
            return torch.kron(h1, h2)

        elif self.paras.fusion_type=='concat':
            return torch.cat([h1, h2], dim=-1)
        else:
            print('Not implemeted') 
            #raise Exception ... 
    
    def forward(self, x, clinical_features, coords=None):

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
        h = self.norm(h)[:, 0] # [B, 512]

        
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

        #----> predict
        logits = self.classifer(m)  #[B, n_classes]
        # pdb.set_trace()
        return m, logits
    

class DTFDTransMILMultimodal(BaseAggregator):
    def __init__(self, paras:DTFDTransMILMultimodalParas):
        super(BaseAggregator, self).__init__()
        
        self.paras = paras
        
        ### model components
        self.attCls = Attention_with_Classifier(L=paras.mDim, num_cls=paras.num_cls, droprate=paras.droprate_2)# .to(paras.device)
        self.TransMILMultimodal = TransMILMultimodal(paras=self.paras)
        
    def forward(self, x):
        inputs, clinical_feats, _, labels = x
        
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
            
            tattFeat_tensor, tPredict = self.TransMILMultimodal(subFeat_tensor, clinical_feats)
            slide_sub_preds.append(tPredict)
            slide_pseudo_feat.append(tattFeat_tensor)
            

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup

        gSlidePred = self.attCls(slide_pseudo_feat)
        # pdb.set_trace()
        return slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred

if __name__ == "__main__":
    
    default_paras = DTFDTransMILMultimodalParas()
    default_paras.input_dim = 1024
    rand_tensor = torch.rand(1, 42, default_paras.input_dim).to('mps')
    model = DTFDTransMILMultimodal(default_paras).to('mps')
    label = torch.tensor(1).unsqueeze(0).to('mps')
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # rand_tensor
    slide_pseudo_feat, slide_sub_preds, slide_sub_labels, gSlidePred = model([rand_tensor, label])
    criterion(slide_sub_preds, slide_sub_labels)
    # pdb.set_trace()
    
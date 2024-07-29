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
from HistoMIL.MODEL.Image.MIL.utils import Attention, FeedForward, PreNorm, FeatureNet, FC_block, FeatureEncoding, Attn_Modality_Gated
from HistoMIL.MODEL.Image.MIL.TransformerMultimodal.paras import TransformerMultimodalParas
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


class TransformerMultimodal(BaseAggregator):
    def __init__(
        self,
        paras:TransformerMultimodalParas
    ):
        super(BaseAggregator, self).__init__()
        assert paras.pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (class token) or mean (mean pooling)'
        
        self.paras = paras
        self.projection = nn.Sequential(nn.Linear(paras.input_dim, paras.heads * paras.dim_head, bias=True), nn.ReLU())
        self.mlp_head = nn.Sequential(nn.LayerNorm(paras.mlp_dim), nn.Linear(paras.mlp_dim, paras.num_classes))
        self.transformer = TransformerBlocks(paras.dim, paras.depth, paras.heads, paras.dim_head, paras.mlp_dim, paras.dropout)

        self.pool = paras.pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, paras.dim))

        self.norm = nn.LayerNorm(paras.dim)
        self.dropout = nn.Dropout(paras.emb_dropout)
        
        self.pos_enc = paras.pos_enc

        
        self.encoder = FeatureNet(paras.encoder_name)

        # # Add a projection layer if input_dim != pretrained networks' input_dim
        # if paras.input_dim != paras.pretrained_input_dim:
        #     self.input_projection = nn.Sequential(nn.Linear(paras.input_dim, paras.heads * paras.dim_head, bias=True), nn.ReLU()) # nn.Linear(paras.input_dim, paras.pretrained_input_dim)
        # else:
        #     self.input_projection = nn.Identity()

        if paras.selective_finetuning:
            self.selected_layers_finetuning()

        ### Clinical feature encoding
        self.clinical_encoding = FeatureEncoding(idx_continuous=self.paras.idx_continuous,
                                                 taxonomy_in=self.paras.taxonomy_in,
                                                 embedding_dim=self.paras.embed_size)
        
        ### Gated attention with dropout for kronecker delta fusion
        self.attn_modalities = Attn_Modality_Gated(dim1_og=256, 
                                                   dim2_og=paras.embed_size // 2, # 64//2
                                                   scale=self.paras.scale) # 512 // 2, 32

        ### post-fusion compression
        if self.paras.fusion_type=='bilinear':
            head_size_in = (256 // self.paras.scale[0] + 1) * (self.paras.embed_size//2 + 1)
        elif self.paras.fusion_type=='kron':
            head_size_in = (256 // self.paras.scale[0]) * (self.paras.embed_size//2) # e.g. 256 * 32
        elif self.paras.fusion_type=='concat':
            head_size_in = 256 // self.paras.scale[0] + self.paras.embed_size//2

        self.post_compression_layer = nn.Sequential(*[FC_block(head_size_in, 512),
                                                      FC_block(512, 128)])


        self.classifer = nn.Linear(128, paras.num_classes)
        
    
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

    def forward(self, x, clinical_features, coords=None, register_hook=False):
        
        b, _, _ = x.shape
        #--------> feature encoder
        # pdb.set_trace()
        x = self.encoder(x)

        # Project input if necessary
        # x = self.input_projection(x)
        
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
        h = self.norm(x)

        #### MULTIMODAL FUSION OCCURS HERE
        clinical_embedding = self.clinical_encoding(clinical_features)
        
        # pdb.set_trace()
        # Attention gates on each modality.
        h, clinical_embedding = self.attn_modalities(h, clinical_embedding)
        
        # Post-compressiong H&E slide embedding. # reduce dim of H&E embedding to avoid overload from kronecker delta
        # h = self.post_compression_layer_he(h)

        # Fusion, default kronecker delta fusion
        m = self.forward_fusion(h, clinical_embedding)
        # pdb.set_trace()
        # Post-compression of multimodal embedding.
        m = self.post_compression_layer(m)

        logits = self.classifer(m)  #[B, n_classes]
        # pdb.set_trace()
        return logits


if __name__ == "__main__":
    
    default_paras = TransformerMultimodalParas()
    
    
    model = TransformerMultimodal(default_paras)
    rand_img_tensor = torch.rand(1, 1, 1024) 
    rand = torch.Tensor([0.5, 0.8, 0.9, 1, 0, 1, 1, 0])
    # binary = (rand >= 0.5).long()
    
    encoder = FeatureEncoding()
    attn_modality = Attn_Modality_Gated()
    y = encoder(rand)
    # y = attn_modality(rand_img_tensor, y)
    temp = model(rand_img_tensor, rand)
    
    pdb.set_trace()
    
    # pdb.set_trace()
    # Load the modified state dictionary into your model
    # model.load_state_dict(new_state_dict)
    # pdb.set_trace()

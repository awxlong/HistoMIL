"""
Implementation of TransMIL with multimodal fusion via late fusion
https://github.com/szc19990412/TransMIL and https://github.com/AIRMEC/HECTOR/blob/main/model.py and https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from HistoMIL.MODEL.Image.MIL.utils import FeatureNet, BaseAggregator, PPEG, NystromTransformerLayer
from HistoMIL.MODEL.Image.MIL.TransMILMultimodal.paras import TransMILMultimodalParas


import pdb

class FC_block(nn.Module):
    '''
    helper class to reduce dimensionality of H&E feature embedding
    '''
    def __init__(self, dim_in, dim_out, act_layer=nn.ReLU, dropout=True, p_dropout_fc=0.25):
        super(FC_block, self).__init__()

        self.fc = nn.Linear(dim_in, dim_out)
        self.act = act_layer()
        self.drop = nn.Dropout(p_dropout_fc) if dropout else nn.Identity()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x
    
    
class FeatureEncoding(nn.Module):
    def __init__(self, idx_continuous=27, embedding_dim=64, depth=1, act_fct='relu', dropout=True, p_dropout=0.25):
        super().__init__()
        '''
        args:
        idx_continuous:int - if your feature vector has a mix of continuous and categorical (binary) features, specify the upper bound idx of continuous features, and the rest of the feature vector are categorical (binary) features
        taxonomy_in:int - number of categories, e.g., 2 for binary

        returns:
        a joint continuous-categorical feature embedding for clinical features
        
        '''

        self.idx_continuous = idx_continuous
        act_fcts = {'relu': nn.ReLU(),
        'elu' : nn.ELU(),
        'tanh': nn.Tanh(),
        'selu': nn.SELU(),
        }
        dropout_module = nn.AlphaDropout(p_dropout) if act_fct=='selu' else nn.Dropout(p_dropout)

        # self.categorical_embedding = nn.Embedding(taxonomy_in, embedding_dim)
        self.continuous_embedding = nn.Linear(self.idx_continuous, embedding_dim)
        # fc_layers_categorical = []
        fc_layers_continuous = []
        for d in range(depth):
            ### forward continuous
            fc_layers_continuous.append(nn.Linear(embedding_dim//(2**d), embedding_dim//(2**(d+1))))
            fc_layers_continuous.append(dropout_module if dropout else nn.Identity())
            fc_layers_continuous.append(act_fcts[act_fct])
            

        self.fc_layers_continuous = nn.Sequential(*fc_layers_continuous)
    def forward(self, x):
        '''
        x is a tensor consisting of a mix of continuous and categorical features.  
        '''
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if input is 1D
        
        # continuous_x, categorical_x = x[:, :self.idx_continuous], x[:, self.idx_continuous:] 
        # pdb.set_trace()
        continuous_x = self.continuous_embedding(x)
        continuous_x = self.fc_layers_continuous(continuous_x)

        # x = torch.cat((continuous_x, categorical_x), dim=1) # (1, 1 + num_categorical, emb_dim//2)
        # Mean pooling
        # pdb.set_trace()
        # x = torch.mean(x, dim=1, keepdim=True).squeeze(0) # (1, emb_dim//2)
        return continuous_x

class Attn_Modality_Gated(nn.Module):
    # Adapted from https://github.com/mahmoodlab/PathomicFusion/blob/master/fusion.py and https://github.com/AIRMEC/HECTOR/blob/main/model.py
    def __init__(self, gate1:bool=True, gate2:bool=True, dim1_og=512, dim2_og=64, use_bilinear=[True,True], scale=[2, 1], dropout_rate=0.25):
        super(Attn_Modality_Gated, self).__init__()
        '''
        args:
        dim1_og:int - dimension of the embedding for the H&E image
        dim2_og:int - dimension of the embedding for the clinical features
        
        '''
        # self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        # can perform attention on latent vectors of lower dimension
        dim1, dim2 = dim1_og//scale[0], dim2_og//scale[1]

        # skip_dim = dim1+dim2+2 if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU()) # e.g. (512, 512)
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear[0] else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1)) # e.g. (512/scale[0], 32, 512/scale[0])
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear[1] else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        # self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        # self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        # self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            # pdb.set_trace()
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear[0] else self.linear_z1(torch.cat((vec1, vec2), dim=1)) # else creates a vector combining both modalities
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear[1] else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)

        return o1, o2
    
class TransMILMultimodal(BaseAggregator):
    def __init__(self, paras:TransMILMultimodalParas):
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
        # ### Clinical feature encoding TREATING ALL VARS AS CONTINUOUS
        # self.clinical_encoding = FeatureEncoding(idx_continuous=self.paras.idx_continuous,
        #                                          taxonomy_in=self.paras.taxonomy_in,
        #                                          embedding_dim=self.paras.embed_size)
        
        self.clinical_encoding = FeatureEncoding(idx_continuous=self.paras.idx_continuous,
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
        
        self.baseline = None # for integrated gradient 
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
        m = self.forward_fusion(h, clinical_embedding) # m is of size (512 // self.paras.scale[0]) * (self.paras.embed_size//2) # e.g. 256 * 32 for kron
        
        # Post-compression of multimodal embedding.
        m = self.post_compression_layer(m)

        #----> predict
        logits = self.classifer(m)  #[B, n_classes]
        # pdb.set_trace()
        return logits
    
    
    def compute_gradients(self, x, clinical_features):
        x.requires_grad = True
        clinical_features.requires_grad = True
        # Compute output
        output = self.forward(x, clinical_features)
        # Compute gradients
        # pdb.set_trace()
        # gradients = torch.autograd.grad(outputs=output, inputs=[x, clinical_features],grad_outputs=torch.ones_like(output))
        gradients = torch.autograd.grad(outputs=output, inputs=[clinical_features], grad_outputs=torch.ones_like(output))
        return gradients
    
    def integrated_gradients(self, x, clinical_features, target=None, steps=50):
        self.eval()
        # Create a baseline if not specified
        if self.baseline is None:
            self.baseline = torch.zeros_like(clinical_features)

        # Calculate the scaled inputs
        alphas = torch.linspace(0, 1, steps).view(-1, 1)  # Shape: [steps, 1]
        alphas = alphas.to(self.baseline.device)
        # pdb.set_trace()
        interpolated = self.baseline + alphas * (clinical_features - self.baseline)
        
        # Compute gradients along the path
        # img_gradients = []
        gradients = []
        for alpha in interpolated:
            # pdb.set_trace()
            grad = self.compute_gradients(x, alpha)
            gradients.append(grad[0])  # Get clinical feature gradients
            # gradients.append(grad[1])  # Get clinical feature gradients

        gradients = torch.stack(gradients)

        # Average the gradients and scale by the input
        avg_gradients = gradients.mean(dim=0)
        integrated_grads = (clinical_features - self.baseline) * avg_gradients
        # Clear the CUDA memory cache
        torch.cuda.empty_cache()
        return integrated_grads
    
    def infer(self, x, clinical_features, coords=None):

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
        
        h = self.norm(h)[:, 0] # [B, 512]

        
        #### MULTIMODAL FUSION OCCURS HERE
        clinical_embedding = self.clinical_encoding(clinical_features)
        
        # Attention gates on each modality.
        h, clinical_embedding = self.attn_modalities(h, clinical_embedding)
        
        # Post-compressiong H&E slide embedding. # reduce dim of H&E embedding to avoid overload from kronecker delta
        # h = self.post_compression_layer_he(h)

        # Fusion, default kronecker delta fusion
        m = self.forward_fusion(h, clinical_embedding) # m is of size (512 // self.paras.scale[0]) * (self.paras.embed_size//2) # e.g. 256 * 32 for kron
        
        # Post-compression of multimodal embedding.
        m = self.post_compression_layer(m)

        #----> predict
        logits = self.classifer(m)  #[B, n_classes]
        if self.paras.task == 'binary': 
            Y_prob = torch.sigmoid(logits)
            Y_hat = torch.round(Y_prob)
        else:
            Y_hat = torch.topk(logits, 1, dim = 1)[1] 
            Y_prob = F.softmax(logits, dim = 1)
        # pdb.set_trace()
        clinical_integrated_gradients = self.integrated_gradients(x, clinical_features) # main comp. bottleneck
        # pdb.set_trace()
        return logits, Y_prob, Y_hat, A_raw, clinical_integrated_gradients
    
    

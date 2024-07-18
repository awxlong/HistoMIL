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
from HistoMIL.MODEL.Image.MIL.GraphTransformer.paras import GraphTransformerParas
from HistoMIL.MODEL.Image.MIL.GraphTransformer.utils import VisionTransformer, GCNBlock
from torch import Tensor
# from HistoMIL import logger

import pdb
class BaseAggregator(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
# from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from torch.nn import Linear
from typing import Optional, Tuple

def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum('ijj->i', x)


def _rank3_diag(x: Tensor) -> Tensor:
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))

    return out

def dense_mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    "https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/mincut_pool.html#dense_mincut_pool"
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss
class Classifier(nn.Module):
    def __init__(self, paras:GraphTransformerParas):
        super(Classifier, self).__init__()

        self.embed_dim = 64
        self.num_layers = 3
        self.node_cluster_num = 100

        self.transformer = VisionTransformer(num_classes=paras.n_class, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.criterion = nn.CrossEntropyLoss()

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(paras.n_features,self.embed_dim, self.bn, self.add_self, self.normalize_embedding, 0., 0)       # 64->128
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)                                          # 100-> 20


    def forward(self,node_feat,adj,graphcam_flag=False):
        # node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        # cls_loss=node_feat.new_zeros(self.num_layers)
        # rank_loss=node_feat.new_zeros(self.num_layers-1)
        X = node_feat # (1, #of nodes, #of features)
        mask = torch.ones((1, X.shape[1])) # (1, #of nodes)
        # p_t=[]
        # pred_logits=0
        # visualize_tools=[]
        # visualize_tools1=[labels.cpu()]
        # embeds=0
        # concats=[]
        
        # layer_acc=[]
                
        X = mask.unsqueeze(2) * X
        X = self.conv1(X, adj, mask)
        s = self.pool1(X)

        if graphcam_flag:
            s_matrix = torch.argmax(s[0], dim=1)
            from os import path
            os.makedirs('graphcam', exist_ok=True)
            torch.save(s_matrix, path.join('graphcam', 's_matrix.pt'))
            torch.save(s[0], path.join('graphcam', 's_matrix_ori.pt'))
            
            if path.exists(path.join('graphcam', 'att_1.pt')):
                os.remove(path.join('graphcam', 'att_1.pt'))
                os.remove(path.join('graphcam', 'att_2.pt'))
                os.remove(path.join('graphcam', 'att_3.pt'))
    
        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)
        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)

        out = self.transformer(X)

        # # loss
        # loss = self.criterion(out, labels)
        # loss = loss + mc1 + o1
        # # pred
        pred = out.data.max(1)[1]

        if graphcam_flag:
            print('GraphCAM enabled')
            p = F.softmax(out)
            torch.save(p, path.join('graphcam', 'prob.pt'))
            index = np.argmax(out.cpu().data.numpy(), axis=-1)

            for index_ in range(p.size(1)):
                one_hot = np.zeros((1, out.size()[-1]), dtype=np.float32)
                one_hot[0, index_] = out[0][index_]
                one_hot_vector = one_hot
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                one_hot = torch.sum(one_hot.cuda() * out)       #!!!!!!!!!!!!!!!!!!!!out-->p
                self.transformer.zero_grad()
                one_hot.backward(retain_graph=True)

                kwargs = {"alpha": 1}
                cam = self.transformer.relprop(torch.tensor(one_hot_vector).to(X.device), method="transformer_attribution", is_ablation=False, 
                                            start_layer=0, **kwargs)

                torch.save(cam, path.join('graphcam', 'cam_{}.pt'.format(index_)))

        return out # labels, loss


if __name__ == "__main__":
    # model_config = {'heads': 8, 
    #             'dim_head': 64, 
    #             'dim': 512, 
    #             'mlp_dim': 512, 
    #             'input_dim':768,
    #             'num_classes':1}
    default_paras = GraphTransformerParas(n_features=1024)
    
    
    model = Classifier(default_paras)
    rand_tensor = torch.rand((1, 29015, 1024))
    uni_adj_matrix = torch.load('/Users/awxlong/Desktop/my-studies/temp_data/CRC/Feature/uni_adj_matrix/temp_sparse_matrix.pt')
    output = model(rand_tensor, uni_adj_matrix.to_dense())
    pdb.set_trace()
    # Load the modified state dictionary into your model
    # model.load_state_dict(new_state_dict)
    # pdb.set_trace()

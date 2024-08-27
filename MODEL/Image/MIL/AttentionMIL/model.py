"""
AttentionMIL Ilse et al. (2018) https://arxiv.org/abs/1802.04712 
copied from https://github.com/peng-lab/HistoBistro/blob/main/models/aggregators/attentionmil.py
"""
import os
import sys
sys.path.append('/Users/awxlong/Desktop/my-studies/hpc_exps/')
# import numpy as np
from einops import repeat
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
#-----------> external network modules 
from HistoMIL.MODEL.Image.MIL.utils import MILAttention, FeatureNet
from HistoMIL.MODEL.Image.MIL.AttentionMIL.paras import AttentionMILParas
# from HistoMIL import logger

import pdb
class BaseAggregator(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

class AttentionMIL(BaseAggregator):
    def __init__(
        self,
        paras:AttentionMILParas, 
        encoder: Optional[nn.Module] = FeatureNet,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        **kwargs
    ) -> None:
        """Create a new attention MIL model.
        Args:
            n_feats:  The nuber of features each bag instance has.
            n_out:  The number of output layers of the model.
            encoder:  A network transforming bag instances into feature vectors.
        """
        super(BaseAggregator, self).__init__()
        self.attention_mil_paras = paras
        self.encoder = nn.Sequential(
            encoder(model_name=self.attention_mil_paras.encoder_name),
            nn.ReLU(),
            nn.Linear(self.attention_mil_paras.input_dim, 256), 
            nn.ReLU()
        )

        # or nn.Sequential(
        #     nn.Linear(self.attention_mil_paras.input_dim, 256), nn.ReLU()
        # )
        self.attention = attention or MILAttention(256)
        self.head = head or nn.Sequential(
            nn.Flatten(),
            # nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.Linear(256, self.attention_mil_paras.num_classes)
        )

    def forward(self, bags, coords=None, tiles=None, **kwargs):
        assert bags.ndim == 3
        if tiles is not None:
            assert bags.shape[0] == tiles.shape[0]
        else:
            tiles = torch.tensor([bags.shape[1]],
                                 device=bags.device).unsqueeze(0)

        embeddings = self.encoder(bags)

        # mask out entries if tiles < num_tiles
        masked_attention_scores = self._masked_attention_scores(
            embeddings, tiles
        )
        weighted_embedding_sums = (masked_attention_scores * embeddings).sum(-2)

        scores = self.head(weighted_embedding_sums)

        return scores # use BCEWithLogitsLoss due to absence of sigmoid

    def _masked_attention_scores(self, embeddings, tiles):
        """Calculates attention scores for all bags.
        Returns:
            A tensor containingtorch.concat([torch.rand(64, 256), torch.rand(64, 23)], -1)
             *  The attention score of instance i of bag j if i < len[j]
             *  0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)

        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = (torch.arange(bag_size).repeat(bs, 1).to(attention_scores.device))

        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < tiles).unsqueeze(-1)
        
        min_value = torch.finfo(attention_scores.dtype).min
        masked_attention = torch.where(
            attention_mask, attention_scores,
            torch.full_like(attention_scores, min_value)
        )
        # masked_attention = torch.where(
        #     attention_mask, attention_scores,
        #     # pdb.set_trace()
        #     torch.full_like(attention_scores, -1e10)
        # )

        return torch.softmax(masked_attention, dim=1)
    
    def infer_masked_attention_scores(self, embeddings, tiles):
        """Calculates attention scores for all bags.
        Returns:
            A tensor containingtorch.concat([torch.rand(64, 256), torch.rand(64, 23)], -1)
             *  The attention score of instance i of bag j if i < len[j]
             *  0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)

        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = (torch.arange(bag_size).repeat(bs, 1).to(attention_scores.device))

        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < tiles).unsqueeze(-1)
        
        min_value = torch.finfo(attention_scores.dtype).min
        masked_attention = torch.where(
            attention_mask, attention_scores,
            torch.full_like(attention_scores, min_value)
        )
        

        return torch.softmax(masked_attention, dim=1), attention_scores
    
    def infer(self, bags, coords=None, tiles=None, **kwargs):
        assert bags.ndim == 3
        if tiles is not None:
            assert bags.shape[0] == tiles.shape[0]
        else:
            tiles = torch.tensor([bags.shape[1]],
                                 device=bags.device).unsqueeze(0)

        embeddings = self.encoder(bags)

        # mask out entries if tiles < num_tiles
        masked_attention_scores, A_raw = self.infer_masked_attention_scores(
            embeddings, tiles
        )
        weighted_embedding_sums = (masked_attention_scores * embeddings).sum(-2)
        # pdb.set_trace()
        A_raw = A_raw.squeeze(-1)
        logits = self.head(weighted_embedding_sums)
        if self.attention_mil_paras.task == 'binary': 
            Y_prob = torch.sigmoid(logits)
            Y_hat = torch.round(Y_prob)
        else:
            Y_hat = torch.topk(logits, 1, dim = 1)[1] 
            Y_prob = F.softmax(logits, dim = 1)
        return logits, Y_prob, Y_hat, A_raw 
    

def test_attentionmil():
    attentionmil = AttentionMIL(num_classes=2, input_dim=1024)
    input_tensor = torch.rand(1, 1, 1024)
    output = attentionmil(input_tensor)
    assert torch.equal(torch.tensor(output.size()), torch.tensor([1, 2]))
if __name__ == "__main__":
    # # model_config = {'heads': 8, 
    # #             'dim_head': 64, 
    # #             'dim': 512, 
    # #             'mlp_dim': 512, 
    # #             'input_dim':768,
    # #             'num_classes':1}
    default_paras = AttentionMILParas(input_dim=1024, \
                                    encoder_name='pre-calculated', 
                                    num_classes=1)
    # default_paras.pretrained_weights_dir = '/Users/awxlong/Desktop/my-studies/hpc_exps/HistoMIL/MODEL/Image/MIL/Transformer/pretrained_weights/'
    # default_paras.selective_finetuning = False
    rand_tensor = torch.rand(1, 421, 1024) 
    model = AttentionMIL(default_paras)

    # pdb.set_trace()

    y = model.infer(rand_tensor)
    pdb.set_trace()
    # Load the modified state dictionary into your model
    # model.load_state_dict(new_state_dict)
    # pdb.set_trace()

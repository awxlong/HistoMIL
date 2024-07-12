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
from HistoMIL.MODEL.Image.MIL.utils import FeatureNet, BaseAggregator, PPEG, NystromTransformerLayer, NystromAttention
from HistoMIL.MODEL.Image.MIL.CAMIL.paras import CAMILParas
from HistoMIL import logger
######################################################################################
#        pytorch module to define structure
######################################################################################
import math
import pdb


### IMPLEMENTATION ADAPTED FROM https://github.com/olgarithmics/ICLR_CAMIL


class MILAttentionLayer(nn.Module):
    """Implementation of the attention-based Deep MIL layer.
    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.
    """

    def __init__(
            self,
            input_dim,
            weight_params_dim,
            use_gated=False,
    ):
        super().__init__()

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        # Initialize weights
        self.v_weight_params = nn.Parameter(
            torch.Tensor(input_dim, self.weight_params_dim),
            requires_grad = True,)
        self.w_weight_params = nn.Parameter(
            torch.Tensor(self.weight_params_dim, 1),
            requires_grad = True)

        if self.use_gated:
            self.u_weight_params = nn.Parameter(
                torch.Tensor(input_dim, self.weight_params_dim),
                requires_grad=True,)
        else:
            self.register_parameter('u_weight_params', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.v_weight_params)
        nn.init.xavier_uniform_(self.w_weight_params)
        if self.use_gated:
            nn.init.xavier_uniform_(self.u_weight_params)

    def forward(self, inputs):
        # Compute attention scores
        instances = self.compute_attention_scores(inputs)

        # Apply softmax over instances such that the output summation is equal to 1
        alpha = F.softmax(instances, dim=0)
        return alpha

    def compute_attention_scores(self, instance):
        # Reserve in-case "gated mechanism" used
        original_instance = instance

        # tanh(v*h_k^T)
        instance = torch.tanh(torch.mm(instance, self.v_weight_params))

        # for learning non-linear relations efficiently
        if self.use_gated:
            gate = torch.sigmoid(
                torch.mm(original_instance, self.u_weight_params)) # maybe torch.matmul(original_instance, self.u_weight_params.T) 
            instance = instance * gate

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return torch.mm(instance, self.w_weight_params) # axes = 1?
    
class NeighborAggregator(nn.Module):
    """
    Aggregation of neighborhood information
    This layer is responsible for aggregating the neighborhood information of the attention matrix through the
    element-wise multiplication with an adjacency matrix. Every row of the produced
    matrix is averaged to produce a single attention score.
    
    # Arguments
        output_dim:            positive integer, dimensionality of the output space
    
    # Input shape
        2D tensor with shape: (n, n)
        2D tensor with shape: (None, None) corresponding to the adjacency matrix
    
    # Output shape
        2D tensor with shape: (1, units) corresponding to the attention coefficients of every instance in the bag
    """

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, inputs):
        data_input, adj_matrix = inputs  # [attention_matrix, sparse_adj] in encoder's forward function

        # Element-wise multiplication of data_input and adj_matrix
        # sparse_data_input = torch.mul(data_input, adj_matrix.to_dense())
        sparse_data_input = adj_matrix * data_input # according to another perplexity's answer
        # Sum along rows
        # reduced_sum = torch.sum(sparse_data_input, dim=1)
        reduced_sum = torch.sparse.sum(sparse_data_input, dim=1)# more efficient apparently
        # Reshape to match the expected shape
        A_raw = reduced_sum.view(-1)

        # Apply softmax to get attention weights
        alpha = F.softmax(A_raw, dim=0)

        return alpha, A_raw

    def extra_repr(self):
        return f'output_dim={self.output_dim}'

class Last_Sigmoid(nn.Module):
    """
    Attention Activation
    This layer contains the last sigmoid layer of the network
    # Arguments
        output_dim:         positive integer, dimensionality of the output space
        subtyping:          boolean, whether to use subtyping or not
        pooling_mode:       string, pooling mode ('max' or 'sum')
        use_bias:           boolean, whether to use bias or not
    # Input shape
        2D tensor with shape: (n, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """

    def __init__(self, input_dim, output_dim, subtyping=False, pooling_mode="sum", use_bias=True):
        super(Last_Sigmoid, self).__init__()
        self.output_dim = output_dim
        self.subtyping = subtyping
        self.pooling_mode = pooling_mode
        self.use_bias = use_bias

        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

    def max_pooling(self, x):
        return torch.max(x, dim=0, keepdim=True)[0]

    def sum_pooling(self, x):
        return torch.sum(x, dim=0, keepdim=True)

    def forward(self, x):
        if self.pooling_mode == 'max':
            x = self.max_pooling(x)
        elif self.pooling_mode == 'sum':
            x = self.sum_pooling(x)

        x = self.fc(x)

        if self.subtyping:
            out = F.softmax(x, dim=1)
        else:
            out = torch.sigmoid(x)

        return out

class CustomAttention(nn.Module):
    def __init__(
            self,
            input_dim,
            weight_params_dim,
            kernel_initializer="xavier_uniform",
            kernel_regularizer=None,
    ):
        super().__init__()

        self.weight_params_dim = weight_params_dim

        # Initialize weights
        self.wq_weight_params = nn.Parameter(torch.Tensor(input_dim, 
                                                          weight_params_dim))
        self.wk_weight_params = nn.Parameter(torch.Tensor(input_dim, 
                                                          weight_params_dim))

        # Initialize weights using the specified initializer
        if kernel_initializer == "xavier_uniform":
            nn.init.xavier_uniform_(self.wq_weight_params)
            nn.init.xavier_uniform_(self.wk_weight_params)
        else:
            # Add other initializers as needed
            raise ValueError(f"Unsupported initializer: {kernel_initializer}")

        # Regularization is typically applied during the loss computation in PyTorch
        self.kernel_regularizer = kernel_regularizer

    def forward(self, inputs):
        return self.compute_attention_scores(inputs)

    def compute_attention_scores(self, instance):
        q = torch.matmul(instance, self.wq_weight_params)
        k = torch.matmul(instance, self.wk_weight_params)

        # dk = torch.tensor(k.size(-1), dtype=torch.float32)
        dk = torch.tensor(k.shape[-1], dtype=torch.float32)

        # matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
        matmul_qk = torch.matmul(q, k.T)
        
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        return scaled_attention_logits

class encoder(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.custom_att = CustomAttention(input_dim=input_dim, 
                                          weight_params_dim=256)
        self.wv = nn.Linear(512, 512)
        self.neigh = NeighborAggregator(output_dim=1)
        self.nyst_att = NystromAttention(dim=512, dim_head=64, heads=8, 
                                         num_landmarks=256, pinv_iterations=6) # from utils

    def forward(self, inputs):
        dense, sparse_adj = inputs  # adjacency matrix

        encoder_output = self.nyst_att(dense.unsqueeze(0), return_attn=False)
        xg = encoder_output.squeeze(0)

        encoder_output = xg + dense

        attention_matrix = self.custom_att(encoder_output)
        norm_alpha, alpha = self.neigh([attention_matrix, sparse_adj])
        value = self.wv(dense)
        xl = torch.mul(norm_alpha, value)

        wei = torch.sigmoid(-xl)
        squared_wei = wei ** 2
        xo = (xl * 2 * squared_wei) + 2 * encoder_output * (1 - squared_wei)
        return xo, alpha
    
class CAMIL(nn.Module):
    def __init__(self, paras:CAMILParas):
        super(CAMIL, self).__init__()
        self.paras = paras
        self.input_shape = paras.input_shape
        self.n_classes = paras.n_classes
        self.subtyping = paras.subtyping

        
        self.attcls = MILAttentionLayer(weight_params_dim=128, 
                                        use_gated=True)
        self.custom_att = CustomAttention(input_dim=paras.input_shape, 
                                          weight_params_dim=256)
        self.wv = nn.Linear(512, 512)
        self.neigh = NeighborAggregator(output_dim=1)
        self.nyst_att = NystromAttention(dim=512, dim_head=64, heads=8, num_landmarks=256,
                                         pinv_iterations=6) # neighboraggregator and nystromattention are inside encoder

        self.encoder = encoder(input_dim=paras.input_shape)

        if self.subtyping:
            self.class_fc = Last_Sigmoid(input_dim=512, output_dim=self.n_classes, subtyping=True)
        else:
            self.class_fc = Last_Sigmoid(input_dim=512, output_dim=1, subtyping=False)

    def forward(self, inputs):
        bag, adjacency_matrix = inputs
        xo, alpha = self.encoder([bag, adjacency_matrix])

        k_alpha = self.attcls(xo)

        attn_output = torch.mul(k_alpha, xo)

        out = self.class_fc(attn_output)
        # alpha = self.neigh(x, adjacency_matrix)
        # k_alpha = self.attcls(x)
        # attn_output = k_alpha * x
        # out = self.class_fc(attn_output)
        return out, alpha, k_alpha

if __name__ == "__main__":
    
    default_paras = CAMILParas()
    rand_tensor = torch.rand(1, 1, 1024) 
    model = CAMIL(paras=default_paras)

    pdb.set_trace()
    
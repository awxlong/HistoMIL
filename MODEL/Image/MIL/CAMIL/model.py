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

from torch.cuda.amp import autocast
from torch.autograd import Function

### IMPLEMENTATION ADAPTED FROM https://github.com/olgarithmics/ICLR_CAMIL



class SparseToDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return input_tensor.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to_sparse()

sparse_to_dense = SparseToDense.apply
class SparseNeighborAggregator(Function):
    @staticmethod
    def forward(ctx, data_input, adj_matrix):
        ctx.save_for_backward(data_input, adj_matrix)
        
        sparse_data_input = adj_matrix * data_input
        reduced_sum = torch.sparse.sum(sparse_data_input, dim=1)
        A_raw = reduced_sum.to_dense().flatten()
        alpha = F.softmax(A_raw, dim=0)
        
        return alpha, A_raw

    @staticmethod
    def backward(ctx, grad_output_alpha, grad_output_A_raw):
        data_input, adj_matrix = ctx.saved_tensors
        
        # Compute gradients for data_input
        grad_data_input = torch.sparse_coo_tensor(
            adj_matrix._indices(),
            adj_matrix._values() * grad_output_alpha[adj_matrix._indices()[1]],
            adj_matrix.size()
        ).to_dense()
        
        # We don't compute gradients for adj_matrix as it's typically fixed
        grad_adj_matrix = None
        
        return grad_data_input, grad_adj_matrix
    

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
        # pdb.set_trace()
        instance = torch.tanh(torch.matmul(instance, self.v_weight_params))

        # for learning non-linear relations efficiently
        if self.use_gated:
            gate = torch.sigmoid(
                torch.matmul(original_instance, self.u_weight_params)) # maybe torch.matmul(original_instance, self.u_weight_params.T) 
            instance = instance * gate

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return torch.matmul(instance, self.w_weight_params) # axes = 1?
    
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
        # Element-wise multiplication of data_input and adj_matrix
        # dense_data_input = torch.mul(data_input, adj_matrix.to_dense())
        # sparse_data_input = adj_matrix * data_input # according to another perplexity's answer; sparse_data_input is sparse
        # pdb.set_trace()
        # sparse_data_input = torch.sparse.mm(adj_matrix, data_input)
        
        # Convert to dense, sum, and convert back to sparse if needed
        # dense_data_input = sparse_data_input.to_dense()
        
        # dense_data_input = torch.mul(data_input, adj_matrix)
        # reduced_dense_sum = torch.sum(dense_data_input, dim=1)
        # # Sum along rows
        # reduced_sum = torch.sum(sparse_data_input, dim=1)
        # reduced_sum = torch.sparse.sum(sparse_data_input, dim=1)# more efficient apparently
        # The below MAY trigger a reshape not implemented during backward pass
        # # Reshape to match the expected shape
        # # A_raw = reduced_sum.view(-1)
        # # A_raw = reduced_sum.reshape(-1)
        # # A_raw = reduced_sum.view(data_input.size(1))

        # ### perplexity
        # # Ensure the sparse tensor is coalesced
        # reduced_sum = reduced_sum.coalesce()     # most likely triggers the reshape not implemented during backward pass    

        # # Get the indices and values
        # indices = reduced_sum.indices().squeeze()
        # values = reduced_sum.values()

        # # Create a new dense tensor with the desired shape
        # A_raw = torch.zeros(data_input.size(1), device=reduced_sum.device)
        # # Fill in the values
        # A_raw[indices] = values
        # pdb.set_trace()
        # Reshape to match the original Keras implementation
        # A_raw = reduced_dense_sum.view(-1)
        # Convert to dense (this is necessary for softmax)
        # pdb.set_trace()
        # Reshape to match the original Keras implementation; dense tensor occupy more memory
        # A_raw = reduced_sum.to_dense().view(data_input.size(1))
        # pdb.set_trace()
        # Element-wise multiplication of sparse adj_matrix with dense data_input
        # sparse_data_input = torch.sparse.mm(adj_matrix, data_input)
        # Reshape data_input to (num_patches, num_features)
        data_input, adj_matrix = inputs  # [attention_matrix, sparse_adj] in encoder's forward function

        data_input = data_input.squeeze(0)# .float()
        
        # sparse_data_input = adj_matrix * data_input
        # # Perform sparse operation outside of autocast
        
        # # pdb.set_trace()
        # # Sum along the rows (dim=1)
        # reduced_sum = torch.sparse.sum(sparse_data_input, dim=1)
      
        # # Flatten the tensor to (num_patches,)
        # A_raw = reduced_sum.flatten()
        
        # # Apply softmax
        # alpha = F.softmax(A_raw, dim=0)
        alpha, A_raw = SparseNeighborAggregator.apply(data_input, adj_matrix)
        
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

    def __init__(self, input_dim, output_dim, subtyping, 
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 pooling_mode="sum", use_bias=True):
        super(Last_Sigmoid, self).__init__()
        
        self.output_dim = output_dim
        self.subtyping = subtyping
        self.pooling_mode = pooling_mode
        self.use_bias = use_bias

        # Initialize weights
        if kernel_initializer == 'glorot_uniform':
            self.kernel = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(input_dim, output_dim)))
        else:
            self.kernel = nn.Parameter(torch.randn(input_dim, output_dim))

        # Initialize bias
        if use_bias:
            if bias_initializer == 'zeros':
                self.bias = nn.Parameter(torch.zeros(output_dim))
            else:
                self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)

    def max_pooling(self, x):
        return torch.max(x, dim=0, keepdim=True)[0]

    def sum_pooling(self, x):
        return torch.sum(x, dim=0, keepdim=True)

    def forward(self, x):
        # pdb.set_trace()
        if x.size(0) == 1:
            x = x.squeeze(0)

        # Apply pooling
        if self.pooling_mode == 'max':
            x = self.max_pooling(x)
        elif self.pooling_mode == 'sum':
            x = self.sum_pooling(x)

        # Apply linear transformation
        x = torch.matmul(x, self.kernel)
        
        if self.use_bias:
            x = x + self.bias
        # pdb.set_trace()
        # # Apply activation; commented out since we use BCEWithLogitsLoss
        # if self.subtyping:
        #     out = F.softmax(x, dim=-1)
        # else:
        #     out = torch.sigmoid(x)

        return x
    

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
        ### can futher optimize this using F.scaled_dot_product 
        # q = torch.matmul(instance, self.wq_weight_params)
        # k = torch.matmul(instance, self.wk_weight_params)

        # # dk = torch.tensor(k.size(-1), dtype=torch.float32)
        # dk = torch.tensor(k.shape[-1], dtype=torch.int32)
        # # pdb.set_trace()
        # matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
        # # matmul_qk = torch.tensordot(q, k.transpose(-2, -1), dims=1) # could also be this
        
        # scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        
        chunk_size = 1024  # Adjust this value based on your GPU memory
        q_chunks = []
        k_chunks = []
        
        for i in range(0, instance.size(0), chunk_size):
            chunk = instance[i:i+chunk_size]
            q_chunks.append(torch.matmul(chunk, self.wq_weight_params))
            k_chunks.append(torch.matmul(chunk, self.wk_weight_params))
        
        q = torch.cat(q_chunks, dim=0)
        k = torch.cat(k_chunks, dim=0)

        # Use torch.sqrt() directly on a scalar
        dk = torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))

        # Use torch.bmm for batch matrix multiplication
        # Reshape q and k for bmm
        q_reshaped = q.view(-1, q.size(-2), q.size(-1))
        k_reshaped = k.view(-1, k.size(-2), k.size(-1))
        
        # Compute attention scores in chunks
        attention_chunks = []
        for i in range(0, q_reshaped.size(0), chunk_size):
            q_chunk = q_reshaped[i:i+chunk_size]
            k_chunk = k_reshaped[i:i+chunk_size]
            
            matmul_qk = torch.bmm(q_chunk, k_chunk.transpose(1, 2))
            scaled_chunk = matmul_qk / dk
            attention_chunks.append(scaled_chunk)
        
        scaled_attention_logits = torch.cat(attention_chunks, dim=0)
        
        # Reshape back to original dimensions
        scaled_attention_logits = scaled_attention_logits.view(q.shape[:-1] + (k.size(-2),))
        # pdb.set_trace()
        return scaled_attention_logits


class encoder(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.custom_att = CustomAttention(input_dim=input_dim, 
                                          weight_params_dim=256)
        self.wv = nn.Linear(input_dim, input_dim)
        self.neigh = NeighborAggregator(output_dim=1)
        self.nyst_att = NystromAttention(dim=input_dim, dim_head=64, heads=8, 
                                         num_landmarks=256, pinv_iterations=6) # from utils

    def forward(self, inputs):
        dense, sparse_adj = inputs  # adjacency matrix

        encoder_output = self.nyst_att(dense, return_attn=False)

        xg = encoder_output.squeeze(0) # ([1, #patches, input_dim])

        encoder_output = xg + dense

        attention_matrix = self.custom_att(encoder_output)

        norm_alpha, alpha = self.neigh([attention_matrix, sparse_adj]) # attention coefficients

        value = self.wv(dense)
        # pdb.set_trace()
        norm_alpha = norm_alpha.unsqueeze(1) # reshape to (num_nodes, 1) as per perplexity
        xl = torch.mul(norm_alpha, value) # (1, #patches, 512)

        wei = torch.sigmoid(-xl) # (1, #patches, 512)
        squared_wei = wei ** 2 # (1, #patches, 512)
        # pdb.set_trace()
        xo = (xl * 2 * squared_wei) + 2 * encoder_output * (1 - squared_wei)
        return xo, alpha
    
class CAMIL(nn.Module):
    def __init__(self, paras:CAMILParas):
        super().__init__()
        self.paras = paras
        self.input_shape = paras.input_shape
        self.n_classes = paras.num_classes
        self.subtyping = paras.subtyping

        
        self.attcls = MILAttentionLayer(input_dim=self.input_shape,
                                        weight_params_dim=128, 
                                        use_gated=True)
        self.custom_att = CustomAttention(input_dim=paras.input_shape, 
                                          weight_params_dim=256)
        self.wv = nn.Linear(paras.input_shape, 512)
        self.neigh = NeighborAggregator(output_dim=1)
        self.nyst_att = NystromAttention(dim=512, dim_head=64, heads=8, num_landmarks=256,
                                         pinv_iterations=6) # neighboraggregator and nystromattention are inside encoder

        self.encoder = encoder(input_dim=paras.input_shape)

        if self.subtyping:
            self.class_fc = Last_Sigmoid(input_dim=self.input_shape, output_dim=self.n_classes, subtyping=True)
        else:
            self.class_fc = Last_Sigmoid(input_dim=self.input_shape, output_dim=1, subtyping=False)

    def forward(self, inputs):
        bag, adjacency_matrix = inputs

        xo, alpha = self.encoder([bag, adjacency_matrix])

        k_alpha = self.attcls(xo)
        
        # pdb.set_trace()

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
    
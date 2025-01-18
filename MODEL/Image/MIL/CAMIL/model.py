"""
IMPLEMENTATION OF CAMIL ADAPTED FROM https://github.com/olgarithmics/ICLR_CAMIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#-----------> external network modules 
from HistoMIL.MODEL.Image.MIL.utils import  NystromAttention
from HistoMIL.MODEL.Image.MIL.CAMIL.paras import CAMILParas

import pdb



# sparse_to_dense = SparseToDense.apply
class SparseNeighborAggregation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data_input, adj_matrix):
        ctx.save_for_backward(data_input, adj_matrix)
        
        # Element-wise multiplication
        sparse_data_input = adj_matrix * data_input
        
        # Sum along the rows (dim=1)
        reduced_sum = torch.sparse.sum(sparse_data_input, dim=1)
        
        # Flatten the tensor
        A_raw = reduced_sum.to_dense().flatten()
        
        return A_raw

    @staticmethod
    def backward(ctx, grad_output):
        data_input, adj_matrix = ctx.saved_tensors
        
        # Reshape grad_output to match adj_matrix's shape
        grad_output_reshaped = grad_output.view(adj_matrix.size(0), 1)
        
        # Calculate gradients for data_input
        grad_data = torch.zeros_like(data_input)
        indices = adj_matrix._indices()
        values = adj_matrix._values()
        
        for i in range(indices.size(1)):
            row, col = indices[:, i]
            grad_data[col] += grad_output_reshaped[row] * values[i]
        
        # We don't need to calculate gradients for adj_matrix
        return grad_data, None
    

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
        
        self.v_weight = nn.Linear(input_dim, self.weight_params_dim)
        
        self.w_weight = nn.Linear(self.weight_params_dim, 1)

        if self.use_gated:
            self.u_weight = nn.Linear(input_dim, self.weight_params_dim)
        else:
            self.register_parameter('u_weight', None)

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
        instance = torch.tanh(self.v_weight(instance)) # torch.tanh(torch.matmul(instance, self.v_weight_params))

        # for learning non-linear relations efficiently
        if self.use_gated:
            # gate = torch.sigmoid(
            #     torch.matmul(original_instance, self.u_weight_params)) # maybe torch.matmul(original_instance, self.u_weight_params.T) 
            gate = torch.sigmoid(self.u_weight(original_instance))
            instance = instance * gate

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return self.w_weight(instance) # torch.matmul(instance, self.w_weight_params) # axes = 1?
    
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
        self.sparse_aggregation = SparseNeighborAggregation.apply


    def forward(self, inputs):
        
        data_input, adj_matrix = inputs  # [attention_matrix, sparse_adj] in encoder's forward function

        
        sparse_data_input = adj_matrix * data_input
        
        reduced_sum = torch.sparse.sum(sparse_data_input, dim=1)
      
        # # Flatten the tensor to (num_patches,)
        A_raw = reduced_sum.to_dense().flatten()
        
        # # Apply softmax
        # alpha = F.softmax(A_raw, dim=0)

        # A_raw = self.sparse_aggregation(data_input, adj_matrix)
        
        # Apply softmax
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

    def __init__(self, input_dim, output_dim, subtyping, 
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 pooling_mode="mean", use_bias=True):
        super(Last_Sigmoid, self).__init__()
        
        self.output_dim = output_dim
        self.subtyping = subtyping
        self.pooling_mode = pooling_mode
        self.use_bias = use_bias

        # Initialize weights
        if kernel_initializer == 'glorot_uniform':
            self.kernel = nn.Linear(input_dim, output_dim) # nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(input_dim, output_dim)))
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

    def mean_pooling(self, x):
        return torch.mean(x, dim=0, keepdim=True)
    
    def forward(self, x):
        # pdb.set_trace()
        if x.size(0) == 1:
            x = x.squeeze(0)
        # pdb.set_trace()
        # Apply pooling
        if self.pooling_mode == 'max':
            x = self.max_pooling(x)
        elif self.pooling_mode == 'sum':
            x = self.sum_pooling(x)
        elif self.pooling_mode == 'mean':
            x = self.mean_pooling(x)
        # pdb.set_trace()
        # Apply linear transformation
        x = self.kernel(x) # torch.matmul(x, self.kernel)
        
        if self.use_bias:
            x = x + self.bias
        
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

        self.wq = nn.Linear(input_dim, self.weight_params_dim)
        self.wk = nn.Linear(input_dim, self.weight_params_dim)
    
    def forward(self, inputs, sparse_adj_matrix):
        # Cast input to float16 for mixed-precision training. Can comment this out if 
        # you're sure model weights are not in float16
        inputs = inputs.to(torch.float16)

        # Compute Q and K
        q = self.wq(inputs)
        k = self.wk(inputs)
        q = q.squeeze(0)
        k = k.squeeze(0) # [#patches, 256]
        # pdb.set_trace()
        dk = k.size(-1) ** 0.5 
        
        # Get indices of non-zero elements in sparse_adj_matrix
        indices = sparse_adj_matrix._indices()
        
        # Compute attention scores only for non-zero indices
        q_selected = q[indices[0]]  # Select relevant rows from q
        k_selected = k[indices[1]]  # Select relevant columns from k
        
        # Compute dot product for selected elements
        attn_scores = torch.sum(q_selected * k_selected, dim=-1)
        
        # Scale the attention scores
        # dk = (self.weight_params_dim ** 0.5)
        attn_scores = attn_scores / dk

        # Create a new sparse matrix with computed attention scores
        values = attn_scores
        sparse_attn_matrix = torch.sparse_coo_tensor(indices, values, sparse_adj_matrix.size())

        return sparse_attn_matrix
    


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

        encoder_output = self.nyst_att(dense, return_attn=False) # this is the nystroformer

        xg = encoder_output.squeeze(0) # ([1, #patches, input_dim])

        encoder_output = xg + dense # key + query
        # pdb.set_trace()
        attention_matrix = self.custom_att(encoder_output, sparse_adj) # one of the main comp. bottleneck

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
    
        return out, alpha, k_alpha
    
    def infer(self, inputs):
        bag, adjacency_matrix = inputs

        xo, A_raw = self.encoder([bag, adjacency_matrix]) # alpha is A_raw

        k_alpha = self.attcls(xo)
        
        # pdb.set_trace()

        attn_output = torch.mul(k_alpha, xo)

        logits = self.class_fc(attn_output)
        
        if self.paras.task == 'binary': 
            Y_prob = torch.sigmoid(logits)
            Y_hat = torch.round(Y_prob)
        else:
            Y_hat = torch.topk(logits, 1, dim = 1)[1] 
            Y_prob = F.softmax(logits, dim = 1)

        A_raw = A_raw.unsqueeze(0) # [B=1, N of patches] add batch dimension
        # pdb.set_trace()
        return logits, Y_prob, Y_hat, A_raw

if __name__ == "__main__":
    ### Testing CAMIL
    device = 'cpu'
    with torch.amp.autocast(device_type='cuda'):
        default_paras = CAMILParas()
        rand_tensor = torch.rand(1, 29015, 1024, dtype=torch.bfloat16).to(device) 
        uni_adj_matrix = torch.load('/Users/awxlong/Desktop/my-studies/temp_data/CRC/Feature/uni_adj_matrix/temp_sparse_matrix.pt')

        model = CAMIL(paras=default_paras).to(device)
        y = model.infer([rand_tensor, uni_adj_matrix])
    pdb.set_trace()
    

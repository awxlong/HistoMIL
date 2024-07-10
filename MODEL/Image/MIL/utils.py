"""
Hacked together from https://github.com/lucidrains
"""

import torch
import timm
from HistoMIL import logger

import math
from functools import wraps
from math import ceil, pi
from typing import Optional

# import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum, nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
#--------> commonly used feature encoder function for MIL

import pdb

class BaseAggregator(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class FeatureNet(torch.nn.Module):
    def __init__(self,model_name,pretrained:bool=True):
        super().__init__()
        logger.info(f"FeatureNet:: Use: {model_name} ")
        self.name = model_name
        if model_name == "pre-calculated":
            self.pre_trained = None
        else:# get pretrained model for feature extraction
            self.pre_trained = timm.create_model(model_name, 
                                            pretrained=pretrained, 
                                            num_classes=0)
        self.freeze_flag = False

    def freeze(self):
        if self.pre_trained is not None:
            if not self.freeze_flag:
                for params in self.parameters():
                    params.requires_grad = False
                
                for layer in self.modules():
                    if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
                        layer.track_running_stats = False

                self.freeze_flag = True
        
    def unfreeze(self):
        if self.pre_trained is not None:
            if self.freeze_flag:
                for param in self.parameters():
                    param.requires_grad = True
                for layer in self.modules():
                    if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
                        layer.track_running_stats = True
                self.freeze_flag = False

    #@torchsnooper.snoop()
    def forward(self, x):
        if self.pre_trained is None:
            return x
        x = self.pre_trained(x)
        return x

    def get_features_bag(self, bag):
        with torch.no_grad():
            fv = []
            for batch_of_patches in bag:
                fv.append(self.forward(batch_of_patches))
            identity_matrix = torch.eye(fv.shape[0], dtype=torch.bool)
            return torch.cat(fv),identity_matrix
        

# --------------------
# Helpers copied from utils.py from https://github.com/peng-lab/HistoBistro/tree/main
# --------------------

def get_loss(name, **kwargs):
    # Check if the name is a valid loss name
    if name in nn.__dict__:
        # Get the loss class from the torch.nn module
        loss_class = getattr(nn, name)
        # Instantiate the loss with the reduction option
        loss = loss_class(**kwargs)
        # Return the loss
        return loss
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid loss name: {name}")
    


def get_optimizer(name, model, lr=0.01, wd=0.1):
    # Check if the name is a valid optimizer name
    if name in optim.__dict__:
        # Get the optimizer class from the torch.optim module
        optimizer_class = getattr(optim, name)
        # Instantiate the optimizer with the model parameters and the learning rate
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=wd)
        # Return the optimizer
        # pdb.set_trace()
        return optimizer
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid optimizer name: {name}")


def get_scheduler(name, optimizer, optim_config):
    # Check if the name is a valid scheduler name
    if name in lr_scheduler.__dict__:
        # Get the scheduler class from the torch.optim.lr_scheduler module
        scheduler_class = getattr(lr_scheduler, name)
        # Instantiate the scheduler with the optimizer and other keyword arguments
        scheduler = scheduler_class(optimizer, **optim_config)
        # Return the scheduler
        # pdb.set_trace()
        return scheduler
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid scheduler name: {name}")


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(
        1., max_freq / 2, num_bands, device=device, dtype=dtype
    )
    scales = scales[(*((None, ) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {
            out_field:
                (edges.src[src_field] *
                 edges.dst[dst_field]).sum(-1, keepdim=True)
        }

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {
            field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))
        }

    return func

# --------------------
# Activations
# --------------------

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

# --------------------
# Normalization
# --------------------

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        # pdb.set_trace()
        return self.fn(x, **kwargs) # calls the attention module

# --------------------
# Positional Embeddings
# --------------------

class LearnedPositionalEmbedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, pad_index: int
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, pad_index)
        self.num_embeddings = num_embeddings
        self.pad_index = pad_index

    def forward(self, input):
        positions = self._make_positions(input, self.pad_index)
        return self.embedding(positions)

    def max_positions(self):
        if self.pad_index is not None:
            return self.num_embeddings - self.pad_index - 1
        else:
            return self.num_embeddings

    def _make_positions(self, tensor, pad_index: int):
        masked = tensor.ne(pad_index).long()
        return torch.cumsum(masked, dim=1) * masked + pad_index


class SinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=100_000):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * \
            -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# --------------------
# FeedForward
# --------------------

class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# --------------------
# Attentions
# --------------------

class Attention(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=512 // 8, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, register_hook=False):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )

        
        # print(q.shape)
        # print(k.shape)
        # pdb.set_trace()
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attn = self.attend(dots)
        # Using memory-efficient attention
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.training and 0.1 or 0.0)


        # save self-attention maps
        self.save_attention_map(attn)
        if register_hook:
            attn.register_hook(self.save_attn_gradients)

        # out = torch.matmul(attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(attn, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def get_self_attention(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        return attn


class PerceiverAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MILAttention(nn.Module):
    """
    A network calculating an embedding's importance weight.
    """
    def __init__(self, n_in: int, n_latent: Optional[int] = None):
        super().__init__()
        n_latent = n_latent or (n_in + 1) // 2
        self.linear1 = nn.Linear(n_in, n_latent)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(n_latent, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


# --------------------
# Layers 
# --------------------

class TransformerLayer(nn.Module):
    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        heads=8,
        use_ff=True,
        use_norm=True
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim // heads)
        self.use_ff = use_ff
        self.use_norm = use_norm
        if self.use_ff:
            self.ff = FeedForward()

    def forward(self, x, register_hook=False):
        if self.use_norm:
            x = x + self.attn(self.norm(x), register_hook=register_hook)
        else:
            x = x + self.attn(x, register_hook=register_hook)

        if self.use_ff:
            x = self.ff(x) + x
        return x

    def get_self_attention(self, x):
        if self.use_norm:
            attn = self.attn.get_self_attention(self.norm(x))
        else:
            attn = self.attn.get_self_attention(x)

        return attn
    

### FOR AttentionMIL: copied from https://github.com/peng-lab/HistoBistro/blob/main/models/aggregators/model_utils.py
class MILAttention(nn.Module):
    """
    A network calculating an embedding's importance weight.
    """
    def __init__(self, n_in: int, n_latent: Optional[int] = None):
        super().__init__()
        n_latent = n_latent or (n_in + 1) // 2
        self.linear1 = nn.Linear(n_in, n_latent)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(n_latent, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


# --------------------
# Positional Embeddings
# --------------------

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat
                                                       ) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads,
                heads, (kernel_size, 1),
                padding=(padding, 0),
                groups=heads,
                bias=False
            )

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v)
        )

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(
                ~(mask[..., None] * mask_landmarks[..., None, :]), mask_value
            )
            sim2.masked_fill_(
                ~(mask_landmarks[..., None] * mask_landmarks[..., None, :]),
                mask_value
            )
            sim3.masked_fill_(
                ~(mask_landmarks[..., None] * mask[..., None, :]), mask_value
            )

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(
            lambda t: t.softmax(dim=-1), (sim1, sim2, sim3)
        )
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out

class NystromTransformerLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=
            6, # number of moore-penrose iterations 
            residual=
            True, # whether to do an extra residual with the value or not
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

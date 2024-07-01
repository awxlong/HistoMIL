import torch
import torch.nn as nn
from einops import repeat


from utils import Attention, FeedForward, PreNorm

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


class Transformer(BaseAggregator):
    def __init__(
        self,
        *,
        num_classes,
        input_dim=2048,
        dim=512,
        depth=2,
        heads=8,
        mlp_dim=512,
        pool='cls',
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        pos_enc=None,
    ):
        super(BaseAggregator, self).__init__()
        assert pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (class token) or mean (mean pooling)'

        self.projection = nn.Sequential(nn.Linear(input_dim, heads*dim_head, bias=True), nn.ReLU())
        self.mlp_head = nn.Sequential(nn.LayerNorm(mlp_dim), nn.Linear(mlp_dim, num_classes))
        self.transformer = TransformerBlocks(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)
        
        self.pos_enc = pos_enc

    def forward(self, x, coords=None, register_hook=False):
        b, _, _ = x.shape

        x = self.projection(x)

        if self.pos_enc:
            x = x + self.pos_enc(coords)

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)
        x = self.transformer(x, register_hook=register_hook)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(self.norm(x))


if __name__ == "__main__":
    model = Transformer(num_classes=2)
    state_dict = torch.load('MSI_high_CRC_model.pth')

    # Remove the 'model.' prefix from the keys
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('model.', '')
        new_state_dict[new_key] = v

    
    # Get the keys from both dictionaries
    pretrained_keys = set(new_state_dict.keys())
    model_keys = set(model.state_dict().keys())

    # Find keys that are in the pretrained weights but not in the model
    extra_keys = pretrained_keys - model_keys

    # Find keys that are in the model but not in the pretrained weights
    missing_keys = model_keys - pretrained_keys

    # Print the results
    print("Keys in pretrained weights but not in model:")
    for key in extra_keys:
        print(f"  {key}")

    print("\nKeys in model but not in pretrained weights:")
    for key in missing_keys:
        print(f"  {key}")

    # If you want to see the corresponding shapes of the tensors:
    print("\nShapes of extra keys in pretrained weights:")
    for key in extra_keys:
        print(f"  {key}: {state_dict[key].shape}")

    print("\nShapes of missing keys in model:")
    for key in missing_keys:
        print(f"  {key}: {model.state_dict()[key].shape}")
    pos_weight = new_state_dict['criterion.pos_weight'].item()
    print(f"Positive class weight used in training: {pos_weight}")
    # pdb.set_trace()
    # Load the modified state dictionary into your model
    model.load_state_dict(new_state_dict)
    model.load_state_dict(torch.load('MSI_high_CRC_model.pth'), strict=False)
    # pdb.set_trace()

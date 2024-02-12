import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.module import Attention, PreNorm, FeedForward, CrossAttention
import numpy as np



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MultiScaleTransformerEncoder(nn.Module):

    def __init__(self, axial_dim = 96, axial_depth = 4, axial_heads =3, axial_dim_head = 32, axial_mlp_dim = 384,
                 coronal_dim = 192, coronal_depth = 1, coronal_heads = 3, coronal_dim_head = 64, coronal_mlp_dim = 768,
                 cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0.):
        super().__init__()
        self.transformer_enc_axial = Transformer(axial_dim, axial_depth, axial_heads, axial_dim_head, axial_mlp_dim)
        self.transformer_enc_coronal = Transformer(coronal_dim, coronal_depth, coronal_heads, coronal_dim_head, coronal_mlp_dim)

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(axial_dim, coronal_dim),
                nn.Linear(coronal_dim, axial_dim),
                PreNorm(coronal_dim, CrossAttention(coronal_dim, heads = cross_attn_heads, dim_head = coronal_dim_head, dropout = dropout)),
                nn.Linear(coronal_dim, axial_dim),
                nn.Linear(axial_dim, coronal_dim),
                PreNorm(axial_dim, CrossAttention(axial_dim, heads = cross_attn_heads, dim_head = axial_dim_head, dropout = dropout)),
            ]))

    def forward(self, xa, xc):

        xa = self.transformer_enc_axial(xa)
        xc = self.transformer_enc_coronal(xc)

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            axial_class = xa[:, 0]
            x_axial = xa[:, 1:]
            coronal_class = xc[:, 0]
            x_coronal = xc[:, 1:]

            # Cross Attn for coronal Patch
            cal_q = f_ls(coronal_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_axial), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xc = torch.cat((cal_out, x_coronal), dim=1)

            # Cross Attn for axialer Patch
            cal_q = f_sl(axial_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_coronal), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xa = torch.cat((cal_out, x_axial), dim=1)

        return xa, xc


class CACTUS(nn.Module): # MODIFIER PARAMS ICI
    def __init__(self, image_size, channels, num_classes, patch_size_axial = 16, patch_size_coronal = 16, axial_dim = 192,
                 coronal_dim = 192, axial_depth = 4, coronal_depth = 4, cross_attn_depth = 1, multi_scale_enc_depth = 3,
                 heads = 3, pool = 'cls', dropout = 0., emb_dropout = 0., scale_dim = 4):
        super().__init__()

        assert image_size % patch_size_axial == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_axial = (image_size // patch_size_axial) ** 2
        patch_dim_axial = channels * patch_size_axial ** 2

        assert image_size % patch_size_coronal == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches_coronal = (image_size // patch_size_coronal) ** 2
        patch_dim_coronal = channels * patch_size_coronal ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        self.to_patch_embedding_axial = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size_axial, p2 = patch_size_axial),
            nn.Linear(patch_dim_axial, axial_dim),
        )

        self.to_patch_embedding_coronal = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size_coronal, p2=patch_size_coronal),
            nn.Linear(patch_dim_coronal, coronal_dim),
        )

        self.pos_embedding_axial = nn.Parameter(torch.randn(1, num_patches_axial + 1, axial_dim))
        self.cls_token_axial = nn.Parameter(torch.randn(1, 1, axial_dim))
        self.dropout_axial = nn.Dropout(emb_dropout)

        self.pos_embedding_coronal = nn.Parameter(torch.randn(1, num_patches_coronal + 1, coronal_dim))
        self.cls_token_coronal = nn.Parameter(torch.randn(1, 1, coronal_dim))
        self.dropout_coronal = nn.Dropout(emb_dropout)

        self.multi_scale_transformers = nn.ModuleList([])
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(MultiScaleTransformerEncoder(axial_dim=axial_dim, axial_depth=axial_depth,
                                                                              axial_heads=heads, axial_dim_head=axial_dim//heads,
                                                                              axial_mlp_dim=axial_dim*scale_dim,
                                                                              coronal_dim=coronal_dim, coronal_depth=coronal_depth,
                                                                              coronal_heads=heads, coronal_dim_head=coronal_dim//heads,
                                                                              coronal_mlp_dim=coronal_dim*scale_dim,
                                                                              cross_attn_depth=cross_attn_depth, cross_attn_heads=heads,
                                                                              dropout=dropout))

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head_axial = nn.Sequential(
            nn.LayerNorm(axial_dim),
            nn.Linear(axial_dim, num_classes)
        )

        self.mlp_head_coronal = nn.Sequential(
            nn.LayerNorm(coronal_dim),
            nn.Linear(coronal_dim, num_classes)
        )

        # Custom parameter for learnable weighted sum
        self.W = nn.Parameter(torch.rand(2))


    def forward(self, img):

        xa = self.to_patch_embedding_axial(img[:, :3, :, :]) # (axial patch)
        b, n, _ = xa.shape

        cls_token_axial = repeat(self.cls_token_axial, '() n d -> b n d', b = b)
        xa = torch.cat((cls_token_axial, xa), dim=1)
        xa += self.pos_embedding_axial[:, :(n + 1)]
        xa = self.dropout_axial(xa)

        xc = self.to_patch_embedding_coronal(img[:, 3:, :, :]) # (coronal patch)
        b, n, _ = xc.shape

        cls_token_coronal = repeat(self.cls_token_coronal, '() n d -> b n d', b=b)
        xc = torch.cat((cls_token_coronal, xc), dim=1)
        xc += self.pos_embedding_coronal[:, :(n + 1)]
        xc = self.dropout_coronal(xc)

        for multi_scale_transformer in self.multi_scale_transformers:
            xa, xc = multi_scale_transformer(xa, xc)
        
        xa = xa.mean(dim = 1) if self.pool == 'mean' else xa[:, 0]
        xc = xc.mean(dim = 1) if self.pool == 'mean' else xc[:, 0]

        xa = self.mlp_head_axial(xa)
        xc = self.mlp_head_coronal(xc)

        # Sum of sagittal and coronal matrices
        x = xa + xc

        # # Learnable weighted sum
        # wS, wC = self.W
        # x = wS*xa + wC*xc

        return x
    
    

if __name__ == "__main__":
    
    img = torch.ones([1, 6, 32, 32])

    model = CACTUS(
            image_size = 32,
            channels = 3,
            num_classes = 2,
            patch_size_axial = 8, 
            patch_size_coronal = 8, 
            axial_dim = 96,
            coronal_dim = 96, 
            axial_depth = 2, 
            coronal_depth = 2
            )

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]

    

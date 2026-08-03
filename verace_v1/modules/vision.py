"""
Vision Encoder Module for Verace V1
Patch-based ViT encoder with 2x2 spatial token merging into the language model's hidden width.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VeraceVisionBlock(nn.Module):
    def __init__(self, embed_dim: int = 1152, num_heads: int = 12):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, bias=False, batch_first=True)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = res + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VeraceVisionEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1152,
        num_layers: int = 27,
        num_heads: int = 12,
        patch_size: int = 14,
        in_channels: int = 3,
        projector_dim: int = 16384
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        
        self.blocks = nn.ModuleList([
            VeraceVisionBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.RMSNorm(embed_dim)
        
        self.projector = nn.Sequential(
            nn.Linear(embed_dim * 4, projector_dim, bias=False),
            nn.GELU(),
            nn.Linear(projector_dim, projector_dim, bias=False)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, c, h, w = images.shape
        patches = self.patch_embed(images)
        hp, wp = patches.shape[2], patches.shape[3]
        x = patches.flatten(2).transpose(1, 2)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        x_2d = x.view(b, hp, wp, self.embed_dim)
        if hp % 2 != 0 or wp % 2 != 0:
            x_2d = F.pad(x_2d, (0, 0, 0, wp % 2, 0, hp % 2))
            hp_pad, wp_pad = x_2d.shape[1], x_2d.shape[2]
        else:
            hp_pad, wp_pad = hp, wp
            
        x_grouped = x_2d.view(b, hp_pad // 2, 2, wp_pad // 2, 2, self.embed_dim)
        x_grouped = x_grouped.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_shuffled = x_grouped.view(b, (hp_pad // 2) * (wp_pad // 2), self.embed_dim * 4)
        
        visual_tokens = self.projector(x_shuffled)
        return visual_tokens

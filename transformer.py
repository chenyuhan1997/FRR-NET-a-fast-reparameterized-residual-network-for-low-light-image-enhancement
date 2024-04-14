from einops import rearrange
from typing import Optional, List
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#-------------------------------------------------------------------------------------------------------------
class GeLu(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))
        return x * cdf
#------------------------------------------------------------------------------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn, layer=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
#-------------------------------------------------------------------------------------------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
#------------------------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.pos = None
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        if self.pos is not None:
            pos = rearrange(self.pos, 'b n (h d) -> b h n d', h=h)
            q += pos
            k += pos
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Attention_Dec(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., out=None):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.pos = None

        self.to_qv = nn.Linear(dim, dim * 2, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if out is  None else nn.Sequential(
            nn.Linear(dim, out),
            nn.Dropout(dropout)
        )

    def forward(self, x, tar, mask=None):
        b, n, _, h = *x.shape, self.heads
        qvk = self.to_qv(x).chunk(2, dim=-1)
        qvk += (self.to_k(tar),)
        q, v, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qvk)
        if self.pos is not None:
            pos = rearrange(self.pos, 'b n (h d) -> b h n d', h=h)
            q += pos
            k += pos

        dots = torch.einsum('bhid,bhjd->bhij', k, q) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GeLu(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, out_dim=None):

        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
        self.out = nn.Sequential(
            GeLu(),
            nn.Linear(dim, out_dim),
            nn.Dropout(dropout)
        ) if out_dim is not None else None

    def forward(self, x, mask=None, pos=None):

        for attn, ff in self.layers:
            if pos is not None:
                for m in attn.modules():
                    if isinstance(m, Attention):
                        setattr(m, 'pos', pos)
            x = attn(x, mask=mask)
            x = ff(x)
        if self.out is not None:
            x = self.out(x)
        return x

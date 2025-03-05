import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import itertools
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite, DropPath, to_2tuple
import numpy as np
import itertools
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Any, Optional
def val2list(x: list | tuple | Any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)

def get_same_padding(kernel_size: int) -> int:
    assert kernel_size % 2 > 0, "Kernel size should be an odd number"
    return kernel_size // 2 

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        dropout: float = 0,
        norm="bn2d",
        act_func="relu"
    ):
        super().__init__()
        padding = get_same_padding(kernel_size) * dilation
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=use_bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_bias: bool = False,
    ):
        super().__init__()
        self.depth_conv = ConvLayer(
            in_channels, in_channels, kernel_size, stride, groups=in_channels, use_bias=use_bias
        )
        self.point_conv = ConvLayer(in_channels, out_channels, 1, use_bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

# Mobile Bottleneck Convolution (MBConv)
class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 4,
        reduction: int = 4,  # For SE block
        use_bias: bool = False,
    ):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.inverted_conv = ConvLayer(in_channels, mid_channels, 1, 1, use_bias=use_bias) if expand_ratio > 1 else nn.Identity()
        self.depth_conv = ConvLayer(mid_channels, mid_channels, kernel_size, stride, groups=mid_channels, use_bias=use_bias)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // reduction, mid_channels, 1),
            nn.Sigmoid()
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=use_bias),
            nn.BatchNorm2d(out_channels)
        )  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = x * self.se(x)
        x = self.point_conv(x)
        if self.use_residual:
            x = x + identity
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        use_bias: bool = False,
    ):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        self.spatial_conv = ConvLayer(in_channels, mid_channels, kernel_size, stride, use_bias=use_bias)
        self.point_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=use_bias),
            nn.BatchNorm2d(out_channels)
        )  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x

class Conv2d_BN(nn.Sequential):
    def __init__(self, in_channels, out_channels, ks=1, stride=1, pad=0, dilation=1, groups=1,resolution=None):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, ks, stride, pad, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int, input_resolution: int):
        super().__init__()
        hid_dim = dim * 4
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(self.se(x))
        x = self.conv3(x)
        return x
    
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
    
class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        return x + self.m(x)


class FFN(nn.Module):
    def __init__(self, ed: int, h: int, resolution: int):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.pw2 = Conv2d_BN(h, ed)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(x)
        x = self.relu(x)
        x = self.pw2(x)
        return x


class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0: # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1) # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        
        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio, 
                                resolution=window_resolution,
                                kernels=kernels,)

    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))
               
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)
            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.permute(0, 3, 1, 2)
        return x


class EfficientViTBlock(torch.nn.Module):    
    """ A basic EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, type: str, ed: int, kd: int, nh=8, ar=4, resolution=14, window_resolution=7, kernels=[5, 5, 5, 5]):
        super().__init__()
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed ))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))
        self.mixer = Residual(LocalWindowAttention(ed, kd, nh, ar, resolution, window_resolution, kernels)) if type == 's' else nn.Identity()
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))

class EfficientViT(nn.Module):
    def __init__(self, img_size=640,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 # Use down_ops to indicate downsampling between stages.
                 # Here, a down_op of ['subsample', 2] stride 2.
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False,
                 frozen_stages=0,
                 pretrained=None):
        super().__init__()

        # Overlap Patch Embedding
        
        resolution = img_size
        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1, resolution=resolution // 2),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, resolution=resolution // 4),
            nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, resolution=resolution // 8)
        )
        
        # Apply DSConv after patch embedding.
        
        self.post_patch_dsconv = DSConv(embed_dim[0], embed_dim[0], kernel_size=3, stride=1)

        # Stages
        resolution = img_size // patch_size  
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        
        # EfficientViTBlock layers.
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        self.blocks = [self.blocks1, self.blocks2, self.blocks3]

        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
        zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            stride = do[1] if do[0] == 'subsample' else 1
            if stride > 1:
                resolution = (resolution - 1) // stride + 1
            blk = self.blocks[i]  # block list (blocks1, blocks2, blocks3)

            # For stage 0, use embed_dim[0]; for later stages, use the previous stage's channels as input.
            in_channels = embed_dim[i]
            out_channels = ed  
            blk.append(
                MBConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,  # Downsample if specified in down_ops
                    expand_ratio=4
                )
            )
            # Append EfficientViTBlock layers for this stage.
            for d in range(dpth):
                blk.append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                # Build EfficientViT downsample block
                blk = eval('self.blocks' + str(i+2))
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(torch.nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)),
                ))
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_
                blk.append(torch.nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution)),
                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)),
                ))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

        # Freeze patch embedding if needed.
        self.frozen_stages = frozen_stages
        self._freeze_stages()
        if pretrained is not None:
            self.init_weights(pretrained=pretrained)
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
    
    def init_weights(self, pretrained=None):
        if pretrained is not None:
            # Load the checkpoint file path
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Load the checkpoint in device.
            checkpoint = torch.load(pretrained, map_location=device)
            # If the checkpoint contains a 'state_dict' key, extract it
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            # Load the checkpoint into the model
            msg = self.load_state_dict(checkpoint, strict=False)
            print("Pretrained weights loaded with message:", msg)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}
    
    def train(self, mode=True):
        super(EfficientViT, self).train(mode)
        self._freeze_stages()
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
    


    def forward(self, x):
        x = self.patch_embed(x)
        x = self.post_patch_dsconv(x)
        feat_stage1 = self.blocks1(x)
        feat_stage2 = self.blocks2(feat_stage1)
        feat_stage3 = self.blocks3(feat_stage2)
        return {"stage2": feat_stage1, "stage3": feat_stage2, "stage4": feat_stage3}
    
    
EfficientViT_m0 = {
        'patch_size': 16,
        'embed_dim': [64, 128, 192],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m1 = {
        'patch_size': 16,
        'embed_dim': [128, 144, 192],
        'depth': [1, 2, 3],
        'num_heads': [2, 3, 3],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m2 = {
        'patch_size': 16,
        'embed_dim': [128, 192, 224],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 2],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m3 = {
        'patch_size': 16,
        'embed_dim': [128, 240, 320],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],
    }

EfficientViT_m4 = {
        'patch_size': 16,
        'embed_dim': [128, 256, 384],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m5 = {
        'patch_size': 16,
        'embed_dim': [192, 288, 384],
        'depth': [1, 3, 4],
        'num_heads': [3, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }
EfficientViT_configs = {
    'EfficientViT_m0': EfficientViT_m0,
    'EfficientViT_m1': EfficientViT_m1,
    'EfficientViT_m2': EfficientViT_m2,
    'EfficientViT_m3': EfficientViT_m3,
    'EfficientViT_m4': EfficientViT_m4,
    'EfficientViT_m5': EfficientViT_m5,
}


import torch
import torch.nn.functional as F
import torch.nn as nn
from efficientvit import ConvLayer, MBConv, FusedMBConv
from typing import Optional
from efficientvit import EfficientViT
from torch.nn import Hardswish
from typing import Any, Optional, List, Dict
def IdentityLayer():
    """Return an identity module."""
    return nn.Identity()


class ResidualBlock(nn.Module):
    """
    A simple residual block that adds the output of a block to its input.
    """
    def __init__(self, block, skip):
        super().__init__()
        self.block = block
        self.skip = skip

    def forward(self, x):
        return self.block(x) + self.skip(x)
                                         
class OpSequential(nn.Sequential):
    """
    A sequential module that ignores None entries.
    """
    def forward(self, input):
        for module in self:
            if module is not None:
                input = module(input)
        return input

class EfficientViTSeg(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, head: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> dict:
        feed_dict = self.backbone(x)
        return self.head(feed_dict)

# InstanceSegHead :
class InstanceSegHead(torch.nn.Module):
    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        stride_list: list[int],  # Unused but kept for compatibility
        head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        final_expand: Optional[float],
        emb_dim: int,
        n_classes: int,
        dropout=0,
        norm="bn2d",
        act_func="hswish",
    ):
        super().__init__()
        # Process multi-scale features without upsampling here
        self.inputs = torch.nn.ModuleDict()
        for fid, in_channel in zip(fid_list, in_channel_list):
            self.inputs[fid] = ConvLayer(in_channel, head_width, kernel_size=1, norm=norm, act_func=None)
        
        # Middle fusion
        middle_layers = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(head_width, head_width, expand_ratio=expand_ratio, stride=1)
            elif middle_op == "fmbconv":
                block = FusedMBConv(head_width, head_width, expand_ratio=expand_ratio, stride=1, norm=norm, act_func=(act_func, None))
            else:
                raise NotImplementedError(f"middle_op {middle_op} not implemented")
            middle_layers.append(ResidualBlock(block, IdentityLayer()))
        self.middle = OpSequential(*middle_layers)
        
        # Final branch for foreground segmentation
        final_layers_fg = [
            ConvLayer(head_width, n_classes, kernel_size=1, norm=None, act_func=None)
        ]
        self.final_fg = OpSequential(*final_layers_fg)
        
        # Final branch for instance embeddings
        final_layers_emb = []
        if final_expand is not None:
            final_layers_emb.append(ConvLayer(head_width, int(head_width * final_expand), kernel_size=1, norm=norm, act_func=act_func))
        final_layers_emb.append(
            ConvLayer(
                head_width if final_expand is None else int(head_width * final_expand),
                emb_dim,
                kernel_size=1,
                norm=None,
                act_func=None
            )
        )
        self.final_emb = OpSequential(*final_layers_emb)
    
    def forward(self, x: dict) -> dict:
        target_size = (80, 80)  # Hardcoded for img_size=640 and head_stride=8
        features = []
        for fid, layer in self.inputs.items():
            feat = layer(x[fid])  # Apply ConvLayer only
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            features.append(feat)
        fused = torch.stack(features, dim=0).sum(dim=0)  # [B, head_width, 80, 80]
        fused = self.middle(fused)
        seg_logits = self.final_fg(fused)  # [B, n_classes, 80, 80]
        embedding = self.final_emb(fused)
        return {"seg_logits": seg_logits, "embedding": embedding}


# Define the helper function to create the instance segmentation model.
def efficientvit_instance_seg_custom(emb_dim: int = 16, n_classes: int = 2, **kwargs):  
    backbone = EfficientViT(**kwargs)
    # instance segmentation head.
    head = InstanceSegHead(
        fid_list=kwargs.get("fid_list", ["stage4", "stage3", "stage2"]),
        in_channel_list=kwargs.get("in_channel_list", [192, 128, 64]),
        stride_list=kwargs.get("stride_list", [32, 16, 8]),
        head_stride=kwargs.get("head_stride", 8),
        head_width=kwargs.get("head_width", 64),
        head_depth=kwargs.get("head_depth", 3),
        expand_ratio=kwargs.get("expand_ratio", 4),
        middle_op=kwargs.get("middle_op", "mbconv"),
        final_expand=kwargs.get("final_expand", 4),
        emb_dim=emb_dim,
        n_classes=n_classes,
        dropout=kwargs.get("dropout", 0),
        norm=kwargs.get("norm", "bn2d"),
        act_func=kwargs.get("act_func", "hswish"),
    )
    model = EfficientViTSeg(backbone, head)
    return model

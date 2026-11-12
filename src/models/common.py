import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), nn.BatchNorm2d(out_channels), nn.GELU())

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SCSEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        self.spatial_gate = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel_gate(x) + x * self.spatial_gate(x)


class TopDownFusionBlock(nn.Module):
    def __init__(self, deep_channels, skip_channels, out_channels):
        super().__init__()
        self.deep_proj = ConvBNAct(deep_channels, out_channels, 1, padding=0)
        self.skip_proj = ConvBNAct(skip_channels, out_channels, 1, padding=0)
        self.fuse = nn.Sequential(DepthwiseSeparableConvBNAct(out_channels * 2, out_channels), SCSEBlock(out_channels), DepthwiseSeparableConvBNAct(out_channels, out_channels))

    def forward(self, deep_feat, skip_feat):
        deep_feat = F.interpolate(deep_feat, size=skip_feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.deep_proj(deep_feat), self.skip_proj(skip_feat)], dim=1))


class BoundaryRefineBlock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.gate = nn.Sequential(DepthwiseSeparableConvBNAct(channels + 2, hidden_channels), nn.Conv2d(hidden_channels, channels, 1), nn.Sigmoid())
        self.fuse = nn.Sequential(DepthwiseSeparableConvBNAct(channels + 2, channels), SCSEBlock(channels), DepthwiseSeparableConvBNAct(channels, channels))

    def forward(self, feat, mask_logits, boundary_logits):
        mask_prob = torch.sigmoid(mask_logits)
        boundary_prob = torch.sigmoid(boundary_logits)
        uncertainty = 1.0 - torch.abs(mask_prob * 2.0 - 1.0)
        gate = self.gate(torch.cat([feat, boundary_prob, uncertainty], dim=1))
        refined = self.fuse(torch.cat([feat * (1.0 + gate * boundary_prob), boundary_prob, uncertainty], dim=1))
        return feat + refined

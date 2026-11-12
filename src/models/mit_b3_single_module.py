import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNAct, DepthwiseSeparableConvBNAct


class ShallowFeatureCalibration(nn.Module):
    def __init__(self, channels, init_scale=-5.0, groups=8):
        super().__init__()
        groups = max(1, min(groups, channels))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.local_mix = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.gate = nn.Sequential(nn.Conv2d(channels + 2, channels, 1, bias=True), nn.Sigmoid())
        self.scale_logit = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, feature, low_contrast_prior, edge_prior):
        gate = self.gate(torch.cat([feature, low_contrast_prior, edge_prior], dim=1))
        return feature + torch.sigmoid(self.scale_logit) * gate * self.local_mix(feature)


class MiTB3SingleModuleNet(nn.Module):
    def __init__(self, module="c2_residual", encoder_name="mit_b3", encoder_weights="imagenet", refine_channels=32, residual_scale=0.25):
        super().__init__()
        import segmentation_models_pytorch as smp
        weights = encoder_weights if encoder_weights and encoder_weights.lower() not in {"none", "random"} else None
        self.module = module
        self.residual_scale = residual_scale
        self.unet = smp.Unet(encoder_name=encoder_name, encoder_weights=weights, in_channels=3, classes=1, activation=None)
        valid_channels = [c for c in self.unet.encoder.out_channels if c > 0][-4:]
        c1_channels, c2_channels, c3_channels, c4_channels = valid_channels[0], valid_channels[1], valid_channels[2], valid_channels[3]
        decoder_channels = self.unet.segmentation_head[0].in_channels
        self.decoder_proj = ConvBNAct(decoder_channels, refine_channels, 1, padding=0)
        if module in {"feature_calibration", "feature_calibration_boundary_aux"}:
            self.feature_calibrators = nn.ModuleList([ShallowFeatureCalibration(c1_channels), ShallowFeatureCalibration(c2_channels)])
        else:
            self.feature_calibrators = None
        if module in {"c2_residual", "low_contrast_residual"}:
            self.feature_proj = ConvBNAct(c2_channels, refine_channels, 1, padding=0)
        elif module == "c3_residual":
            self.feature_proj = ConvBNAct(c3_channels, refine_channels, 1, padding=0)
        elif module == "c4_residual":
            self.feature_proj = ConvBNAct(c4_channels, refine_channels, 1, padding=0)
        else:
            self.feature_proj = None
        if module in {"c2_residual", "c3_residual", "c4_residual"}:
            self.fuse = nn.Sequential(DepthwiseSeparableConvBNAct(refine_channels * 2 + 1, refine_channels), DepthwiseSeparableConvBNAct(refine_channels, refine_channels))
            self.residual_head = nn.Conv2d(refine_channels, 1, 1)
            self.gamma = nn.Parameter(torch.tensor(-2.0))
        elif module == "low_contrast_residual":
            self.contrast_proj = ConvBNAct(1, refine_channels, 3)
            self.fuse = nn.Sequential(DepthwiseSeparableConvBNAct(refine_channels * 3 + 1, refine_channels), DepthwiseSeparableConvBNAct(refine_channels, refine_channels))
            self.residual_head = nn.Conv2d(refine_channels, 1, 1)
            self.gamma = nn.Parameter(torch.tensor(-2.0))
        elif module in {"boundary_aux", "rim_completion_aux", "tiny_region_recall_aux", "annotation_transition_aux", "feature_calibration_boundary_aux"}:
            self.boundary_head = nn.Sequential(DepthwiseSeparableConvBNAct(decoder_channels, refine_channels), nn.Conv2d(refine_channels, 1, 1))
        elif module == "low_contrast_edge_aux":
            self.contrast_proj = ConvBNAct(1, refine_channels, 3)
            self.low_contrast_edge_head = nn.Sequential(DepthwiseSeparableConvBNAct(decoder_channels + refine_channels, refine_channels), nn.Conv2d(refine_channels, 1, 1))
        elif module == "deep_supervision":
            self.aux_head = nn.Sequential(ConvBNAct(c3_channels, refine_channels, 1, padding=0), nn.Conv2d(refine_channels, 1, 1))
        elif module in {"baseline", "small_sampler", "feature_calibration"}:
            pass
        else:
            raise ValueError(f"Unsupported single module: {module}")

    @property
    def encoder(self):
        return self.unet.encoder

    @property
    def decoder(self):
        return self.unet.decoder

    @property
    def segmentation_head(self):
        return self.unet.segmentation_head

    def local_contrast(self, x, size):
        gray = x.mean(dim=1, keepdim=True)
        local_mean = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        contrast = (gray - local_mean).abs()
        contrast = contrast / (contrast.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6))
        return F.interpolate(contrast, size=size, mode="bilinear", align_corners=False)

    def low_contrast_prior(self, x, size):
        contrast = self.local_contrast(x, size)
        return (1.0 - contrast).clamp(0.0, 1.0)

    def calibrated_features(self, x, features):
        features = list(features)
        valid_indices = [idx for idx, feat in enumerate(features) if feat.ndim == 4 and feat.shape[1] > 0][-4:]
        if self.feature_calibrators is not None:
            for calibrator, idx in zip(self.feature_calibrators, valid_indices[:2]):
                size = features[idx].shape[-2:]
                features[idx] = calibrator(features[idx], self.low_contrast_prior(x, size), self.local_contrast(x, size))
        return features, [features[idx] for idx in valid_indices]

    def residual_forward(self, x, feature, decoder_output, coarse_logits):
        target_size = feature.shape[-2:]
        feat = self.feature_proj(feature)
        decoder_feat = F.interpolate(self.decoder_proj(decoder_output), size=target_size, mode="bilinear", align_corners=False)
        coarse_small = F.interpolate(torch.sigmoid(coarse_logits.detach()), size=target_size, mode="bilinear", align_corners=False)
        parts = [feat, decoder_feat, coarse_small]
        if self.module == "low_contrast_residual":
            parts.insert(1, self.contrast_proj(self.local_contrast(x, target_size)))
        refine_feat = self.fuse(torch.cat(parts, dim=1))
        residual_logits = F.interpolate(self.residual_head(refine_feat), size=x.shape[-2:], mode="bilinear", align_corners=False)
        main_logits = coarse_logits + self.residual_scale * torch.sigmoid(self.gamma) * residual_logits
        return {"main": main_logits, "coarse": coarse_logits, "residual": residual_logits}

    def forward(self, x):
        features = self.unet.encoder(x)
        features, valid_features = self.calibrated_features(x, features)
        c2, c3, c4 = valid_features[1], valid_features[2], valid_features[3]
        decoder_output = self.unet.decoder(features)
        coarse_logits = self.unet.segmentation_head(decoder_output)
        if self.module in {"baseline", "small_sampler", "feature_calibration"}:
            return coarse_logits
        if self.module in {"boundary_aux", "feature_calibration_boundary_aux"}:
            return {"main": coarse_logits, "boundary": self.boundary_head(decoder_output)}
        if self.module == "rim_completion_aux":
            return {"main": coarse_logits, "rim": self.boundary_head(decoder_output)}
        if self.module == "tiny_region_recall_aux":
            return {"main": coarse_logits, "tiny": self.boundary_head(decoder_output)}
        if self.module == "annotation_transition_aux":
            return {"main": coarse_logits, "transition": self.boundary_head(decoder_output)}
        if self.module == "low_contrast_edge_aux":
            prior = self.low_contrast_prior(x, decoder_output.shape[-2:])
            edge_logits = self.low_contrast_edge_head(torch.cat([decoder_output, self.contrast_proj(prior)], dim=1))
            return {"main": coarse_logits, "low_contrast_edge": edge_logits, "low_contrast_prior": prior.detach()}
        if self.module == "deep_supervision":
            aux_logits = F.interpolate(self.aux_head(c3), size=x.shape[-2:], mode="bilinear", align_corners=False)
            return {"main": coarse_logits, "aux": [aux_logits]}
        if self.module == "c2_residual":
            return self.residual_forward(x, c2, decoder_output, coarse_logits)
        if self.module == "c3_residual":
            return self.residual_forward(x, c3, decoder_output, coarse_logits)
        if self.module in {"c4_residual", "low_contrast_residual"}:
            return self.residual_forward(x, c4 if self.module == "c4_residual" else c2, decoder_output, coarse_logits)
        raise RuntimeError(f"Unhandled module: {self.module}")

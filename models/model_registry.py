"""
Model registry for fair multi-architecture comparison.

All registered models share:
- input: [B, 4, 128, 128]
- output logits: [B, 4, 128, 128]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.architecture import AttentionUNet as MobileAttentionUNet


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _AttentionGateLite(nn.Module):
    def __init__(self, channels_g: int, channels_x: int, channels_out: int):
        super().__init__()
        self.w_g = nn.Conv2d(channels_g, channels_out, kernel_size=1, bias=False)
        self.w_x = nn.Conv2d(channels_x, channels_out, kernel_size=1, bias=False)
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(channels_out, 1, 1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(self.w_g(g) + self.w_x(x))
        return x * alpha


class BasicUNet(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 4, base: int = 32, pretrained: bool = False):
        super().__init__()
        self.enc1 = _conv_block(in_channels, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.enc4 = _conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _conv_block(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = _conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _conv_block(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class AttentionUNetClassic(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 4, base: int = 32, pretrained: bool = False):
        super().__init__()
        self.enc1 = _conv_block(in_channels, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.enc4 = _conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _conv_block(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.att4 = _AttentionGateLite(base * 8, base * 8, base * 8)
        self.dec4 = _conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.att3 = _AttentionGateLite(base * 4, base * 4, base * 4)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.att2 = _AttentionGateLite(base * 2, base * 2, base * 2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.att1 = _AttentionGateLite(base, base, base)
        self.dec1 = _conv_block(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self.att4(d4, e4)], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.att3(d3, e3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(d2, e2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(d1, e1)], dim=1))
        return self.out(d1)


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class ResUNet(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 4, base: int = 32, pretrained: bool = False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = _ResBlock(in_channels, base)
        self.e2 = _ResBlock(base, base * 2)
        self.e3 = _ResBlock(base * 2, base * 4)
        self.e4 = _ResBlock(base * 4, base * 8)
        self.b = _ResBlock(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.d4 = _ResBlock(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = _ResBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = _ResBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = _ResBlock(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(b), e4], dim=1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    builder: Callable[..., nn.Module]
    default_pretrained: bool = False


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "basic_unet": ModelSpec("basic_unet", "Basic U-Net", BasicUNet, False),
    "attention_unet": ModelSpec("attention_unet", "Attention U-Net", AttentionUNetClassic, False),
    "resunet": ModelSpec("resunet", "ResUNet", ResUNet, False),
    "mobilenet_attention_unet": ModelSpec(
        "mobilenet_attention_unet",
        "MobileNetV2 Attention U-Net",
        lambda in_channels=4, num_classes=4, pretrained=True: MobileAttentionUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            use_lightweight_aspp=False,
            use_attention=True,
            use_aspp=True,
        ),
        True,
    ),
    "ablation_full": ModelSpec(
        "ablation_full",
        "Ablation Full (Ours)",
        lambda in_channels=4, num_classes=4, pretrained=True: MobileAttentionUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            use_lightweight_aspp=True,
            use_attention=True,
            use_aspp=True,
        ),
        True,
    ),
    "ablation_no_attention": ModelSpec(
        "ablation_no_attention",
        "Ablation No Attention",
        lambda in_channels=4, num_classes=4, pretrained=True: MobileAttentionUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            use_lightweight_aspp=True,
            use_attention=False,
            use_aspp=True,
        ),
        True,
    ),
    "ablation_no_aspp": ModelSpec(
        "ablation_no_aspp",
        "Ablation No ASPP",
        lambda in_channels=4, num_classes=4, pretrained=True: MobileAttentionUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            use_lightweight_aspp=True,
            use_attention=True,
            use_aspp=False,
        ),
        True,
    ),
    "ablation_original_aspp": ModelSpec(
        "ablation_original_aspp",
        "Ablation Original ASPP",
        lambda in_channels=4, num_classes=4, pretrained=True: MobileAttentionUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            use_lightweight_aspp=False,
            use_attention=True,
            use_aspp=True,
        ),
        True,
    ),
}


def list_model_keys() -> Iterable[str]:
    """Internal helper kept for backward compatibility."""
    return MODEL_REGISTRY.keys()


def list_models() -> Iterable[str]:
    """
    Public helper for external callers.
    
    Example:
        from models.model_registry import list_models
    """
    return list(MODEL_REGISTRY.keys())


def get_model_display_name(model_key: str) -> str:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}")
    return MODEL_REGISTRY[model_key].display_name


def create_model(
    model_key: str,
    in_channels: int = 4,
    num_classes: int = 4,
    pretrained: bool | None = None,
) -> nn.Module:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}. Choices: {', '.join(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_key]
    use_pretrained = spec.default_pretrained if pretrained is None else pretrained
    sig = inspect.signature(spec.builder)
    if "pretrained" in sig.parameters:
        return spec.builder(in_channels=in_channels, num_classes=num_classes, pretrained=use_pretrained)
    return spec.builder(in_channels=in_channels, num_classes=num_classes)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_checkpoint_weights(model: nn.Module, checkpoint_obj: dict | nn.Module | torch.Tensor) -> None:
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        model.load_state_dict(checkpoint_obj["model_state_dict"])
    elif isinstance(checkpoint_obj, dict):
        model.load_state_dict(checkpoint_obj)
    else:
        model.load_state_dict(checkpoint_obj)

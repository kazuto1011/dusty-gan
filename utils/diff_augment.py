# adopted from
# https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py
# modified rand_translation()

# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def augment(x, policy="color,translation,cutout"):
    if policy:
        for p in policy.split(","):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        x = x.contiguous()
    return x


def rand_brightness(x, band=0.5, p=1.0):
    B, _, _, _ = x.shape
    device = x.device
    factor = torch.empty((B, 1, 1, 1), device=device)
    brightness = factor.bernoulli_(p=p) * factor.uniform_(-1, 1) * band
    y = x + brightness
    return y


def rand_saturation(x, band=1.0, p=1.0):
    B, _, _, _ = x.shape
    device = x.device
    factor = torch.empty((B, 1, 1, 1), device=device)
    x_mean = x.mean(dim=1, keepdim=True)
    saturation = factor.bernoulli_(p=p) * factor.uniform_(-1, 1) * band + 1.0
    y = torch.lerp(x_mean, x, saturation)
    return y


def rand_contrast(x, band=0.5, p=1.0):
    B, _, _, _ = x.shape
    device = x.device
    factor = torch.empty((B, 1, 1, 1), device=device)
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    contrast = factor.bernoulli_(p=p) * factor.uniform_(-1, 1) * band + 1.0
    y = torch.lerp(x_mean, x, contrast)
    return y


def rand_translation(x, ratio=(0.0, 1.0 / 8.0), p=1.0):
    B, C, H_orig, W_orig = x.shape
    device = x.device

    pad_w = W_orig // 2
    pad_h = H_orig // 2
    x = F.pad(x, (pad_w, pad_w, 0, 0), "circular")
    x = F.pad(x, (0, 0, pad_h, pad_h), "constant")

    B, C, H, W = x.shape

    ratio_h, ratio_w = _pair(ratio)
    shift_h, shift_w = int(H * ratio_h / 2 + 0.5), int(W * ratio_w / 2 + 0.5)
    translation_h = torch.randint(-shift_h, shift_h + 1, size=[B, 1, 1], device=device)
    translation_w = torch.randint(-shift_w, shift_w + 1, size=[B, 1, 1], device=device)
    grid_batch, grid_h, grid_w = torch.meshgrid(
        torch.arange(B, dtype=torch.long, device=device),
        torch.arange(H, dtype=torch.long, device=device),
        torch.arange(W, dtype=torch.long, device=device),
    )
    grid_h = torch.clamp(grid_h + translation_h + 1, 0, H + 1)
    grid_w = torch.clamp(grid_w + translation_w + 1, 0, W + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    y = (
        x_pad.permute(0, 2, 3, 1)
        .contiguous()[grid_batch, grid_h, grid_w]
        .permute(0, 3, 1, 2)
    )

    mask = torch.empty(B, device=device).bernoulli_(p=p).bool()
    y[~mask] = x[~mask]

    y = y[:, :, pad_h : pad_h + H_orig, pad_w : pad_w + W_orig]
    return y


def rand_cutout(x, ratio=0.5, p=1.0):
    B, C, H, W = x.shape
    device = x.device
    cut_h, cut_w = int(H * ratio + 0.5), int(W * ratio + 0.5)
    offset_x = torch.randint(0, H + (1 - cut_h % 2), size=[B, 1, 1], device=device)
    offset_y = torch.randint(0, W + (1 - cut_w % 2), size=[B, 1, 1], device=device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(B, dtype=torch.long, device=device),
        torch.arange(cut_h, dtype=torch.long, device=device),
        torch.arange(cut_w, dtype=torch.long, device=device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cut_h // 2, min=0, max=H - 1)
    grid_y = torch.clamp(grid_y + offset_y - cut_w // 2, min=0, max=W - 1)
    mask = torch.ones(B, H, W, dtype=x.dtype, device=device)
    mask[grid_batch, grid_x, grid_y] = 0
    y = x * mask.unsqueeze(1)

    mask = torch.empty(B, device=device).bernoulli_(p=p).bool()
    y[~mask] = x[~mask]

    return y


AUGMENT_FNS = {
    "brightness": rand_brightness,
    "saturation": rand_saturation,
    "contrast": rand_contrast,
    "translation": rand_translation,
    "cutout": rand_cutout,
}


class DiffAugment(nn.Module):
    def __init__(self, policy=None, p=1.0):
        super().__init__()
        if policy is None:
            self.policy = [
                "brightness",
                "saturation",
                "contrast",
                "translation",
                "cutout",
            ]
        else:
            self.policy = policy
        self.p = p

    def forward(self, x):
        for p in self.policy:
            x = AUGMENT_FNS[p](x, p=self.p)
        return x

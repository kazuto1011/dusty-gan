import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple


class Pad(nn.Module):
    def __init__(self, padding, horizontal="constant", vertical="constant"):
        super().__init__()
        self.padding = _quadruple(padding)
        self.horizontal = horizontal
        self.vertical = vertical

    def forward(self, h):
        left, right, top, bottom = self.padding
        h = F.pad(h, (left, right, 0, 0), mode=self.horizontal)
        h = F.pad(h, (0, 0, top, bottom), mode=self.vertical)
        return h

    def extra_repr(self):
        return f"padding={self.padding}, horizontal={self.horizontal}, vertical={self.vertical}"


class Blur(nn.Module):
    def __init__(
        self,
        filter_type=[1, 3, 3, 1],
        stride=1,
        padding=1,
        factor=1,
        direction="vh",
        ring=True,
    ):
        super().__init__()
        self.filter_type = filter_type
        self.stride = stride
        self.padding = _quadruple(padding)
        self.factor = factor
        self.direction = direction

        self.pad = Pad(
            padding=self.padding,
            horizontal="circular" if ring else "reflect",
            vertical="reflect",
        )

        kernel = torch.tensor(self.filter_type, dtype=torch.float32)
        if direction == "vh":
            kernel = torch.outer(kernel, kernel)
        elif direction == "v":
            kernel = kernel[:, None]
        elif direction == "h":
            kernel = kernel[None, :]
        else:
            raise ValueError
        kernel /= kernel.sum()
        if factor > 1:
            kernel *= factor ** 2
        self.register_buffer("kernel", kernel[None, None])

    def forward(self, h):
        _, C, _, _ = h.shape
        kernel = self.kernel.repeat(C, 1, 1, 1)
        h = self.pad(h)
        h = F.conv2d(h, kernel, stride=self.stride, padding=0, groups=C)
        return h

    def extra_repr(self):
        return f"filter_type={self.filter_type}, stride={self.stride}, padding={self.padding}, factor={self.factor}, direction={self.direction}"


class BlurVH(nn.Module):
    """
    vertical/horizontal antialiasing from NR-GAN:
    https://arxiv.org/abs/1911.11776
    """

    def __init__(self, ring=True):
        super().__init__()
        self.blur_v = Blur([1, 2, 1], padding=(0, 0, 1, 1), direction="v", ring=ring)
        self.blur_h = Blur([1, 2, 1], padding=(1, 1, 0, 0), direction="h", ring=ring)

    def forward(self, x):
        h_v = self.blur_v(x)
        h_h = self.blur_h(x)
        return torch.cat([h_v, h_h], dim=1)


class FusedLeakyReLU(nn.Module):
    def __init__(self, ch, negative_slope=0.2, scale=math.sqrt(2)):
        super().__init__()
        self.ch = ch
        self.bias = nn.Parameter(torch.zeros(self.ch))
        self.negative_slope = negative_slope
        self.gain = scale

    def forward(self, x):
        if x.ndim == 4:
            bias = self.bias.view(1, self.ch, 1, 1)
        elif x.ndim == 2:
            bias = self.bias
        else:
            raise NotImplementedError
        return F.leaky_relu(x + bias, negative_slope=self.negative_slope) * self.gain

    def extra_repr(self):
        return f"ch={self.ch}, negative_slope={self.negative_slope}, gain={self.gain}"


class EqualLR(nn.Module):
    """
    A wrapper for runtime weight scaling (equalized learning rate).
    https://arxiv.org/abs/1710.10196
    """

    def __init__(self, module, gain: float = 1.0):
        super().__init__()
        self.module = module

        # Runtime scale factor
        self.gain = gain
        fan_in = self.module.weight[0].numel()
        self.scale = 1.0 / math.sqrt(fan_in)

        # Weights are initialized with N(0, 1)
        nn.init.normal_(self.module.weight, 0.0, 1.0)
        if hasattr(self.module, "bias") and self.module.bias is not None:
            nn.init.constant_(self.module.bias, 0.0)

    def forward(self, x):
        return self.module(x * self.scale) * self.gain

    def extra_repr(self):
        return f"gain={self.gain}"

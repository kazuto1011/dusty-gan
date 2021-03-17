import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

# from models.ops.fused_bias_act import FusedLeakyReLU


class Blur(nn.Module):
    def __init__(self, filter_type=[1, 3, 3, 1], padding=1, factor=1, direction="vh"):
        super().__init__()
        self.filter_type = filter_type
        self.padding = _pair(padding)
        self.factor = factor
        self.direction = direction

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
        return F.conv2d(h, kernel, padding=self.padding, groups=C)

    def extra_repr(self):
        return f"filter_type={self.filter_type}, padding={self.padding}, factor={self.factor}, direction={self.direction}"


class FusedLeakyReLU(nn.Module):
    def __init__(self, ch, negative_slope=0.2, scale=math.sqrt(2)):
        super().__init__()
        self.ch = ch
        self.bias = nn.Parameter(torch.zeros(1, self.ch))
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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _quadruple


class Pad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = _quadruple(padding)

    def forward(self, h):
        left, right, top, bottom = self.padding
        h = F.pad(h, (left, right, 0, 0), mode="circular")
        h = F.pad(h, (0, 0, top, bottom), mode="reflect")
        return h

    def extra_repr(self):
        return f"padding={self.padding}"

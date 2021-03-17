import models.ops.common as ops
import models.ops.cylinder as cyl
import torch
from torch import nn


class Proj(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=(4, 16)):
        super().__init__(
            ops.EqualLR(nn.ConvTranspose2d(in_ch, out_ch, kernel, 1, 0, bias=False)),
            ops.FusedLeakyReLU(out_ch),
        )

    def forward(self, x):
        h = x[..., None, None]
        h = super().forward(h)
        return h


class Up(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            cyl.Pad(padding=1),
            ops.EqualLR(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1 + 2, bias=False)),
            ops.FusedLeakyReLU(out_ch),
        )


class Head(nn.Module):
    def __init__(self, in_ch, out_ch={"rgb": 3}):
        super().__init__()
        assert isinstance(out_ch, dict)
        self.in_ch = in_ch
        self.heads = nn.ModuleDict()
        for name, ch in out_ch.items():
            self.heads[name] = nn.Sequential(
                cyl.Pad(padding=1),
                ops.EqualLR(nn.ConvTranspose2d(in_ch, ch, 4, 2, 1 + 2, bias=True)),
            )

    def forward(self, x):
        h = {}
        for name, head in self.heads.items():
            h[name] = head(x)
        return h


class Generator(nn.Sequential):
    def __init__(self, in_ch, out_ch, ch_base=64, ch_max=512, shape=(64, 256)):
        shape_in = (shape[0] >> 4, shape[1] >> 4)
        ch = lambda i: min(ch_base << i, ch_max)
        super().__init__(
            Proj(in_ch, ch(3), shape_in),
            Up(ch(3), ch(2)),
            Up(ch(2), ch(1)),
            Up(ch(1), ch(0)),
            Head(ch(0), out_ch),
        )

    def forward(self, latent):
        h = super().forward(latent)
        h["depth"] = torch.tanh(h["depth"])
        return h


if __name__ == "__main__":
    f = Generator(100, {"depth": 1, "confidence": 2})
    x = torch.randn(5, 100)
    y = f(x)
    print(x.shape)
    for k, v in y.items():
        print(k, v.shape)

import models.ops.common as ops
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
    def __init__(self, in_ch, out_ch, ring=True):
        horizontal = "circular" if ring else "reflect"
        super().__init__(
            ops.Pad(padding=1, horizontal=horizontal, vertical="reflect"),
            ops.EqualLR(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1 + 2, bias=False)),
            ops.FusedLeakyReLU(out_ch),
        )


class Head(nn.Module):
    def __init__(self, in_ch, out_ch={"rgb": 3}, ring=True):
        super().__init__()
        assert isinstance(out_ch, dict)
        self.in_ch = in_ch
        self.heads = nn.ModuleDict()
        horizontal = "circular" if ring else "reflect"
        for name, ch in out_ch.items():
            self.heads[name] = nn.Sequential(
                ops.Pad(padding=1, horizontal=horizontal, vertical="reflect"),
                ops.EqualLR(nn.ConvTranspose2d(in_ch, ch, 4, 2, 1 + 2, bias=True)),
            )

    def forward(self, x):
        h = {}
        for name, head in self.heads.items():
            h[name] = head(x)
        return h


class Generator(nn.Sequential):
    def __init__(
        self,
        in_ch,
        out_ch,
        ch_base=64,
        ch_max=512,
        shape=(64, 256),
        ring=True,
    ):
        shape_in = (shape[0] >> 4, shape[1] >> 4)
        ch = lambda i: min(ch_base << i, ch_max)
        super().__init__(
            Proj(in_ch, ch(3), shape_in),
            Up(ch(3), ch(2), ring),
            Up(ch(2), ch(1), ring),
            Up(ch(1), ch(0), ring),
            Head(ch(0), out_ch, ring),
        )

    def forward(self, latent):
        h = super().forward(latent)
        h["depth"] = torch.tanh(h["depth"])
        return h


class Down(nn.Sequential):
    def __init__(self, in_ch, out_ch, ring=True):
        horizontal = "circular" if ring else "reflect"
        super().__init__(
            ops.Pad(padding=1, horizontal=horizontal, vertical="reflect"),
            ops.EqualLR(nn.Conv2d(in_ch, out_ch, 4, 2, 0, bias=False)),
            ops.FusedLeakyReLU(out_ch),
        )


class Discriminator(nn.Sequential):
    def __init__(self, in_ch, ch_base=64, ch_max=512, shape=(64, 256), ring=True):
        shape_out = (shape[0] >> 4, shape[1] >> 4)
        ch = lambda i: min(ch_base << i, ch_max)
        super().__init__(
            ops.BlurVH(ring),
            Down(in_ch * 2, ch(0), ring),
            Down(ch(0), ch(1), ring),
            Down(ch(1), ch(2), ring),
            Down(ch(2), ch(3), ring),
            ops.EqualLR(nn.Conv2d(ch(3), 1, shape_out, 1, 0)),
        )


if __name__ == "__main__":
    d = Discriminator(1)
    x = torch.randn(5, 1, 64, 256)
    y = d(x)
    print(x.shape)
    print(y.shape)

    g = Generator(100, {"depth": 1, "confidence": 2})
    x = torch.randn(5, 100)
    y = g(x)
    print(x.shape)
    for k, v in y.items():
        print(k, v.shape)

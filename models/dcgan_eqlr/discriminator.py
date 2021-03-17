import models.ops.common as ops
import models.ops.cylinder as cyl
import torch
from torch import nn


class BlurVH(nn.Module):
    """
    vertical/horizontal antialiasing from NR-GAN:
    https://arxiv.org/abs/1911.11776
    """

    def __init__(self):
        super().__init__()
        self.blur_v = ops.Blur([1, 2, 1], padding=(1, 0), direction="v")
        self.blur_h = ops.Blur([1, 2, 1], padding=(0, 1), direction="h")

    def forward(self, x):
        h_v = self.blur_v(x)
        h_h = self.blur_h(x)
        return torch.cat([h_v, h_h], dim=1)


class Down(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            cyl.Pad(padding=1),
            ops.EqualLR(nn.Conv2d(in_ch, out_ch, 4, 2, 0, bias=False)),
            ops.FusedLeakyReLU(out_ch),
        )


class Discriminator(nn.Sequential):
    def __init__(self, in_ch, ch_base=64, ch_max=512, shape=(64, 256)):
        shape_out = (shape[0] >> 4, shape[1] >> 4)
        ch = lambda i: min(ch_base << i, ch_max)
        super().__init__(
            BlurVH(),
            Down(in_ch * 2, ch(0)),
            Down(ch(0), ch(1)),
            Down(ch(1), ch(2)),
            Down(ch(2), ch(3)),
            ops.EqualLR(nn.Conv2d(ch(3), 1, shape_out, 1, 0)),
        )


if __name__ == "__main__":
    f = Discriminator(1)
    x = torch.randn(5, 1, 64, 256)
    y = f(x)
    print(x.shape)
    print(y.shape)

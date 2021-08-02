import torch
import torch.nn.functional as F
from torch import nn


class GumbelSigmoid(nn.Module):
    """
    Binary-case of Gumbel-Softmax
    https://arxiv.org/pdf/1611.00712.pdf
    """

    def __init__(
        self,
        tau: float = 1.0,
        tau_max: float = 1.0,  # valid if tau is None
        hard: bool = True,
        eps: float = 1e-10,
        pixelwise: bool = True,
    ):
        super().__init__()
        self.tau = tau
        self.tau_max = tau_max
        self.hard = hard
        self.eps = eps
        if self.tau is None:
            self.weight = nn.Parameter(torch.tensor(0.0))
        self.pixelwise = pixelwise
        self.fixed_noise = None

    def logistic_noise(self, logits):
        B, _, H, W = logits.shape
        shape = (B, 1, H, W) if self.pixelwise else (B, 1, 1, 1)
        U1 = torch.rand(*shape, device=logits.device)
        U2 = torch.rand_like(U1)
        l = -torch.log(torch.log(U1 + self.eps) / torch.log(U2 + self.eps) + self.eps)
        return l

    def sigmoid_with_temperature(self, logits):
        if self.tau is None:
            inverse_tau = F.softplus(self.weight) + 1.0 / self.tau_max
            return torch.sigmoid(logits * inverse_tau)
        else:
            return torch.sigmoid(logits / self.tau)

    def forward(self, logits, threshold: float = 0.5):
        if self.fixed_noise is None:
            logits = logits + self.logistic_noise(logits)
        else:
            B, C, H, W = logits.shape
            logits = logits + self.fixed_noise.expand(B, -1, -1, -1)

        mask_soft = self.sigmoid_with_temperature(logits)

        if self.hard:
            # 'mask_hard' outputs and 'mask_soft' gradients
            mask_hard = (mask_soft > threshold).float()
            return mask_hard - mask_soft.detach() + mask_soft
        else:
            return mask_soft

    def extra_repr(self):
        return f"hard={self.hard}, eps={self.eps}"


class DUSty1(nn.Module):
    def __init__(self, backbone, tau, drop_const=-1):
        super().__init__()
        self.backbone = backbone
        self.gumbel = GumbelSigmoid(hard=True, tau=tau, pixelwise=True)
        self.register_buffer("drop_const", torch.tensor(drop_const).float())

    def forward(self, latent, **kwargs):
        output = self.backbone(latent, **kwargs)
        output = self.maskout(output)
        return output

    def maskout(self, output, threshold=0.5):
        assert isinstance(output, dict)
        assert "confidence" in output
        assert "depth" in output

        depth = output["depth"]
        mask_logit = output["confidence"]

        mask = self.gumbel(mask_logit, threshold)

        output["depth_orig"] = depth
        output["mask"] = mask
        output["depth"] = mask * depth + (1 - mask) * self.drop_const

        return output


class DUSty2(nn.Module):
    def __init__(self, backbone, tau, drop_const=-1):
        super().__init__()
        self.backbone = backbone
        self.gumbel_pixel = GumbelSigmoid(hard=True, tau=tau, pixelwise=True)
        self.gumbel_image = GumbelSigmoid(hard=True, tau=tau, pixelwise=False)
        self.register_buffer("drop_const", torch.tensor(drop_const).float())

    def forward(self, latent, **kwargs):
        output = self.backbone(latent, **kwargs)
        output = self.maskout(output)
        return output

    def maskout(self, output, threshold=0.5):

        assert isinstance(output, dict)
        assert "confidence" in output
        assert "depth" in output

        depth = output["depth"]
        mask_logit = output["confidence"]

        mask_pixel = self.gumbel_pixel(mask_logit[:, [0]], threshold)
        if self.training:
            mask_image = self.gumbel_image(mask_logit[:, [1]], threshold)
        else:
            mask_image = (mask_logit[:, [1]] > 0.0).float()
        mask = mask_pixel * mask_image

        output["depth_orig"] = depth
        output["mask"] = torch.cat([mask_pixel, mask_image], dim=1)
        output["depth"] = mask * depth + (1 - mask) * self.drop_const

        return output

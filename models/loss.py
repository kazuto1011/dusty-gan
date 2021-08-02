import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mean(tensor: torch.Tensor):
    return tensor.mean(0, keepdim=True)


def average_diff(tensor1: torch.Tensor, tensor2: torch.Tensor):
    if isinstance(tensor1, list):
        tensor = []
        for t1, t2 in zip(tensor1, tensor2):
            tensor.append(t1 - mean(t2))
    else:
        tensor = tensor1 - mean(tensor2)
    return tensor


class GANLoss(nn.Module):
    def __init__(self, metric: str, smoothing: float = 1.0):
        super().__init__()
        self.register_buffer("label_real", torch.tensor(1.0))
        self.register_buffer("label_fake", torch.tensor(0.0))
        self.metric = metric
        self.smoothing = smoothing

    def forward(self, pred_real, pred_fake, mode):
        if mode == "G":
            return self.loss_G(pred_real, pred_fake)
        elif mode == "D":
            return self.loss_D(pred_real, pred_fake)
        else:
            raise ValueError

    def loss_D(self, pred_real, pred_fake):
        loss = 0
        if self.metric == "nsgan":
            loss += F.softplus(-pred_real).mean()
            loss += F.softplus(pred_fake).mean()
        elif self.metric == "wgan":
            loss += -pred_real.mean()
            loss += pred_fake.mean()
        elif self.metric == "lsgan":
            target_real = self.label_real.expand_as(pred_real) * self.smoothing
            target_fake = self.label_fake.expand_as(pred_fake)
            loss += F.mse_loss(pred_real, target_real)
            loss += F.mse_loss(pred_fake, target_fake)
        elif self.metric == "hinge":
            loss += F.relu(1 - pred_real).mean()
            loss += F.relu(1 + pred_fake).mean()
        elif self.metric == "ragan":
            loss += F.softplus(-1 * average_diff(pred_real, pred_fake)).mean()
            loss += F.softplus(average_diff(pred_fake, pred_real)).mean()
        elif self.metric == "rahinge":
            loss += F.relu(1 - average_diff(pred_real, pred_fake)).mean()
            loss += F.relu(1 + average_diff(pred_fake, pred_real)).mean()
        elif self.metric == "ralsgan":
            loss += torch.mean((average_diff(pred_real, pred_fake) - 1.0) ** 2)
            loss += torch.mean((average_diff(pred_fake, pred_real) + 1.0) ** 2)
        else:
            raise NotImplementedError
        return loss

    def loss_G(self, pred_real, pred_fake):
        loss = 0
        if self.metric == "nsgan":
            loss += F.softplus(-pred_fake).mean()
        elif self.metric == "wgan":
            loss += -pred_fake.mean()
        elif self.metric == "lsgan":
            target_real = self.label_real.expand_as(pred_fake)
            loss += F.mse_loss(pred_fake, target_real)
        elif self.metric == "hinge":
            loss += -pred_fake.mean()
        elif self.metric == "ragan":
            loss += F.softplus(average_diff(pred_real, pred_fake)).mean()
            loss += F.softplus(-1 * average_diff(pred_fake, pred_real)).mean()
        elif self.metric == "rahinge":
            loss += F.relu(1 + average_diff(pred_real, pred_fake)).mean()
            loss += F.relu(1 - average_diff(pred_fake, pred_real)).mean()
        elif self.metric == "ralsgan":
            loss += torch.mean((average_diff(pred_real, pred_fake) + 1.0) ** 2)
            loss += torch.mean((average_diff(pred_fake, pred_real) - 1.0) ** 2)
        else:
            raise NotImplementedError
        return loss


class GradientPenalty(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        if mode not in ["zero", "one"]:
            raise NotImplementedError(f"{mode}")
        self.mode = mode

    def forward(self, outputs, inputs):

        for i in inputs:
            i.requires_grad == True

        outputs = outputs.sum()

        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            create_graph=True,
            only_inputs=True,
        )

        grads = torch.cat([g.flatten(start_dim=1) for g in grads], dim=1)
        if self.mode == "zero":
            penalty = 0.5 * (grads ** 2).sum(dim=[1, 2, 3]).mean()
        elif self.mode == "one":
            penalty = ((grads.norm(p=2, dim=1) - 1) ** 2).mean()

        return penalty


class PathLengthRegularization(nn.Module):
    def __init__(self, decay: float = 0.99):
        super().__init__()
        self.decay = decay
        self.register_buffer("pl_ema", torch.tensor(0.0))

    def forward(self, fake_imgs, styles):
        # assert len(styles.shape) == 3, "Mapped styles: NxBxD"

        randn_imgs = torch.randn_like(fake_imgs)
        randn_imgs /= np.sqrt(np.prod(fake_imgs.shape[2:]))
        outputs = (fake_imgs * randn_imgs).sum()

        (grads,) = torch.autograd.grad(
            outputs=outputs, inputs=styles, create_graph=True, only_inputs=True
        )

        # Compute |J*y|
        pl_lengths = grads.pow(2).sum(dim=-1)  # NxB or B
        pl_lengths = torch.sqrt(
            pl_lengths.mean(dim=0) if styles.ndim == 2 else pl_lengths
        )

        # EMA of |J*y|
        pl_ema = self.decay * self.pl_ema + (1.0 - self.decay) * pl_lengths.mean()
        self.pl_ema = pl_ema.detach()

        # Calculate (|J*y|-a)^2
        penalty = (pl_lengths - pl_ema).pow(2).mean()

        return penalty

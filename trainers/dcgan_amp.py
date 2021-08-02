import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import utils
from datasets import define_dataset
from models import define_D, define_G
from models.loss import GANLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from utils import cycle, set_requires_grad, sigmoid_to_tanh, tanh_to_sigmoid
from utils.context_manager import gradient_accumulation
from utils.diff_augment import DiffAugment
from utils.lidar import LiDAR
from utils.metrics.cov_mmd_1nna import compute_cov_mmd_1nna
from utils.metrics.jsd import compute_jsd
from utils.metrics.swd import compute_swd
from utils.sampling.fps import downsample_point_clouds


def parse_params(model, query):
    for name, param in model.named_parameters():
        if query in name:
            yield param


@torch.no_grad()
def ema_inplace(ema_model, new_model, decay):
    ema_params = dict(ema_model.named_parameters())
    new_params = dict(new_model.named_parameters())
    for key in ema_params.keys():
        ema_params[key].copy_(decay * ema_params[key] + (1.0 - decay) * new_params[key])


class Trainer:
    def __init__(self, cfg, local_cfg):
        self.cfg = cfg
        self.local_cfg = local_cfg
        self.device = torch.device(self.local_cfg.gpu)

        # setup models
        self.cfg.model.gen.shape = self.cfg.dataset.shape
        self.cfg.model.dis.shape = self.cfg.dataset.shape
        self.G = define_G(self.cfg)
        self.D = define_D(self.cfg)
        self.G_ema = define_G(self.cfg)
        self.G_ema.eval()
        ema_inplace(self.G_ema, self.G, 0.0)
        self.A = DiffAugment(policy=self.cfg.solver.augment)
        self.lidar = LiDAR(
            num_ring=cfg.dataset.shape[0],
            num_points=cfg.dataset.shape[1],
            min_depth=cfg.dataset.min_depth,
            max_depth=cfg.dataset.max_depth,
            angle_file=osp.join(cfg.dataset.root, "angles.pt"),
        )
        self.lidar.eval()

        self.G.to(self.device)
        self.D.to(self.device)
        self.G_ema.to(self.device)
        self.A.to(self.device)
        self.lidar.to(self.device)

        self.G = DDP(self.G, device_ids=[self.local_cfg.gpu], broadcast_buffers=False)
        self.D = DDP(self.D, device_ids=[self.local_cfg.gpu], broadcast_buffers=False)

        if dist.get_rank() == 0:
            print("minibatch size per gpu:", self.local_cfg.batch_size)
            print("number of gradient accumulation:", self.cfg.solver.num_accumulation)

        self.ema_decay = 0.5 ** (
            self.cfg.solver.batch_size / (self.cfg.solver.smoothing_kimg * 1000)
        )

        # training dataset
        self.dataset = define_dataset(self.cfg.dataset, phase="train")
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.local_cfg.batch_size,
            shuffle=False,
            num_workers=self.local_cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            sampler=torch.utils.data.distributed.DistributedSampler(self.dataset),
            drop_last=True,
        )
        self.loader = cycle(self.loader)

        # validation dataset
        self.val_dataset = define_dataset(self.cfg.dataset, phase="val")
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.local_cfg.batch_size,
            shuffle=True,
            num_workers=self.local_cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=False,
        )

        # loss criterion
        self.loss_weight = dict(self.cfg.solver.loss)
        self.criterion = {}
        self.criterion["gan"] = GANLoss(self.cfg.solver.gan_mode).to(self.device)
        if "gp" in self.loss_weight and self.loss_weight["gp"] > 0.0:
            self.criterion["gp"] = True
        if "pl" in self.loss_weight and self.loss_weight["pl"] > 0.0:
            self.criterion["pl"] = True
            self.pl_ema = torch.tensor(0.0).to(self.device)
        if dist.get_rank() == 0:
            print("loss: {}".format(tuple(self.criterion.keys())))

        # optimizer
        self.optim_G = optim.Adam(
            params=self.G.parameters(),
            lr=self.cfg.solver.lr.alpha.gen,
            betas=(self.cfg.solver.lr.beta1, self.cfg.solver.lr.beta2),
        )
        self.optim_D = optim.Adam(
            params=self.D.parameters(),
            lr=self.cfg.solver.lr.alpha.dis,
            betas=(self.cfg.solver.lr.beta1, self.cfg.solver.lr.beta2),
        )

        # automatic mixed precision
        self.enable_amp = cfg.enable_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)
        if dist.get_rank() == 0 and self.enable_amp:
            print("amp enabled")

        # resume from checkpoints
        self.start_iteration = 0
        if self.cfg.resume is not None:
            state_dict = torch.load(self.cfg.resume, map_location="cpu")
            self.start_iteration = state_dict["step"] // self.cfg.solver.batch_size
            self.G.module.load_state_dict(state_dict["G"])
            self.D.module.load_state_dict(state_dict["D"])
            self.G_ema.load_state_dict(state_dict["G_ema"])
            self.optim_G.load_state_dict(state_dict["optim_G"])
            self.optim_D.load_state_dict(state_dict["optim_D"])
            if "pl" in self.criterion:
                self.criterion["pl"].pl_ema = state_dict["pl_ema"].to(self.device)

        # for visual validation
        self.fixed_noise = torch.randn(
            self.local_cfg.batch_size, cfg.model.gen.in_ch, device=self.device
        )

    def sample_latents(self, B):
        return torch.randn(B, self.cfg.model.gen.in_ch, device=self.device)

    def fetch_reals(self, raw_batch):
        pol = raw_batch["depth"].to(self.device)  # [0,1]
        mask = raw_batch["mask"].to(self.device).float()
        inv = self.lidar.invert_depth(pol)
        inv = sigmoid_to_tanh(inv)  # [-1,1]
        inv = mask * inv + (1 - mask) * self.cfg.model.gen.drop_const
        return inv, mask

    def step(self, i):

        self.G.train()

        scalars = defaultdict(list)
        xs_real = []
        xs_fake = []
        zs = []

        #############################################################
        # train D
        #############################################################

        set_requires_grad(self.D, True)

        self.optim_D.zero_grad(set_to_none=True)
        for j in gradient_accumulation(
            self.cfg.solver.num_accumulation, True, (self.G, self.D)
        ):

            # input data
            x_real, m_real = self.fetch_reals(next(self.loader))
            xs_real.append({"depth": x_real, "mask": m_real})
            B = x_real.shape[0]

            # sample z
            z = self.sample_latents(B)
            zs.append(z)

            loss_D = 0

            # discriminator loss
            with torch.cuda.amp.autocast(enabled=self.enable_amp):
                synth = self.G(latent=z)
                xs_fake.append(synth)

                # augment
                x_real_aug = self.A(x_real).detach().requires_grad_()
                x_fake_aug = self.A(synth["depth"]).detach()

                # forward D
                y_real = self.D(x_real_aug)
                y_fake = self.D(x_fake_aug)

                scalars["loss/D/output/real"].append(y_real.mean().detach())
                scalars["loss/D/output/fake"].append(y_fake.mean().detach())

                # adversarial loss
                loss_GAN = self.criterion["gan"](y_real, y_fake, "D")
                loss_D += self.loss_weight["gan"] * loss_GAN

                scalars["loss/D/adversarial"].append(loss_GAN.detach())

            # r1 gradient penalty
            if "gp" in self.criterion:

                (grads,) = torch.autograd.grad(
                    outputs=self.scaler.scale(y_real.sum()),
                    inputs=[x_real_aug],
                    create_graph=True,
                    only_inputs=True,
                )

                # unscale
                grads = grads / self.scaler.get_scale()

                with torch.cuda.amp.autocast(enabled=self.enable_amp):
                    r1_penalty = (grads ** 2).sum(dim=[1, 2, 3]).mean()
                    scalars["loss/D/gradient_penalty"].append(r1_penalty.detach())
                    loss_D += (self.loss_weight["gp"] / 2) * r1_penalty
                    loss_D += 0.0 * y_real.squeeze()[0]

            loss_D /= float(self.cfg.solver.num_accumulation)
            self.scaler.scale(loss_D).backward()

        # update D parameters
        self.scaler.step(self.optim_D)

        #############################################################
        # train G
        #############################################################

        set_requires_grad(self.D, False)

        self.optim_G.zero_grad(set_to_none=True)
        for j in gradient_accumulation(
            self.cfg.solver.num_accumulation, True, (self.G, self.D)
        ):
            loss_G = 0

            # generator loss
            with torch.cuda.amp.autocast(enabled=self.enable_amp):
                # augment
                x_real_aug = self.A(xs_real[j]["depth"]).detach()
                x_fake_aug = self.A(xs_fake[j]["depth"])

                # forward D
                y_real = self.D(x_real_aug)
                y_fake = self.D(x_fake_aug)

                # adversarial loss
                loss_GAN = self.criterion["gan"](y_real, y_fake, "G")
                loss_G += self.loss_weight["gan"] * loss_GAN

                scalars["loss/G/adversarial"].append(loss_GAN.detach())

            # path length regularization
            if "pl" in self.criterion:
                # forward G with smaller batch
                B_pl = len(xs_real[j]["depth"]) // 2
                z_pl = self.sample_latents(B_pl).requires_grad_()

                # perturb images
                with torch.cuda.amp.autocast(enabled=self.enable_amp):
                    synth_pl = self.G(latent=z_pl)
                    x_pl = synth_pl["depth"]
                    noise_pl = torch.randn_like(x_pl)
                    noise_pl /= np.sqrt(np.prod(x_pl.shape[2:]))
                    outputs = (x_pl * noise_pl).sum()

                (grads,) = torch.autograd.grad(
                    outputs=self.scaler.scale(outputs),
                    inputs=[z_pl],
                    create_graph=True,
                    only_inputs=True,
                )

                # unscale
                grads = grads / self.scaler.get_scale()

                with torch.cuda.amp.autocast(enabled=self.enable_amp):
                    # compute |J*y|
                    pl_lengths = grads.pow(2).sum(dim=-1)
                    pl_lengths = torch.sqrt(pl_lengths)
                    # ema of |J*y|
                    pl_ema = self.pl_ema.lerp(pl_lengths.mean(), 0.01)
                    self.pl_ema.copy_(pl_ema.detach())
                    # calculate (|J*y|-a)^2
                    pl_penalty = (pl_lengths - pl_ema).pow(2).mean()

                    scalars["loss/G/path_length/baseline"].append(self.pl_ema.detach())
                    scalars["loss/G/path_length"].append(pl_penalty.detach())

                    loss_G += self.loss_weight["pl"] * pl_penalty
                    loss_G += 0.0 * x_pl[0, 0, 0, 0]

            loss_G /= float(self.cfg.solver.num_accumulation)
            self.scaler.scale(loss_G).backward()

        # update G parameters
        self.scaler.step(self.optim_G)

        self.scaler.update()

        ema_inplace(self.G_ema, self.G.module, self.ema_decay)

        # gather scalars from all devices
        for key, scalar_list in scalars.items():
            scalar = torch.mean(torch.stack(scalar_list))
            dist.all_reduce(scalar)  # sum over gpus
            scalar /= dist.get_world_size()
            scalars[key] = scalar.item()

        return scalars

    def postprocess(self, synth):
        synth = utils.postprocess(synth, self.lidar)
        return synth

    @torch.no_grad()
    def generate(self, ema=False):
        if ema:
            self.G_ema.eval()
            synth = self.G_ema(self.fixed_noise)
        else:
            self.G.eval()
            synth = self.G(self.fixed_noise)
        synth = self.postprocess(synth)
        return synth

    @torch.no_grad()
    def validation(self):
        def inv_to_xyz(inv, tol=1e-8):
            inv = tanh_to_sigmoid(inv).clamp_(0, 1)
            xyz = self.lidar.inv_to_xyz(inv, tol)
            xyz = xyz.flatten(2).transpose(1, 2)  # (B,N,3)
            xyz = downsample_point_clouds(xyz, self.cfg.solver.validation.num_points)
            return xyz

        self.G_ema.eval()

        data = defaultdict(list)
        N = len(self.val_dataset)

        # real data
        for item in tqdm(
            self.val_loader,
            desc="real data",
            dynamic_ncols=True,
            disable=not dist.get_rank() == 0,
            leave=False,
        ):
            x_real, m_real = self.fetch_reals(item)
            data["real-2d"].append(x_real)
            points = inv_to_xyz(x_real)
            data["real-3d"].append(points)

        # fake data
        for _ in tqdm(
            range(0, N, self.local_cfg.batch_size),
            desc="synthetic data",
            dynamic_ncols=True,
            disable=not dist.get_rank() == 0,
            leave=False,
        ):
            latent = self.sample_latents(self.local_cfg.batch_size)
            x_fake = self.G_ema(latent=latent)["depth"]
            data["fake-2d"].append(x_fake)
            points = inv_to_xyz(x_fake)
            data["fake-3d"].append(points)

        for key in data.keys():
            data[key] = torch.cat(data[key], dim=0)[:N]

        scores = {}
        scores.update(compute_swd(data["fake-2d"], data["real-2d"]))
        scores["jsd"] = compute_jsd(data["fake-3d"] / 2.0, data["real-3d"] / 2.0)
        scores.update(
            compute_cov_mmd_1nna(data["fake-3d"], data["real-3d"], 512, ("cd",))
        )

        return scores

    def save_models(self, suffix, step):
        torch.save(
            {
                "step": step,
                "G": self.G.module.state_dict(),
                "D": self.D.module.state_dict(),
                "G_ema": self.G_ema.state_dict(),
                "optim_G": self.optim_G.state_dict(),
                "optim_D": self.optim_D.state_dict(),
                "pl_ema": self.pl_ema.detach().cpu()
                if "pl" in self.criterion
                else None,
            },
            osp.join("models", "checkpoint_{}.pth".format(str(suffix))),
        )

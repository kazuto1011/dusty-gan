import os.path as osp
import re
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.optim as optim
from datasets import define_dataset
from metrics.jsd import compute_jsd
from metrics.point_clouds import compute_cov_mmd_1nna
from metrics.swd import compute_swd
from models import define_D, define_G
from models.loss import GANLoss, GradientPenalty, PathLengthRegularization
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from utils import cycle, denorm_range, norm_range, set_requires_grad, zero_grad
from utils.context_manager import gradient_accumulation
from utils.diff_augment import DiffAugment
from utils.lidar import LiDAR


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
        self.G = define_G(self.cfg.model.gen)
        self.D = define_D(self.cfg.model.dis)
        self.G_ema = define_G(self.cfg.model.gen)
        self.G_ema.eval()
        ema_inplace(self.G_ema, self.G, 0.0)
        self.lidar = LiDAR(
            num_ring=cfg.dataset.shape[0],
            num_points=cfg.dataset.shape[1],
            min_depth=cfg.dataset.min_depth,
            max_depth=cfg.dataset.max_depth,
            angle_file=osp.join(cfg.dataset.root, "angles.pt"),
        )
        self.lidar.eval()
        self.A = DiffAugment(policy=self.cfg.solver.augment)

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

        # dataset
        self.dataset = define_dataset(self.cfg.dataset, phase="train")
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.local_cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            sampler=torch.utils.data.distributed.DistributedSampler(self.dataset),
            drop_last=True,
        )
        self.loader = cycle(self.loader)

        self.val_dataset = define_dataset(self.cfg.dataset, phase="val")
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.local_cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # Loss criterion
        lazy_ratio_G = 1.0
        lazy_ratio_D = 1.0
        self.loss_weight = dict(self.cfg.solver.loss)
        self.criterion = {}
        self.criterion["gan"] = GANLoss(
            self.cfg.solver.gan_mode, self.cfg.solver.label.smoothing
        )
        if "gp" in self.loss_weight and self.loss_weight["gp"] > 0.0:
            gp_mode = "one" if self.cfg.solver.gan_mode == "wgan" else "zero"
            self.criterion["gp"] = GradientPenalty(gp_mode)
            lazy_ratio_D = self.cfg.solver.lazy.gp / (self.cfg.solver.lazy.gp + 1.0)
        if "pl" in self.loss_weight and self.loss_weight["pl"] > 0.0:
            self.criterion["pl"] = PathLengthRegularization()
            lazy_ratio_G = self.cfg.solver.lazy.pl / (self.cfg.solver.lazy.pl + 1.0)
        for criteria in self.criterion.values():
            criteria.to(self.device)
        if dist.get_rank() == 0:
            print("loss: {}".format(tuple(self.criterion.keys())))

        # Lazy regularization
        lr_G = self.cfg.solver.lr.alpha.gen * lazy_ratio_G
        lr_D = self.cfg.solver.lr.alpha.dis * lazy_ratio_D
        beta1_G = self.cfg.solver.lr.beta1 ** lazy_ratio_G
        beta2_G = self.cfg.solver.lr.beta2 ** lazy_ratio_G
        beta1_D = self.cfg.solver.lr.beta1 ** lazy_ratio_D
        beta2_D = self.cfg.solver.lr.beta2 ** lazy_ratio_D

        # Optimizer
        self.optim_G = optim.Adam(
            params=self.G.parameters(),
            lr=lr_G,
            betas=(beta1_G, beta2_G),
        )
        self.optim_D = optim.Adam(
            params=self.D.parameters(),
            lr=lr_D,
            betas=(beta1_D, beta2_D),
        )

        # Resume from checkpoints
        self.start_iteration = 0
        if self.cfg.resume is not None:
            state_dict = torch.load(self.cfg.resume, map_location="cpu")
            checkpoint_file = self.cfg.resume.split("/")[-1]
            print("Resume from {}".format(checkpoint_file))
            self.start_iteration = int(re.findall("[0-9]+", checkpoint_file)[0])
            self.G.load_state_dict(state_dict["G"])
            self.D.load_state_dict(state_dict["D"])
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
        inv = norm_range(inv)  # [-1,1]
        inv = mask * inv + (1 - mask) * self.cfg.model.gen.drop_const
        return inv, mask

    def step(self, i):

        self.G.train()

        scalars = defaultdict(list)

        #############################################################
        # Setup real data
        #############################################################

        # Input data
        real, _ = self.fetch_reals(next(self.loader))
        fakes = []
        latents = []
        B, _, _, _ = real.shape

        #############################################################
        # Training D
        #############################################################

        set_requires_grad(self.D, True)

        zero_grad(self.optim_D)
        for j in gradient_accumulation(
            self.cfg.solver.num_accumulation, True, (self.G, self.D)
        ):
            # Forward G
            latent = self.sample_latents(B)
            synth = self.G(latent=latent)
            fakes.append(synth)
            latents.append(latent)

            # Forward D
            y_real = self.D(self.A(real).detach())
            y_fake = self.D(self.A(synth["depth"]).detach())

            # Adversarial loss
            loss_GAN = self.criterion["gan"](y_real, y_fake, "D")
            loss_D = self.loss_weight["gan"] * loss_GAN

            scalars["loss/D/output/real"].append(y_real.mean().detach())
            scalars["loss/D/output/fake"].append(y_fake.mean().detach())
            scalars["loss/D/adversarial"].append(loss_GAN.detach())

            loss_D /= float(self.cfg.solver.num_accumulation)
            loss_D.backward()

        self.optim_D.step()

        #############################################################
        # Gradient penalty
        #############################################################

        if "gp" in self.criterion and i % self.cfg.solver.lazy.gp == 0:
            zero_grad(self.optim_D)
            for j in gradient_accumulation(
                self.cfg.solver.num_accumulation, True, (self.G, self.D)
            ):
                # Gradient penalty (skip augment)
                if self.criterion["gp"].mode == "zero":
                    input_gp = real.detach()
                elif self.criterion["gp"].mode == "one":
                    eps = torch.rand(B, 1, 1, 1, device=self.device)
                    real_ = real.detach()
                    fake_ = fakes[j]["depth"].detach()
                    input_gp = eps * real_ + (1 - eps) * fake_
                else:
                    raise NotImplementedError

                input_gp = self.A(input_gp)
                input_gp.requires_grad = True
                pred = self.D(input_gp)

                loss_gp = self.criterion["gp"](pred, input_gp)
                loss_D = self.loss_weight["gp"] * self.cfg.solver.lazy.gp * loss_gp
                loss_D += 0.0 * pred[0].squeeze()

                scalars["loss/D/gradient_penalty"].append(loss_gp.detach())

                loss_D /= float(self.cfg.solver.num_accumulation)
                loss_D.backward()

            self.optim_D.step()

        #############################################################
        # Training G
        #############################################################

        set_requires_grad(self.D, False)

        zero_grad(self.optim_G)
        for j in gradient_accumulation(
            self.cfg.solver.num_accumulation, True, (self.G, self.D)
        ):
            # Forward D
            y_real = self.D(self.A(real).detach())
            y_fake = self.D(self.A(fakes[j]["depth"]))

            # Adversarial loss
            loss_GAN = self.criterion["gan"](y_real, y_fake, "G")
            loss_G = self.loss_weight["gan"] * loss_GAN

            scalars["loss/G/adversarial"].append(loss_GAN.detach())

            # Diversity sensitive regularization
            if "ds" in self.criterion:
                loss_ds = self.criterion["ds"](fakes[j]["mask"], latents[j])
                loss_G += self.loss_weight["ds"] * loss_ds

                scalars["loss/G/noise_diversity"].append(loss_ds.detach())

            loss_G /= float(self.cfg.solver.num_accumulation)
            loss_G.backward()

        self.optim_G.step()

        #############################################################
        # Path length regularization
        #############################################################

        if "pl" in self.criterion and i % self.cfg.solver.lazy.pl == 0:
            zero_grad(self.optim_G)
            for j in gradient_accumulation(
                self.cfg.solver.num_accumulation, True, (self.G, self.D)
            ):
                # Forward G with smaller batch
                latent = self.sample_latents(B // 2)
                latent.requires_grad = True
                synth = self.G(latent=latent)
                fake = synth["depth"]

                loss_pl = self.criterion["pl"](fake, latent)
                loss_G = self.loss_weight["pl"] * self.cfg.solver.lazy.pl * loss_pl
                loss_G += 0.0 * synth["depth"][0, 0, 0, 0]
                scalars["loss/G/path_length"].append(loss_pl.detach())
                scalars["loss/G/path_length/baseline"].append(
                    self.criterion["pl"].pl_ema.detach()
                )

                loss_G /= float(self.cfg.solver.num_accumulation)
                loss_G.backward()

            self.optim_G.step()

        ema_inplace(self.G_ema, self.G.module, self.ema_decay)

        # gather the scalars on all devices
        for key, scalar_list in scalars.items():
            scalar = torch.mean(torch.stack(scalar_list))
            dist.all_reduce(scalar)
            scalar /= dist.get_world_size()
            scalars[key] = scalar.item()

        return scalars

    @torch.no_grad()
    def generate(self, ema=False):
        if ema:
            self.G_ema.eval()
            return self.G_ema(self.fixed_noise)
        else:
            self.G.eval()
            return self.G(self.fixed_noise)

    @torch.no_grad()
    def validation(self):
        def subsample(points, N=-1):
            if N == -1:
                return points
            else:
                return torch.stack(
                    [p[torch.randperm(len(p), device=self.device)[:N]] for p in points],
                    dim=0,
                )

        def inv_to_points(inv, tol=1e-8):
            inv = denorm_range(inv).clamp_(0, 1)
            xyz = self.lidar.inv_depth_to_points(inv, tol)
            xyz = xyz.flatten(2).transpose(1, 2)  # B,N,3
            xyz = subsample(xyz, self.cfg.solver.validation.num_points)
            return xyz

        self.G_ema.eval()

        # real data
        reals = defaultdict(list)
        for data in tqdm(
            self.val_loader,
            desc="validation",
            dynamic_ncols=True,
            disable=not dist.get_rank() == 0,
            leave=False,
        ):
            inv, mask = self.fetch_reals(data)
            reals["2d"].append(inv)
            points = inv_to_points(inv)
            reals["3d"].append(points)
        for key in reals.keys():
            reals[key] = torch.cat(reals[key], dim=0)

        # fake data
        fakes = defaultdict(list)
        for i in tqdm(
            range(0, len(reals["2d"]), self.local_cfg.batch_size),
            desc="Validation",
            dynamic_ncols=True,
            disable=not dist.get_rank() == 0,
            leave=False,
        ):
            latent = self.sample_latents(self.local_cfg.batch_size)
            inv = self.G_ema(latent=latent)["depth"]
            fakes["2d"].append(inv)
            points = inv_to_points(inv)
            fakes["3d"].append(points)
        for key in fakes.keys():
            fakes[key] = torch.cat(fakes[key], dim=0)[: len(reals[key])]

        scores = {}
        scores.update(compute_swd(fakes["2d"], reals["2d"]))
        scores["jsd"] = compute_jsd(fakes["3d"] / 2.0, reals["3d"] / 2.0)
        scores.update(compute_cov_mmd_1nna(fakes["3d"], reals["3d"], 512, ("cd",)))

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
                "pl_ema": self.criterion["pl"].pl_ema.detach().cpu()
                if "pl" in self.criterion
                else None,
            },
            osp.join("models", "checkpoint_{}.pth".format(str(suffix))),
        )

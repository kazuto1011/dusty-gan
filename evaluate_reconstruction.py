import argparse
import datetime
import os
import os.path as osp
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.nn.parallel import DataParallel as DP
from tqdm import tqdm

import utils
from datasets import define_dataset
from utils.metrics.cov_mmd_1nna import compute_cd
from utils.metrics.depth import compute_depth_accuracy, compute_depth_error

if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--save-dir-path", type=str, default=".")
    parser.add_argument("--tol", type=float, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--distance", default="l1", choices=["l1", "l2"])
    args = parser.parse_args()

    cfg, G, lidar, device = utils.setup(
        args.model_path,
        args.config_path,
        ema=True,
        fix_noise=True,
    )

    utils.set_requires_grad(G, False)
    G = DP(G)

    # hyperparameters
    num_step = 1000
    perturb_latent = True
    noise_ratio = 0.75
    noise_sigma = 1.0
    lr_rampup_ratio = 0.05
    lr_rampdown_ratio = 0.25

    # prepare reference
    dataset = define_dataset(cfg.dataset, phase="test")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    # -------------------------------------------------------------------
    # utilities
    # -------------------------------------------------------------------
    def preprocess_reals(raw_batch):
        xyz = raw_batch["xyz"].to(device)
        depth = raw_batch["depth"].to(device)
        mask = raw_batch["mask"].to(device).float()
        inv = lidar.invert_depth(depth)
        inv = mask * inv + (1 - mask) * 0.0
        return inv, mask, xyz

    # stylegan2's schedule
    def lr_schedule(iteration):
        t = iteration / num_step
        gamma = min(1.0, (1.0 - t) / lr_rampdown_ratio)
        gamma = 0.5 - 0.5 * np.cos(gamma * np.pi)
        gamma = gamma * min(1.0, t / lr_rampup_ratio)
        return gamma

    # -------------------------------------------------------------------
    # run inversion
    # -------------------------------------------------------------------

    n = 0
    results = defaultdict(list)
    for i, item in enumerate(tqdm(loader)):
        inv_ref, mask_ref, xyz_ref = preprocess_reals(item)
        batch_size_i = len(inv_ref)

        # trainable latent code
        latent = torch.randn(batch_size_i, cfg.model.gen.in_ch, device=device)
        latent.div_(latent.pow(2).mean(dim=1, keepdim=True).add(1e-9).sqrt())
        latent = torch.nn.Parameter(latent).requires_grad_()

        optim = utils.SphericalOptimizer(params=[latent], lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_schedule)

        # optimize the latent
        for current_step in tqdm(range(num_step), leave=False):
            progress = current_step / num_step

            # noise
            w = max(0.0, 1.0 - progress / noise_ratio)
            noise_strength = 0.05 * noise_sigma * w ** 2
            noise = noise_strength * torch.randn_like(latent)

            # forward G
            out = G(latent + noise if perturb_latent else latent)

            if "dusty" in cfg.model.gen.arch:
                inv_gen = utils.tanh_to_sigmoid(out["depth_orig"])
            else:
                inv_gen = utils.tanh_to_sigmoid(out["depth"])

            # loss
            loss = utils.masked_loss(inv_ref, inv_gen, mask_ref, args.distance)

            # per-sample gradients
            optim.zero_grad()
            loss.backward(gradient=torch.ones_like(loss))
            optim.step()
            scheduler.step()

        # post-processing
        out = utils.postprocess(out, lidar, tol=args.tol)
        points_gen = utils.flatten(out["points"])
        points_ref = utils.flatten(xyz_ref)
        depth_gen = lidar.revert_depth(inv_gen, norm=False)
        depth_ref = lidar.revert_depth(inv_ref, norm=False)

        # evaluation
        cd = compute_cd(points_ref, points_gen)
        results["cd"] += cd.tolist()
        accuracies = compute_depth_accuracy(depth_ref, depth_gen, mask_ref)
        results["accuracy_1"] += accuracies["accuracy_1"].tolist()
        results["accuracy_2"] += accuracies["accuracy_2"].tolist()
        results["accuracy_3"] += accuracies["accuracy_3"].tolist()
        errors = compute_depth_error(depth_ref, depth_gen, mask_ref)
        results["rmse"] += errors["rmse"].tolist()
        results["rmse_log"] += errors["rmse_log"].tolist()
        results["abs_rel"] += errors["abs_rel"].tolist()
        results["sq_rel"] += errors["sq_rel"].tolist()
        results["tol"] += [args.tol] * batch_size_i

        _, _, H, W = out["depth"].shape
        if "dusty" in cfg.model.gen.arch:
            total_drop = (1 - out["mask"]).sum(dim=[1, 2, 3]) / (H * W)
            results["drop_gen"] += total_drop.tolist()
        else:
            mask = (torch.abs(out["depth"] - 0.0) > args.tol).float()
            total_drop = (1 - mask).sum(dim=[1, 2, 3]) / (H * W)
            results["drop_gen"] += total_drop.tolist()

        _, _, H, W = mask_ref.shape
        total_drop = (1 - mask_ref).sum(dim=[1, 2, 3]) / (H * W)
        results["drop_ref"] += total_drop.tolist()

        n += batch_size_i

    # save results
    os.makedirs(args.save_dir_path, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat()
    save_path = osp.join(args.save_dir_path, f"{timestamp}.csv")
    df = pd.DataFrame(results)
    df.to_csv(save_path)
    print(f"Saved: {save_path}")

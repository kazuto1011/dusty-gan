import argparse
import datetime
import json
import os
import os.path as osp
import pprint
from collections import defaultdict

import torch
from tqdm import tqdm

import utils
from datasets import define_dataset
from utils import sigmoid_to_tanh, tanh_to_sigmoid
from utils.metrics.cov_mmd_1nna import compute_cov_mmd_1nna
from utils.metrics.jsd import compute_jsd
from utils.metrics.swd import compute_swd
from utils.sampling.fps import downsample_point_clouds

if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--save-dir-path", type=str, default=".")
    parser.add_argument("--num-test", type=int, default=5000)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--tol", type=float, default=0)
    parser.add_argument("--compute-gt", action="store_true")
    args = parser.parse_args()

    cfg, G, lidar, device = utils.setup(
        args.model_path,
        args.config_path,
        ema=True,
        fix_noise=True,
    )

    # -------------------------------------------------------------------
    # utilities
    # -------------------------------------------------------------------
    def sample_latents(B):
        return torch.randn(B, cfg.model.gen.in_ch, device=device)

    def preprocess_reals(raw_batch):
        xyz = raw_batch["xyz"].to(device)
        points = xyz.flatten(2).transpose(1, 2)  # B,N,3
        depth = raw_batch["depth"].to(device)  # [0,1]
        mask = raw_batch["mask"].to(device).float()
        inv = lidar.invert_depth(depth)
        inv = sigmoid_to_tanh(inv)  # [-1,1]
        inv = mask * inv + (1 - mask) * cfg.model.gen.drop_const
        return inv, mask, points

    def project_2d_to_3d(inv, tol=1e-8):
        inv = tanh_to_sigmoid(inv).clamp_(0, 1)
        xyz = lidar.inv_to_xyz(inv, tol)
        points = xyz.flatten(2).transpose(1, 2)  # (B,N,3)
        points = downsample_point_clouds(points, args.num_points)
        return points

    # -------------------------------------------------------------------
    # real data
    # -------------------------------------------------------------------
    reals = {}
    for subset in ("train", "test"):
        cache_path = f"data/cache_{cfg.dataset.name}_{subset}_{args.num_points}.pt"
        if osp.exists(cache_path):
            reals[subset] = torch.load(cache_path, map_location="cpu")
            print("loaded:", cache_path)
        else:
            loader = torch.utils.data.DataLoader(
                define_dataset(cfg.dataset, phase=subset),
                batch_size=cfg.solver.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            reals[subset] = defaultdict(list)
            for data in tqdm(
                loader,
                desc=f"real data ({subset})",
                dynamic_ncols=True,
                leave=False,
            ):
                inv, mask, points = preprocess_reals(data)
                reals[subset]["2d"].append(inv)
                points = downsample_point_clouds(points, k=args.num_points)
                reals[subset]["3d"].append(points)
            for key in reals[subset].keys():
                reals[subset][key] = torch.cat(reals[subset][key], dim=0)
            torch.save(reals[subset], cache_path)
            print("cached:", cache_path)

    # -------------------------------------------------------------------
    # subsampling time series
    # -------------------------------------------------------------------
    for mode in ("2d", "3d"):
        for subset in ("train", "test"):
            if args.num_test != -1:
                skip = len(reals[subset][mode]) // args.num_test
                limit = skip * args.num_test + 1
                reals[subset][mode] = reals[subset][mode][skip:limit:skip].to(device)
            else:
                reals[subset][mode] = reals[subset][mode].to(device)
            print("real", subset, mode, tuple(reals[subset][mode].shape))

    # -------------------------------------------------------------------
    # scores of training set
    # -------------------------------------------------------------------
    if args.compute_gt:
        print("training set only")
        scores = {}
        scores_swd = compute_swd(
            image1=reals["train"]["2d"],
            image2=reals["test"]["2d"],
        )
        scores.update(scores_swd)
        scores["jsd"] = compute_jsd(
            pcs_gen=reals["train"]["3d"] / 2.0,
            pcs_ref=reals["test"]["3d"] / 2.0,
        )
        score_3d = compute_cov_mmd_1nna(
            pcs_gen=reals["train"]["3d"],
            pcs_ref=reals["test"]["3d"],
            batch_size=512,
            metrics=("cd",),
        )
        scores.update(score_3d)
        scores["#test"] = args.num_test
        scores["#points"] = args.num_points
        pprint.pprint(scores)

        gt_dir = f"outputs/logs/dataset={cfg.dataset.name}/gt/evaluation/tol=0"
        os.makedirs(gt_dir, exist_ok=True)
        timestamp = datetime.datetime.now().isoformat()
        save_path = osp.join(gt_dir, "{}.json".format(timestamp))
        with open(save_path, "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4, sort_keys=True)
        quit()

    # -------------------------------------------------------------------
    # synthetic data
    # -------------------------------------------------------------------
    N_test = len(reals["test"]["2d"])
    fakes = defaultdict(list)
    for i in tqdm(
        range(0, N_test, cfg.solver.batch_size),
        desc="synthetic data",
        dynamic_ncols=True,
        leave=False,
    ):
        latent = sample_latents(cfg.solver.batch_size)
        inv = G(latent=latent)["depth"]
        fakes["2d"].append(inv)
        points = project_2d_to_3d(inv, tol=args.tol)
        fakes["3d"].append(points)
    for key in fakes.keys():
        fakes[key] = torch.cat(fakes[key], dim=0)[:N_test]

    # -------------------------------------------------------------------
    # evaluation
    # -------------------------------------------------------------------
    scores = {}
    scores_swd = compute_swd(
        image1=fakes["2d"],
        image2=reals["test"]["2d"],
    )
    scores.update(scores_swd)
    scores["jsd"] = compute_jsd(
        pcs_gen=fakes["3d"] / 2.0,
        pcs_ref=reals["test"]["3d"] / 2.0,
    )
    scores_3d = compute_cov_mmd_1nna(
        pcs_gen=fakes["3d"],
        pcs_ref=reals["test"]["3d"],
        batch_size=512,
        metrics=("cd",),
    )
    scores.update(scores_3d)
    scores["#test"] = args.num_test
    scores["#points"] = args.num_points
    pprint.pprint(scores)

    # save results
    os.makedirs(args.save_dir_path, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat()
    save_path = osp.join(args.save_dir_path, f"{timestamp}.csv")
    with open(save_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4, sort_keys=True)
    print(f"Saved: {save_path}")

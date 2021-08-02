import argparse
import json
import os.path as osp
from collections import defaultdict

import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from tqdm import tqdm

import utils
from datasets import define_dataset
from utils import sigmoid_to_tanh, tanh_to_sigmoid
from utils.metrics.cov_mmd_1nna import compute_cov_mmd_1nna
from utils.metrics.jsd import compute_jsd
from utils.sampling.fps import downsample_point_clouds


def evaluation(config, fakes, lidar):
    def project_2d_to_3d(inv, tol=1e-8):
        inv = tanh_to_sigmoid(inv).clamp_(0, 1)
        xyz = lidar.inv_to_xyz(inv, tol)
        xyz = xyz.flatten(2).transpose(1, 2)  # (B,N,3)
        xyz = downsample_point_clouds(xyz, config["num_points"])
        return xyz

    fakes_3d = []
    for i in range(0, N_val, config["batch_size"]):
        fakes_3d.append(
            project_2d_to_3d(
                fakes["2d"][i : i + config["batch_size"]],
                tol=config["tol"],
            )
        )
    fakes_3d = torch.cat(fakes_3d, dim=0)

    scores = compute_cov_mmd_1nna(
        pcs_gen=fakes_3d,
        pcs_ref=reals["val"]["3d"],
        batch_size=512,
        metrics=("cd",),
        verbose=False,
    )
    scores["jsd"] = compute_jsd(
        pcs_gen=fakes_3d / 2.0,
        pcs_ref=reals["val"]["3d"] / 2.0,
    )
    scores["#points"] = config["num_points"]

    scores["weighted"] = (
        1.0 * scores["1-nn-accuracy-cd"]
        + 100 * scores["mmd-cd"]
        + -1.0 * scores["cov-cd"]
        + 10 * scores["jsd"]
    )

    tune.report(**scores)


if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-test", type=int, default=-1)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--tol", type=float, default=0)
    args = parser.parse_args()

    cfg, G, lidar, device, kwargs = utils.setup(
        args.model_path, ema=True, fix_noise=True
    )

    # -------------------------------------------------------------------
    # utilities
    # -------------------------------------------------------------------
    def sample_latents(B):
        return torch.randn(B, cfg.model.gen.in_ch, device=device)

    def fetch_reals(raw_batch):
        xyz = raw_batch["xyz"].to(device)
        xyz = xyz.flatten(2).transpose(1, 2)  # B,N,3
        pol = raw_batch["depth"].to(device)  # [0,1]
        mask = raw_batch["mask"].to(device).float()
        inv = lidar.invert_depth(pol)
        inv = sigmoid_to_tanh(inv)  # [-1,1]
        inv = mask * inv + (1 - mask) * cfg.model.gen.drop_const
        return inv, mask, xyz

    # -------------------------------------------------------------------
    # real data
    # -------------------------------------------------------------------
    reals = {}
    for subset in ("val",):
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
                inv, mask, points = fetch_reals(data)
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
        for subset in ("val",):
            if args.num_test != -1:
                skip = len(reals[subset][mode]) // args.num_test
                limit = skip * args.num_test + 1
                reals[subset][mode] = reals[subset][mode][skip:limit:skip].to(device)
            else:
                reals[subset][mode] = reals[subset][mode].to(device)
            print("real", subset, mode, tuple(reals[subset][mode].shape))

    torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # synthetic data
    # -------------------------------------------------------------------
    N_val = len(reals["val"]["2d"])
    fakes = defaultdict(list)
    for i in tqdm(
        range(0, N_val, cfg.solver.batch_size),
        desc="synthetic data",
        dynamic_ncols=True,
        leave=False,
    ):
        latent = sample_latents(cfg.solver.batch_size)
        inv = G(latent=latent)["depth"]
        fakes["2d"].append(inv)
    fakes["2d"] = torch.cat(fakes["2d"], dim=0)[:N_val]

    torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # evaluation
    # -------------------------------------------------------------------

    objective = "weighted"
    search_alg = HyperOptSearch(metric=objective, mode="min")
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=4)
    reporter = CLIReporter(
        metric_columns=["1-nn-accuracy-cd", "mmd-cd", "cov-cd", "jsd", "weighted"]
    )

    analysis = tune.run(
        tune.with_parameters(evaluation, fakes=fakes, lidar=lidar),
        config={
            "tol": tune.qloguniform(1e-3, 1e-1, 5e-4),
            "num_points": args.num_points,
            "batch_size": cfg.solver.batch_size,
        },
        resources_per_trial={"cpu": 4, "gpu": 0.5},
        num_samples=100,
        local_dir=osp.join(kwargs["project_dir"], "tol_tuning"),
        search_alg=search_alg,
        progress_reporter=reporter,
    )

    best_config = analysis.get_best_config(metric=objective, mode="min")
    print("Best config: ", best_config)

    with open(
        osp.join(kwargs["project_dir"], "tol_tuning", "best_config.json"), "w"
    ) as f:
        json.dump(best_config, f, ensure_ascii=False, indent=4, sort_keys=True)

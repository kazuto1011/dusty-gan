import argparse
import datetime
import json
import os
import os.path as osp
import pprint
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets import define_dataset
from metrics.jsd import compute_jsd
from metrics.point_clouds import compute_cov_mmd_1nna
from metrics.swd import compute_swd
from models import define_G
from utils import denorm_range, get_device, norm_range
from utils.lidar import LiDAR

if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-test", type=int, default=5000)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--tol", type=float, default=0)
    parser.add_argument("--compute-gt", action="store_true")
    args = parser.parse_args()

    device = get_device(True)
    assert ".pth" in args.model_path
    project_dir = "/".join(args.model_path.split("/")[:-2])
    config_path = osp.join(project_dir, ".hydra/config.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.model.gen.shape = cfg.dataset.shape
    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint["G_ema"]
    print(cfg.method_name)
    print("#iterations:", checkpoint["step"])

    evaluation_dir = osp.join(project_dir, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)

    timestamp = datetime.datetime.now().isoformat()

    lidar = LiDAR(
        num_ring=cfg.dataset.shape[0],
        num_points=cfg.dataset.shape[1],
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        angle_file=osp.join(cfg.dataset.root, "angles.pt"),
    ).to(device)

    def sample_latents(B):
        return torch.randn(B, cfg.model.gen.in_ch, device=device)

    def fetch_reals(raw_batch):
        xyz = raw_batch["xyz"].to(device)
        xyz = xyz.flatten(2).transpose(1, 2)  # B,N,3
        pol = raw_batch["depth"].to(device)  # [0,1]
        mask = raw_batch["mask"].to(device).float()
        inv = lidar.invert_depth(pol)
        inv = norm_range(inv)  # [-1,1]
        inv = mask * inv + (1 - mask) * cfg.model.gen.drop_const
        return inv, mask, xyz

    def subsample(points):
        if args.num_points == -1:
            return points
        else:
            return torch.stack(
                [
                    p[torch.randperm(len(p), device=points.device)[: args.num_points]]
                    for p in points
                ],
                dim=0,
            )

    def inv_to_points(inv, tol=1e-8):
        inv = denorm_range(inv).clamp_(0, 1)
        xyz = lidar.inv_depth_to_points(inv, tol)
        xyz = xyz.flatten(2).transpose(1, 2)  # B,N,3
        xyz = subsample(xyz)
        return xyz

    # -------------------------------------------------------------------
    # Real data (test)
    # -------------------------------------------------------------------
    reals_test_path = osp.join(evaluation_dir, "test_data.pt")
    if not osp.exists(reals_test_path):
        test_dataset = define_dataset(cfg.dataset, phase="test")
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.solver.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=False,
        )
        print(test_dataset)

        reals_test = defaultdict(list)
        for data in tqdm(
            test_loader,
            desc="real data (test)",
            dynamic_ncols=True,
            leave=False,
        ):
            inv, mask, points = fetch_reals(data)
            reals_test["2d"].append(inv)
            points = subsample(points)
            reals_test["3d"].append(points)

        for key in reals_test.keys():
            reals_test[key] = torch.cat(reals_test[key], dim=0)
        torch.save(reals_test, reals_test_path)
    else:
        reals_test = torch.load(reals_test_path, map_location="cpu")

    # -------------------------------------------------------------------
    # Real data (train)
    # -------------------------------------------------------------------
    reals_train_path = osp.join(evaluation_dir, "train_data.pt")
    if not osp.exists(reals_train_path):
        train_dataset = define_dataset(cfg.dataset, phase="train")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.solver.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=False,
        )
        print(train_dataset)

        reals_train = defaultdict(list)
        for data in tqdm(
            train_loader,
            desc="real data (train)",
            dynamic_ncols=True,
            leave=False,
        ):
            inv, mask, points = fetch_reals(data)
            reals_train["2d"].append(inv)
            points = subsample(points)
            reals_train["3d"].append(points)
        for key in reals_train.keys():
            reals_train[key] = torch.cat(reals_train[key], dim=0)
        torch.save(reals_train, reals_train_path)
    else:
        reals_train = torch.load(reals_train_path, map_location="cpu")

    for key in reals_train.keys():
        reals_train[key] = subsample(reals_train[key][: args.num_test].to(device))
        reals_test[key] = subsample(reals_test[key][: args.num_test].to(device))
        print(key, reals_test[key].shape)

    # -------------------------------------------------------------------
    # Scores of training set
    # -------------------------------------------------------------------
    if args.compute_gt:
        scores = {}
        scores.update(compute_swd(reals_train["2d"], reals_test["2d"]))
        scores["jsd"] = compute_jsd(
            pcs_gen=reals_train["3d"] / 2.0,
            pcs_ref=reals_test["3d"] / 2.0,
            batchsize=128,
        )
        score_3d = compute_cov_mmd_1nna(
            reals_train["3d"], reals_test["3d"], 2048, ("cd",)
        )
        scores.update(score_3d)
        gt_dir = osp.join(evaluation_dir, "gt")
        os.makedirs(gt_dir, exist_ok=True)
        scores_path = osp.join(gt_dir, "{}.json".format(timestamp))
        with open(scores_path, "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4, sort_keys=True)
        quit()

    # -------------------------------------------------------------------
    # Synthetic data
    # -------------------------------------------------------------------
    G = define_G(cfg.model.gen)
    G.eval()
    G.load_state_dict(state_dict)
    G.to(device)

    fakes = defaultdict(list)
    for i in tqdm(
        range(0, args.num_test, cfg.solver.batch_size),
        desc="synthetic data",
        dynamic_ncols=True,
        leave=False,
    ):
        latent = sample_latents(cfg.solver.batch_size)
        inv = G(latent=latent)["depth"]
        fakes["2d"].append(inv)
        points = inv_to_points(inv, tol=args.tol)
        fakes["3d"].append(points)
    for key in fakes.keys():
        fakes[key] = torch.cat(fakes[key], dim=0)[: args.num_test]

    torch.cuda.empty_cache()

    scores = {}
    # -------------------------------------------------------------------
    # 2.5D evaluation
    # -------------------------------------------------------------------
    scores_swd = compute_swd(
        image1=denorm_range(fakes["2d"]),
        image2=denorm_range(reals_test["2d"]),
        batch_size=128,
    )
    scores.update(scores_swd)

    # -------------------------------------------------------------------
    # 3D evaluation
    # -------------------------------------------------------------------
    scores["jsd"] = compute_jsd(
        pcs_gen=fakes["3d"] / 2.0,
        pcs_ref=reals_test["3d"] / 2.0,
        batchsize=128,
    )

    scores_3d = compute_cov_mmd_1nna(
        pcs_gen=fakes["3d"],
        pcs_ref=reals_test["3d"],
        batch_size=512,
        metrics=("cd",),
    )
    scores.update(scores_3d)

    pprint.pprint(scores)

    method_dir = osp.join(evaluation_dir, f"synth_{args.tol}")
    os.makedirs(method_dir, exist_ok=True)
    scores_path = osp.join(method_dir, "{}.json".format(timestamp))
    with open(scores_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4, sort_keys=True)
    print(f"Saved: {scores_path}")

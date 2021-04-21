import argparse
import multiprocessing
import os
import os.path as osp
from glob import glob

import joblib
import matplotlib.cm as cm
import numba
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from datasets.kitti import KITTIOdometry

# support semantic kitti only for this script
labelmap = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,  # "car"
    11: 2,  # "bicycle"
    13: 5,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,  # "motorcycle"
    16: 5,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,  # "truck"
    20: 5,  # "other-vehicle"
    30: 6,  # "person"
    31: 7,  # "bicyclist"
    32: 8,  # "motorcyclist"
    40: 9,  # "road"
    44: 10,  # "parking"
    48: 11,  # "sidewalk"
    49: 12,  # "other-ground"
    50: 13,  # "building"
    51: 14,  # "fence"
    52: 0,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,  # "lane-marking" to "road" ---------------------------------mapped
    70: 15,  # "vegetation"
    71: 16,  # "trunk"
    72: 17,  # "terrain"
    80: 18,  # "pole"
    81: 19,  # "traffic-sign"
    99: 0,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,  # "moving-car" to "car" ------------------------------------mapped
    253: 7,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,  # "moving-person" to "person" ------------------------------mapped
    255: 8,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,  # "moving-truck" to "truck" --------------------------------mapped
    259: 5,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}
_n_classes = max(labelmap.values()) + 1
_colors = cm.turbo(np.asarray(range(_n_classes)) / (_n_classes - 1))[:, :3] * 255
palette = list(np.uint8(_colors).flatten())


@numba.jit
def scatter(arrary, index, value):
    for (h, w), v in zip(index, value):
        arrary[h, w] = v
    return arrary


def projection(source, grid, order, H, W):
    assert source.ndim == 2, source.ndim
    C = source.shape[1]
    proj = np.zeros((H, W, C))
    proj = np.asarray(proj, dtype=source.dtype)
    proj = scatter(proj, grid[order], source[order])
    return proj


def process_point_clouds(point_path):
    save_dir = lambda x: x.replace("dataset", "dusty-gan")

    # setup point clouds
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
    xyz = points[:, :3]  # xyz
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    depth = np.linalg.norm(xyz, ord=2, axis=1)
    order = np.argsort(-depth)

    # the i-th quadrant
    # suppose the points are ordered counterclockwise
    quads = np.zeros_like(x)
    quads[(x >= 0) & (y >= 0)] = 0  # 1st
    quads[(x < 0) & (y >= 0)] = 1  # 2nd
    quads[(x < 0) & (y < 0)] = 2  # 3rd
    quads[(x >= 0) & (y < 0)] = 3  # 4th

    # split between the 3rd and 1st quadrants
    diff = np.roll(quads, 1) - quads
    (start_inds,) = np.where(diff == 3)  # number of lines
    inds = list(start_inds) + [len(quads)]  # add the last index

    # vertical grid
    line_idx = 63  # ...0
    grid_h = np.zeros_like(x)
    for i in reversed(range(len(start_inds))):
        grid_h[inds[i] : inds[i + 1]] = line_idx
        line_idx -= 1

    # horizontal grid
    yaw = -np.arctan2(y, x)  # [-pi,pi]
    grid_w = (yaw / np.pi + 1) / 2 % 1  # [0,1]
    grid_w = np.floor(grid_w * W)

    grid = np.stack((grid_h, grid_w), axis=-1).astype(np.int32)
    proj = projection(points, grid, order, H, W)

    save_path = save_dir(point_path).replace(".bin", ".npy")
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    np.save(save_path, proj)

    # for semantic kitti
    label_path = point_path.replace("/velodyne", "/labels")
    label_path = label_path.replace(".bin", ".label")
    if osp.exists(label_path):
        labels = np.fromfile(label_path, dtype=np.int32).reshape((-1, 1))
        labels = np.vectorize(labelmap.__getitem__)(labels & 0xFFFF)
        labels = projection(labels, grid, order, H, W)
        save_path = save_dir(label_path).replace(".label", ".png")
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        labels = Image.fromarray(np.uint8(labels[..., 0]), mode="P")
        labels.putpalette(palette)
        labels.save(save_path)


def compute_avg_angles(loader):
    def mean(tensor_, dim):
        tensor = tensor_.clone()
        kwargs = {"dim": dim, "keepdim": True}
        valid = (~torch.isnan(tensor)).float()
        tensor[tensor.isnan()] = 0
        tensor = torch.sum(tensor * valid, **kwargs) / valid.sum(**kwargs)
        return tensor

    max_depth = loader.dataset.max_depth

    xyz = []
    for item in tqdm(loader):
        xyz_batch = item["xyz"]
        xyz.append(xyz_batch)

    xyz = torch.cat(xyz, dim=0)
    depth = torch.norm(xyz, p=2, dim=1, keepdim=True) * max_depth
    print(depth.shape, depth.max(), depth.min())

    valid = (depth > 1e-8).float()
    total_valid = valid.sum(dim=0)

    xy, z = xyz[:, :2], xyz[:, [2]]
    r = torch.norm(xy, p=2, dim=1, keepdim=True)

    x = xyz[:, [0]]
    y = xyz[:, [1]]

    pitch = torch.atan2(z, r)
    yaw = torch.atan2(y, x)

    pitch = torch.sum(pitch * valid, dim=0) / total_valid
    yaw = torch.sum(yaw * valid, dim=0) / total_valid
    angles = torch.cat([pitch, yaw], dim=0)

    mean_p = mean(pitch, 2).expand_as(pitch)
    mean_y = mean(yaw, 1).expand_as(yaw)
    mean_angles = torch.cat([mean_p, mean_y], dim=0)

    valid = (~torch.isnan(angles)).float()
    angles[torch.isnan(angles)] = 0.0
    angles = valid * angles + (1 - valid) * mean_angles

    return angles


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True)
    args = parser.parse_args()

    # 2D maps

    split_dirs = sorted(glob(osp.join(args.root_dir, "dataset/sequences", "*")))
    H, W = 64, 256

    for split_dir in tqdm(split_dirs):
        point_paths = sorted(glob(osp.join(split_dir, "velodyne", "*.bin")))
        joblib.Parallel(
            n_jobs=multiprocessing.cpu_count(), verbose=10, pre_dispatch="all"
        )(
            [
                joblib.delayed(process_point_clouds)(point_path)
                for point_path in point_paths
            ]
        )

    # average angles

    dataset = KITTIOdometry(
        root=osp.join(args.root_dir, "dusty-gan/sequences"),
        split="train",
        shape=(64, 256),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        drop_last=False,
    )
    N = len(dataset)
    print(dataset)

    angles = compute_avg_angles(loader)
    torch.save(angles, osp.join(args.root_dir, "dusty-gan/angles.pt"))

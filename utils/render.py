import matplotlib
import matplotlib.cm as cm
import numba
import numpy as np
import torch

from utils import denorm_range
from utils.geometry import estimate_surface_normal


@numba.jit
def scatter(array, index, value, mask):
    B = array.shape[0]
    for i in range(B):
        for (x, y), v, m in zip(index[i], value[i], mask[i]):
            if m:
                # x in height and y in width
                # lidar coords to image coords
                array[i, -x, -y] = v
    return array


def colorize(tensor, cmap="turbo", vmax=0.4):
    tensor = tensor[..., 0]
    normalizer = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    tensor = mapper.to_rgba(tensor)[..., :3]
    return tensor


def render_pcs(points, normals, L=512, zoom=3.0, R=None):
    B, _, _ = points.shape
    if R is not None:
        points = points @ R
    points = denorm_range(points * zoom)
    points = points.detach().cpu().numpy()
    points = np.int32(points[..., :2] * L)
    mask = (0 < points) & (points < L - 1)
    mask = np.logical_and.reduce(mask, axis=2, keepdims=True)
    normals = normals.detach().cpu().numpy()
    bev = np.ones((B, L, L, 3))
    bev = scatter(bev, points, normals, mask)
    bev = torch.from_numpy(bev.transpose(0, 3, 1, 2))
    return bev


def make_normal(xyz):
    normals = -estimate_surface_normal(xyz, mode="mean")
    normals[normals != normals] = 0.0
    normals = denorm_range(normals).clamp_(0.0, 1.0)
    return normals


def euler_angles_to_rotation_matrix(theta):
    R_x = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(theta[0]), -torch.sin(theta[0])],
            [0, torch.sin(theta[0]), torch.cos(theta[0])],
        ],
        device=theta.device,
    )

    R_y = torch.tensor(
        [
            [torch.cos(theta[1]), 0, torch.sin(theta[1])],
            [0, 1, 0],
            [-torch.sin(theta[1]), 0, torch.cos(theta[1])],
        ],
        device=theta.device,
    )

    R_z = torch.tensor(
        [
            [torch.cos(theta[2]), -torch.sin(theta[2]), 0],
            [torch.sin(theta[2]), torch.cos(theta[2]), 0],
            [0, 0, 1],
        ],
        device=theta.device,
    )

    matrices = [R_x, R_y, R_z]
    R = torch.mm(matrices[2], torch.mm(matrices[1], matrices[0]))
    return R

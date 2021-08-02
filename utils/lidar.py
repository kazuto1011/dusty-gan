import os

import einops
import torch
import torch.nn.functional as F
from torch import nn

from . import render


class Coordinate(nn.Module):
    def __init__(self, min_depth, max_depth, shape, drop_const=0) -> None:
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.H, self.W = shape
        self.drop_const = drop_const
        self.register_buffer("angle", self.init_coordmap(self.H, self.W))

    def init_coordmap(self, H, W):
        raise NotImplementedError

    @staticmethod
    def normalize_minmax(tensor, vmin: float, vmax: float):
        return (tensor - vmin) / (vmax - vmin)

    @staticmethod
    def denormalize_minmax(tensor, vmin: float, vmax: float):
        return tensor * (vmax - vmin) + vmin

    def invert_depth(self, norm_depth):
        # depth to inverse depth
        depth = self.denormalize_minmax(norm_depth, self.min_depth, self.max_depth)
        disp = 1 / depth
        norm_disp = self.normalize_minmax(disp, 1 / self.max_depth, 1 / self.min_depth)
        return norm_disp

    def revert_depth(self, norm_disp, norm=True):
        # inverse depth to depth
        disp = self.denormalize_minmax(
            norm_disp, 1 / self.max_depth, 1 / self.min_depth
        )
        depth = 1 / disp
        if norm:
            return self.normalize_minmax(depth, self.min_depth, self.max_depth)
        else:
            return depth

    def pol_to_xyz(self, polar):
        assert polar.dim() == 4
        grid_cos = torch.cos(self.angle)
        grid_sin = torch.sin(self.angle)
        grid_x = polar * grid_cos[:, [0]] * grid_cos[:, [1]]
        grid_y = polar * grid_cos[:, [0]] * grid_sin[:, [1]]
        grid_z = polar * grid_sin[:, [0]]
        return torch.cat((grid_x, grid_y, grid_z), dim=1)

    def xyz_to_pol(self, xyz):
        return torch.norm(xyz, p=2, dim=1, keepdim=True)

    def inv_to_xyz(self, inv_depth, tol=1e-8):
        valid = torch.abs(inv_depth - self.drop_const) > tol
        depth = self.revert_depth(inv_depth)  # [0,1] depth
        depth = depth * (self.max_depth - self.min_depth) + self.min_depth
        depth /= self.max_depth
        depth *= valid
        points = self.pol_to_xyz(depth)
        return points

    def points_to_depth(self, xyz, drop_value=1, tol=1e-8, tau=2):
        assert xyz.ndim == 3
        device = xyz.device

        x = xyz[..., [0]]
        y = xyz[..., [1]]
        z = xyz[..., [2]]
        r = torch.norm(xyz[..., :2], p=2, dim=2, keepdim=True)
        depth_1d = torch.norm(xyz, p=2, dim=2, keepdim=True)
        weight = 1.0 / torch.exp(tau * depth_1d)
        depth_1d = depth_1d * self.max_depth
        weight *= ((depth_1d > self.min_depth) & (depth_1d < self.max_depth)).detach()

        angle_u = torch.atan2(z, r)  # elevation
        angle_v = torch.atan2(y, x)  # azimuth
        angle_uv = torch.cat([angle_u, angle_v], dim=2)
        angle_uv = einops.rearrange(angle_uv, "b n c -> b n 1 c")
        angle_uv_ref = einops.rearrange(self.angle, "b c h w -> b 1 (h w) c")

        _, ids = torch.norm(angle_uv - angle_uv_ref, p=2, dim=3).min(dim=2)
        id_to_uv = einops.rearrange(
            torch.stack(
                torch.meshgrid(
                    torch.arange(self.H, device=device),
                    torch.arange(self.W, device=device),
                ),
                dim=-1,
            ),
            "h w c -> (h w) c",
        )
        uv = F.embedding(ids, id_to_uv).float()
        depth_2d = render.bilinear_rasterizer(uv, weight * depth_1d, (self.H, self.W))
        depth_2d /= render.bilinear_rasterizer(uv, weight, (self.H, self.W)) + 1e-8
        valid = depth_2d != 0
        depth_2d = self.minmax_norm(depth_2d, self.min_depth, self.max_depth)
        depth_2d[~valid] = drop_value

        return depth_2d, valid


class LiDAR(Coordinate):
    def __init__(
        self,
        num_ring,
        num_points,
        min_depth,
        max_depth,
        angle_file,
    ):
        assert os.path.exists(angle_file), angle_file
        self.angle_file = angle_file
        super().__init__(
            min_depth=min_depth,
            max_depth=max_depth,
            shape=(num_ring, num_points),
        )

    def init_coordmap(self, H, W):
        angle = torch.load(self.angle_file)[None]
        angle = F.interpolate(angle, size=(H, W), mode="bilinear")
        return angle

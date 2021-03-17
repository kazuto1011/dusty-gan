import os

import torch
from torch import nn


class Coordinate(nn.Module):
    def __init__(self, min_depth, max_depth, shape, drop_const=0) -> None:
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.H, self.W = shape
        self.drop_const = drop_const
        self.register_buffer("grid", self.init_coordmap(self.H, self.W))

    def init_coordmap(self, H, W):
        raise NotImplementedError

    def minmax_norm(self, tensor, vmin: float, vmax: float):
        return (tensor - vmin) / (vmax - vmin)

    def minmax_denorm(self, tensor, vmin: float, vmax: float):
        return tensor * (vmax - vmin) + vmin

    def invert_depth(self, norm_depth):
        # depth to inverse depth
        depth = self.minmax_denorm(norm_depth, self.min_depth, self.max_depth)
        disp = 1 / depth
        norm_disp = self.minmax_norm(disp, 1 / self.max_depth, 1 / self.min_depth)
        return norm_disp

    def revert_depth(self, norm_disp, norm=True):
        # inverse depth to depth
        disp = self.minmax_denorm(norm_disp, 1 / self.max_depth, 1 / self.min_depth)
        depth = 1 / disp
        if norm:
            return self.minmax_norm(depth, self.min_depth, self.max_depth)
        else:
            return depth

    def invert_cyl(self, norm_cyl):
        norm_cyl[:, 0] = self.invert_depth(norm_cyl[:, 0])
        return norm_cyl

    def revert_cyl(self, norm_cyl):
        norm_cyl[:, 0] = self.revert_depth(norm_cyl[:, 0])
        return norm_cyl

    def inv_depth_to_points(self, inv_depth, threshold=1e-8):
        valid = torch.abs(inv_depth - self.drop_const) > threshold
        depth = self.revert_depth(inv_depth)  # [0,1] depth
        depth = depth * (self.max_depth - self.min_depth) + self.min_depth
        depth /= self.max_depth
        depth *= valid
        points = self.pol_to_xyz(depth)
        return points

    def pol_to_xyz(self, polar):
        assert polar.dim() == 4
        grid_cos = torch.cos(self.grid)
        grid_sin = torch.sin(self.grid)
        grid_x = polar * grid_cos[:, [0]] * grid_cos[:, [1]]
        grid_y = polar * grid_cos[:, [0]] * grid_sin[:, [1]]
        grid_z = polar * grid_sin[:, [0]]
        return torch.cat((grid_x, grid_y, grid_z), dim=1)

    def pol_to_cyl(self, pol_depth):
        _, _, H, W = pol_depth.shape
        device = pol_depth.device
        grid_elev, _ = self._generate_coordmap(H, W, device)
        map_R = pol_depth * torch.cos(grid_elev)
        map_Z = pol_depth * torch.sin(grid_elev)
        return torch.cat((map_R, map_Z), dim=1)

    def xyz_to_pol(self, xyz):
        return torch.norm(xyz, p=2, dim=1, keepdim=True)

    def xyz_to_cyl(self, xyz):
        xy, z = xyz[:, :2], xyz[:, [2]]
        r = torch.norm(xy, p=2, dim=1, keepdim=True)
        return torch.cat([r, z], dim=1)

    def cyl_to_xyz(self, cyl):
        r, z = cyl[:, [0]], cyl[:, [1]]
        grid_cos = torch.cos(self.grid)
        grid_sin = torch.sin(self.grid)
        x = r * grid_cos[:, [1]]
        y = r * grid_sin[:, [1]]
        return torch.cat((x, y, z), dim=1)

    def cyl_to_pol(self, cyl):
        r, z = cyl[:, [0]], cyl[:, [1]]
        return torch.sqrt(r ** 2 + z ** 2)

    def cyl_to_mask(self, cyl, threshhold=1e-8):
        r, z = cyl[:, [0]], cyl[:, [1]]
        mask = (r > threshhold) & (torch.abs(z) > threshhold)
        return mask


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
        grid = torch.load(self.angle_file)[None]
        return grid

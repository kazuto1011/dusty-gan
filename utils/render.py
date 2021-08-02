import kornia
import matplotlib
import matplotlib.cm as cm
import numba
import numpy as np
import torch
import torch.nn.functional as F


def colorize(tensor, cmap="turbo", vmax=1.0):
    assert tensor.ndim == 2, "got {}".format(tensor.ndim)
    normalizer = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    tensor = mapper.to_rgba(tensor)[..., :3]
    return tensor


def render_point_clouds(
    xyz,
    normals,
    L=512,
    R=None,
    t=None,
    focal_length=1.0,
):
    xyz[..., 2] *= -1

    # extrinsic parameters
    if R is not None:
        assert R.shape[-2:] == (3, 3)
        xyz = xyz @ R
    if t is not None:
        assert t.shape[-1:] == (3,)
        xyz += t

    B, N, _ = xyz.shape
    device = xyz.device

    # intrinsic parameters
    K = torch.eye(3, device=device)
    K[0, 0] = focal_length  # fx
    K[1, 1] = focal_length  # fy
    K[0, 2] = 0.5  # cx, xyz in [-1,1]
    K[1, 2] = 0.5  # cy
    K = K[None]

    # project 3d points onto the image plane
    uv = kornia.geometry.project_points(xyz, K)

    uv = uv * L
    mask = (0 < uv) & (uv < L - 1)
    mask = torch.logical_and(mask[..., [0]], mask[..., [1]])

    # normals = normals.flatten(2).transpose(1, 2)  # B,N,3
    normals = normals * mask

    # z-buffering
    uv = L - uv
    depth = torch.norm(xyz, p=2, dim=-1, keepdim=True)  # B,N,1
    weight = 1.0 / torch.exp(3.0 * depth)
    weight *= (depth > 1e-8).detach()
    bev = bilinear_rasterizer(uv, weight * normals, (L, L))
    bev /= bilinear_rasterizer(uv, weight, (L, L)) + 1e-8
    return bev


def bilinear_rasterizer(coords, values, out_shape):
    """
    https://github.com/VCL3D/SphericalViewSynthesis/blob/master/supervision/splatting.py
    """

    # B,N,C = values.shape
    # B,N,2 = coords.shape

    B, _, C = values.shape
    H, W = out_shape
    device = coords.device

    h = coords[..., [0]].expand(-1, -1, C)
    w = coords[..., [1]].expand(-1, -1, C)

    # Four adjacent pixels
    h_t = torch.floor(h)
    h_b = h_t + 1  # == torch.ceil(h)
    w_l = torch.floor(w)
    w_r = w_l + 1  # == torch.ceil(w)

    h_t_safe = torch.clamp(h_t, 0.0, H - 1)
    h_b_safe = torch.clamp(h_b, 0.0, H - 1)
    w_l_safe = torch.clamp(w_l, 0.0, W - 1)
    w_r_safe = torch.clamp(w_r, 0.0, W - 1)

    weight_h_t = (h_b - h) * (h_t == h_t_safe).detach().float()
    weight_h_b = (h - h_t) * (h_b == h_b_safe).detach().float()
    weight_w_l = (w_r - w) * (w_l == w_l_safe).detach().float()
    weight_w_r = (w - w_l) * (w_r == w_r_safe).detach().float()

    # Bilinear weights
    weight_tl = weight_h_t * weight_w_l
    weight_tr = weight_h_t * weight_w_r
    weight_bl = weight_h_b * weight_w_l
    weight_br = weight_h_b * weight_w_r

    # For stability
    weight_tl *= (weight_tl >= 1e-3).detach().float()
    weight_tr *= (weight_tr >= 1e-3).detach().float()
    weight_bl *= (weight_bl >= 1e-3).detach().float()
    weight_br *= (weight_br >= 1e-3).detach().float()

    values_tl = values * weight_tl  # (B,N,C)
    values_tr = values * weight_tr
    values_bl = values * weight_bl
    values_br = values * weight_br

    indices_tl = (w_l_safe + W * h_t_safe).long()
    indices_tr = (w_r_safe + W * h_t_safe).long()
    indices_bl = (w_l_safe + W * h_b_safe).long()
    indices_br = (w_r_safe + W * h_b_safe).long()

    render = torch.zeros(B, H * W, C).to(device)
    render.scatter_add_(dim=1, index=indices_tl, src=values_tl)
    render.scatter_add_(dim=1, index=indices_tr, src=values_tr)
    render.scatter_add_(dim=1, index=indices_bl, src=values_bl)
    render.scatter_add_(dim=1, index=indices_br, src=values_br)
    render = render.reshape(B, H, W, C).permute(0, 3, 1, 2)

    return render

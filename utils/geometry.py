# re-implemented the following codes with pytorch:
# https://github.com/wkentaro/morefusion/blob/master/morefusion/geometry/estimate_pointcloud_normals.py
# https://github.com/jmccormac/pySceneNetRGBD/blob/master/calculate_surface_normals.py

import torch
import torch.nn.functional as F


def estimate_surface_normal(points, d=2, mode="closest"):
    # estimates a surface normal map from coordinated point clouds

    assert points.dim() == 4, "(B,3,H,W) tensor is expected"
    B, C, H, W = points.shape
    assert C == 3, "points must have (x,y,z)"
    device = points.device

    points = F.pad(points, (0, 0, d, d), mode="constant", value=float("inf"))
    points = F.pad(points, (d, d, 0, 0), mode="circular")
    points = points.permute(0, 2, 3, 1)  # (B,H,W,3)

    # 8 adjacent offsets
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 |   | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    offsets = torch.tensor(
        [
            # (dh,dw)
            (-d, 0),  # 0
            (-d, d),  # 1
            (0, d),  # 2
            (d, d),  # 3
            (d, 0),  # 4
            (d, -d),  # 5
            (0, -d),  # 6
            (-d, -d),  # 7
        ],
        device=device,
    )

    # (B,H,W) indices
    b = torch.arange(B, device=device)[:, None, None]
    h = torch.arange(H, device=device)[None, :, None]
    w = torch.arange(W, device=device)[None, None, :]
    k = torch.arange(8, device=device)

    # anchor points
    b1 = b[:, None]  # (B,1,1,1)
    h1 = h[:, None] + d  # (1,1,H,1)
    w1 = w[:, None] + d  # (1,1,1,W)
    anchors = points[b1, h1, w1]  # (B,H,W,3) -> (B,1,H,W,3)

    # neighbor points
    offset = offsets[k]  # (8,2)
    b2 = b1
    h2 = h1 + offset[None, :, 0, None, None]  # (1,8,H,1)
    w2 = w1 + offset[None, :, 1, None, None]  # (1,8,1,W)
    points1 = points[b2, h2, w2]  # (B,8,H,W,3)

    # anothor neighbor points
    offset = offsets[(k + 2) % 8]
    b3 = b1
    h3 = h1 + offset[None, :, 0, None, None]
    w3 = w1 + offset[None, :, 1, None, None]
    points2 = points[b3, h3, w3]  # (B,8,H,W,3)

    if mode == "closest":
        # find the closest neighbor pair
        diff = torch.norm(points1 - anchors, dim=4)
        diff += torch.norm(points2 - anchors, dim=4)
        i = torch.argmin(diff, dim=1)  # (B,H,W)
        # get normals by cross product
        anchors = anchors[b, 0, h, w]  # (B,H,W,3)
        points1 = points1[b, i, h, w]  # (B,H,W,3)
        points2 = points2[b, i, h, w]  # (B,H,W,3)
        vector1 = points1 - anchors
        vector2 = points2 - anchors
        normals = torch.cross(vector1, vector2, dim=-1)  # (B,H,W,3)
    elif mode == "mean":
        # get normals by cross product
        vector1 = points1 - anchors
        vector2 = points2 - anchors
        normals = torch.cross(vector1, vector2, dim=-1)  # (B,8,H,W,3)
        normals = normals.mean(dim=1)  # (B,H,W,3)
    else:
        raise NotImplementedError(mode)

    normals /= torch.norm(normals, dim=3, keepdim=True)
    normals = normals.permute(0, 3, 1, 2)  # (B,3,H,W)

    return normals


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # sphere
    B, C, H, W = (5, 3, 64, 256)
    h = torch.linspace(0.5, -0.5, H) * np.pi
    h = h.view(1, 1, H, 1).expand(B, 1, H, W)
    w = torch.linspace(-0.5, 0.5, W) * 2 * np.pi
    w = w.view(1, 1, 1, W).expand(B, 1, H, W)
    g = torch.cat([h, w], dim=1)
    c, s = torch.cos(g), torch.sin(g)
    x = c[:, [0]] * c[:, [1]]
    y = c[:, [0]] * s[:, [1]]
    z = s[:, [0]]
    points = torch.cat([x, y, z], dim=1)

    normal = estimate_surface_normal(points, mode="mean")
    normal[normal != normal] = 0.0

    normalize = lambda x: (x + 1) / 2
    to_numpy = lambda x: x.detach().cpu().numpy().transpose(1, 2, 0)

    fig, ax = plt.subplots(4, 1)
    ax[0].imshow(to_numpy(normalize(points)[0, [0]]))
    ax[0].set_title("x")
    ax[1].imshow(to_numpy(normalize(points)[0, [1]]))
    ax[1].set_title("y")
    ax[2].imshow(to_numpy(normalize(points)[0, [2]]))
    ax[2].set_title("z")
    ax[3].imshow(to_numpy(normalize(normal)[0]))
    ax[3].set_title("normal")
    plt.show()

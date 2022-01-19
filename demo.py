import os.path as osp
import sys
from collections import OrderedDict

import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

import utils
from datasets import define_dataset
from utils.geometry import euler_angles_to_rotation_matrix
from utils.interp import lerp, slerp
from utils.metrics.distance import chamfer_distance
from utils.render import render_point_clouds

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # False
torch.backends.cudnn.deterministic = True
color_scale = 1 / 0.4  # for visibility

#############################################################
# utilities
#############################################################


def to_np(tensor, new_W=None):
    tensor = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    tensor = np.uint8(tensor * 255)
    H, W, _ = tensor.shape
    if new_W is not None:
        scale = new_W / W
        tensor = cv2.resize(
            tensor,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
    return tensor


class SphericalOptimizer(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.params = params

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for param in self.params:
            param.data.div_(param.pow(2).mean(dim=1, keepdim=True).add(1e-9).sqrt())
        return loss


def masked_loss(img_ref, img_gen, mask, distance="l1"):
    if distance == "l1":
        loss = F.l1_loss(img_ref, img_gen, reduction="none")
    elif distance == "l2":
        loss = F.mse_loss(img_ref, img_gen, reduction="none")
    else:
        raise NotImplementedError
    loss = (loss * mask).sum(dim=(1, 2, 3))
    loss = loss / mask.sum(dim=(1, 2, 3))
    return loss


def dropout_noise(mask, rate=0.5):
    noise = torch.rand_like(mask)
    noise = (noise < rate).float()
    return mask * noise


def sparse_hlines(mask, rate=0.5):
    skip = int(1 / rate)
    sparse = torch.zeros_like(mask)
    sparse[:, :, ::skip] = 1.0
    return mask * sparse


def sparse_vlines(mask, rate=0.5):
    skip = int(1 / rate)
    sparse = torch.zeros_like(mask)
    sparse[:, :, :, ::skip] = 1.0
    return mask * sparse


def random_lines(mask, rate=0.5):
    _, C, H, W = mask.shape
    perm = torch.randperm(H)[: int(H * (1 - rate))]
    mask[:, :, perm] = 0.0
    return mask


def corrupt_half(mask):
    _, C, H, W = mask.shape
    mask[..., W // 2 :] = 0.0
    return mask


def corrupt_quarter(mask):
    _, C, H, W = mask.shape
    mask[..., : W // 4] = 0.0
    mask[..., W // 2 : W * 3 // 4] = 0.0
    return mask


def additive_noise(depth, strength=0.01):
    noise = torch.randn_like(depth) * strength
    return depth + noise


def closing(inv):
    inv = kornia.filters.median_blur(inv, (3, 3))
    valid = torch.zeros_like(inv)
    while (1 - valid).sum() > 0:
        valid = (inv > 1e-8).float()
        filled = F.max_pool2d(inv, (3, 3), 1, (1, 1))
        inv = valid * inv + (1 - valid) * filled
    return inv


def apply_corruption(dep_ref, mask_ref, corruption):
    torch.manual_seed(0)
    if corruption == "additive noise":
        dep_ref = additive_noise(dep_ref, 0.01)
    elif corruption == "low resolution":
        mask_ref = sparse_hlines(mask_ref, 1 / 8)
    elif corruption == "dropout":
        mask_ref = dropout_noise(mask_ref, rate=0.1)
    elif corruption == "closing":
        dep_ref = closing(dep_ref)
        mask_ref = torch.ones_like(mask_ref)
    return dep_ref, mask_ref


@st.cache(allow_output_mutation=True, show_spinner=True)
def get_feature_shapes(G, z_shape, device):

    feature_shapes = {}
    hooks = []

    def forward_hook(name):
        def _forward_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                feature_shapes[name] = (module.__class__.__name__, output.shape)
            return output

        return _forward_hook

    for name, module in G.named_modules():
        hooks.append(module.register_forward_hook(forward_hook(name)))

    # dummy
    with torch.no_grad():
        G(torch.randn(*z_shape, device=device))

    for h in hooks:
        h.remove()

    return feature_shapes


@st.cache(allow_output_mutation=True, show_spinner=True)
def setup_synthesis():
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    return utils.setup(model_path, config_path, ema=True, fix_noise=True)


@st.cache(allow_output_mutation=True, show_spinner=True)
def setup_inversion():
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    cfg, G, lidar, device = utils.setup(
        model_path, config_path, ema=True, fix_noise=True
    )
    if osp.exists(cfg.dataset.root):
        dataset = define_dataset(cfg.dataset, phase="test")
    else:
        dataset = None
    return cfg, G, lidar, device, dataset


def set_view_options(device):
    t_z = (
        st.slider(
            "zoom",
            min_value=1,
            max_value=120,
            value=60,
            step=1,
            format="%dm",
        )
        / 120.0
    )
    yaw = (
        -st.slider(
            "yaw",
            min_value=-180,
            max_value=180,
            value=-45,
            step=1,
            format="%d°",
        )
        / 180
        * np.pi
    )
    pitch = (
        st.slider(
            "pitch",
            min_value=0,
            max_value=90,
            value=60,
            step=1,
            format="%d°",
        )
        / 180
        * np.pi
    )
    t = torch.tensor([0.1 * t_z, 0.0, t_z], device=device)
    angles = torch.tensor([0, pitch, yaw], device=device).float()
    R = euler_angles_to_rotation_matrix(angles)
    cmap = st.selectbox("color map", plt.colormaps(), plt.colormaps().index("turbo"))
    return R, t, cmap


def set_synthesis_options():
    num_samples = int(st.slider("#samples", 1, 8, value=4))
    latent_type = st.selectbox("latent type", ["random", "lerp", "slerp"])
    return num_samples, latent_type


def set_inversion_options(n_max):
    n = int(st.number_input("sample ID", min_value=0, max_value=n_max))
    corruption = st.selectbox(
        "corruption",
        options=[None, "additive noise", "low resolution", "dropout", "closing"],
        help="corruption on the target data",
    )
    distance = st.multiselect(
        "loss",
        options=["l1", "l2", "chamfer"],
        default=["l1"],
        help='"l1" and "l2" are 2D metrics, and "chamfer" is 3D metrics',
    )
    num_step = int(
        st.number_input(
            "#iterations",
            value=1000,
            help="number of steps to optimize the latent vevtor",
        )
    )
    return n, corruption, distance, num_step


#############################################################
# demos
#############################################################


@torch.no_grad()
def synthesis():
    cfg, G, lidar, device = setup_synthesis()

    with st.sidebar.expander("run options"):
        num_samples, latent_type = set_synthesis_options()

    with st.sidebar.expander("view options"):
        R, t, cmap = set_view_options(device)

    run_synthesis = st.button("run")

    if run_synthesis:
        if latent_type == "random":
            latent = torch.randn(num_samples, cfg.model.gen.in_ch, device=device)
        elif latent_type == "slerp":
            latent = torch.randn(2, cfg.model.gen.in_ch, device=device)
            latent = [
                slerp(w, latent[[0]], latent[[1]])
                for w in torch.linspace(0, 1, num_samples)
            ]
            latent = torch.cat(latent, dim=0)
        elif latent_type == "lerp":
            latent = torch.randn(2, cfg.model.gen.in_ch, device=device)
            latent = [
                lerp(w, latent[[0]], latent[[1]])
                for w in torch.linspace(0, 1, num_samples)
            ]
            latent = torch.cat(latent, dim=0)

        out = G(latent)
        out = utils.postprocess(out, lidar)

        export = []
        if "depth_orig" in out:
            tensor = utils.colorize(out["depth_orig"] * color_scale, cmap)
            export.append(("inverse_depth", tensor))
        if "confidence" in out:
            if out["confidence"].shape[1] == 2:
                tensor = utils.colorize(out["confidence"][:, [0]], cmap)
                export.append(("measurability pix", tensor))
                tensor = utils.colorize(out["confidence"][:, [1]], cmap)
                export.append(("measurability img", tensor))
                export.append(("mask pix", out["mask"][:, [0]]))
                export.append(("mask img", out["mask"][:, [1]]))
                export.append(("mask", torch.prod(out["mask"], dim=1, keepdim=True)))
            else:
                tensor = utils.colorize(out["confidence"], cmap)
                export.append(("measurability", tensor))
                export.append(("mask", out["mask"]))
        tensor = utils.colorize(out["depth"] * color_scale, cmap)
        export.append(("inverse depth w/ point drops", tensor))
        export.append(("point normal", out["normals"]))
        bev = render_point_clouds(
            utils.flatten(out["points"]),
            utils.flatten(out["normals"]),
            L=512,
            R=R,
            t=t,
        )
        bev_alpha = torch.all(bev != 0.0, dim=1, keepdim=True).float()
        export.append(("point clouds", torch.cat([bev, bev_alpha], dim=1)))

        cols = st.columns(num_samples)

        for i in range(num_samples):
            with cols[i]:
                for caption, tensor in export:
                    st.image(
                        to_np(tensor[i]),
                        caption=caption,
                        use_column_width=True,
                        output_format="png",
                    )


def inversion():
    cfg, G, lidar, device, dataset = setup_inversion()

    if dataset is None:
        st.write("please set the dataset path!")
        st.markdown("e.g. `ln -s /path/to/your/kitti_odometry ./data/kitti_odometry`")
        return

    # options
    with st.sidebar.expander("run options"):
        n, corruption, distance, num_step = set_inversion_options(n_max=len(dataset))

        num_code = st.select_slider(
            "#latents",
            [2 ** i for i in range(7)],
            help="if >1, mGANprior is applied",
        )
        if num_code != 1:
            feature_shapes = get_feature_shapes(G, (1, cfg.model.gen.in_ch), device)
            layer_name = st.selectbox(
                "composition layer",
                list(feature_shapes.keys()),
                help="a layer name to fuse the multiple latents",
            )
            _, feature_shape = feature_shapes[layer_name]
            _, feature_ch, _, _ = feature_shape  # B,C,H,W

    with st.sidebar.expander("view options"):
        R, t, cmap = set_view_options(device)

    # stylegan2 settings
    perturb_latent = True
    noise_ratio = 0.75
    noise_sigma = 1.0
    lr_rampup_ratio = 0.05
    lr_rampdown_ratio = 0.25

    def lr_schedule(iteration):
        t = iteration / num_step
        gamma = np.clip((1.0 - t) / lr_rampdown_ratio, None, 1.0)
        gamma = 0.5 - 0.5 * np.cos(gamma * np.pi)
        gamma = gamma * np.clip(t / lr_rampup_ratio, None, 1.0)
        return gamma

    # get target data
    item = dataset.__getitem__(n)
    dep_ref = item["depth"][None].to(device).float()
    mask_ref = item["mask"][None].to(device).float()
    inv_ref_full = lidar.invert_depth(dep_ref)
    inv_ref_full = mask_ref * inv_ref_full + (1 - mask_ref) * 0.0
    points_ref_full = lidar.inv_to_xyz(inv_ref_full)
    normals_ref_full = utils.xyz_to_normal(points_ref_full, mode="closest")

    # corruption process
    dep_ref, mask_ref = apply_corruption(dep_ref, mask_ref, corruption)
    inv_ref = lidar.invert_depth(dep_ref)
    inv_ref = mask_ref * inv_ref + (1 - mask_ref) * 0.0

    points_ref = lidar.inv_to_xyz(inv_ref)
    bev_ref = render_point_clouds(
        utils.flatten(points_ref),
        utils.flatten(normals_ref_full),
        L=512,
        R=R,
        t=t,
    )

    run_inversion = st.button("run")
    progress_title = st.text("progress")
    progress_bar = st.progress(0)
    cols = st.columns(2)

    with cols[0]:
        st.text(f"target #{n}")
        st.image(
            np.dstack(
                [
                    to_np(bev_ref[0]),
                    to_np(torch.all(bev_ref != 0.0, dim=1, keepdim=True).float()[0]),
                ]
            ),
            caption="point clouds",
            output_format="png",
            use_column_width=True,
        )
        st.image(
            to_np(utils.colorize(inv_ref * color_scale, cmap)[0]),
            caption="inverse depth",
            output_format="png",
            use_column_width=True,
        )
        st.image(
            to_np(mask_ref[0]),
            caption="mask",
            output_format="png",
            use_column_width=True,
        )
        if corruption != None:
            st.image(
                to_np(utils.colorize(inv_ref_full * color_scale, cmap)[0]),
                caption="inverse depth (full)",
                output_format="png",
                use_column_width=True,
            )

    with cols[1]:
        st.text("inversion")
        if len(distance) == 0:
            st.error("loss should be selected")
            return
        show_gen_bev = st.empty()
        show_gen_depth = st.empty()
        show_gen_depth_orig = st.empty()
        show_gen_mask = st.empty()

    if run_inversion:
        # trainable latent code
        torch.manual_seed(0)
        latent = torch.randn(num_code, cfg.model.gen.in_ch, device=device)
        latent.div_(latent.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt())
        latent = torch.nn.Parameter(latent).requires_grad_()

        optim_z = SphericalOptimizer([latent], lr=0.1)
        scheduler_z = torch.optim.lr_scheduler.LambdaLR(optim_z, lr_lambda=lr_schedule)

        if num_code != 1:
            alpha = torch.full(
                (num_code, feature_ch, 1, 1),
                fill_value=1 / num_code,
                device=device,
            )
            alpha = torch.nn.Parameter(alpha).requires_grad_()

            # multi-code inversion
            def feature_composition(m, i, o):
                o = (o * alpha).sum(dim=0, keepdim=True)
                return o

            hooks = []
            for name, module in G.named_modules():
                module._forward_hooks = OrderedDict()  # reset hooks
                if name == layer_name:
                    hooks.append(module.register_forward_hook(feature_composition))

            optim_a = torch.optim.Adam([alpha], lr=0.001)
            scheduler_a = torch.optim.lr_scheduler.LambdaLR(
                optim_a, lr_lambda=lr_schedule
            )

        # optimize the latent
        for cur_step in range(num_step):
            progress = cur_step / num_step

            # noise
            w = max(0.0, 1.0 - progress / noise_ratio)
            noise_strength = 0.05 * noise_sigma * w ** 2
            noise = noise_strength * torch.randn_like(latent)

            # forward G
            out = G(latent + noise if perturb_latent else latent)
            out = utils.postprocess(out, lidar)

            if "dusty" in cfg.model.gen.arch:
                inv_gen = out["depth_orig"]
            else:
                inv_gen = out["depth"]

            # loss
            loss = 0
            if "chamfer" in distance:
                dl, dr = chamfer_distance(
                    utils.flatten(points_ref),
                    utils.flatten(out["points"]),
                )
                loss += dl.mean(dim=1) + dr.mean(dim=1)
            if "l1" in distance:
                loss += masked_loss(inv_ref, inv_gen, mask_ref, "l1").mean()
            if "l2" in distance:
                loss += masked_loss(inv_ref, inv_gen, mask_ref, "l2").mean()

            # per-sample gradients
            optim_z.zero_grad()
            if num_code != 1:
                optim_a.zero_grad()
            loss.backward(gradient=torch.ones_like(loss))
            optim_z.step()
            scheduler_z.step()
            if num_code != 1:
                optim_a.step()
                scheduler_a.step()

            # make figures
            if "depth_orig" in out:
                inv_orig_gen = utils.colorize(out["depth_orig"] * color_scale, cmap)
            if "mask" in out:
                if out["mask"].shape[1] == 2:
                    mask_gen = torch.prod(out["mask"], dim=1, keepdim=True)
                else:
                    mask_gen = out["mask"]

            inv_gen = utils.colorize(out["depth"] * color_scale, cmap)
            bev_gen = render_point_clouds(
                utils.flatten(out["points"]),
                utils.flatten(out["normals"]),
                L=512,
                R=R,
                t=t,
            )

            show_gen_bev.image(
                np.dstack([to_np(bev_gen[0]), to_np((bev_gen != 0.0).float()[0, [0]])]),
                caption="point clouds",
                output_format="png",
                use_column_width=True,
            )
            show_gen_depth.image(
                to_np(inv_gen[0]),
                caption="inverse depth w/ point drops",
                output_format="png",
                use_column_width=True,
            )
            if "depth_orig" in out:
                show_gen_depth_orig.image(
                    to_np(inv_orig_gen[0]),
                    caption="inverse depth",
                    output_format="png",
                    use_column_width=True,
                )
            if "confidence" in out:
                show_gen_mask.image(
                    to_np(mask_gen[0]),
                    caption="sampled mask",
                    output_format="png",
                    use_column_width=True,
                )

            progress_bar.progress(progress)
            progress_title.text(f"progress {int(progress*100):d}%")

        progress_bar.progress(1.0)
        progress_title.text(f"progress completed!")
        st.balloons()


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.title("dusty-gan demo")
    st.text(
        'Kazuto Nakashima and Ryo Kurazume, "Learning to Drop Points for LiDAR Scan Synthesis", IROS 2021'
    )

    st.sidebar.title("settings")
    mode = st.sidebar.selectbox("mode", ["synthesis", "inversion"])

    if mode == "synthesis":
        synthesis()
    elif mode == "inversion":
        inversion()

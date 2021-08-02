import gc
import os.path as osp

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import define_D, define_G
from models.dusty import GumbelSigmoid
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.geometry import estimate_surface_normal
from utils.lidar import LiDAR


def init_weights(cfg):
    init_type = cfg.init.type
    gain = cfg.init.gain
    nonlinearity = cfg.relu_type

    def init_func(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight, gain=gain)
            elif init_type == "kaiming":
                if nonlinearity == "relu":
                    nn.init.kaiming_normal_(m.weight, 0, "fan_in", "relu")
                elif nonlinearity == "leaky_relu":
                    nn.init.kaiming_normal_(m.weight, 0.2, "fan_in", "learky_relu")
                else:
                    raise NotImplementedError(f"Unknown nonlinearity: {nonlinearity}")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(f"Unknown initialization: {init_type}")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_func


def set_requires_grad(net, requires_grad: bool = True):
    for param in net.parameters():
        param.requires_grad = requires_grad


def zero_grad(optim):
    for group in optim.param_groups:
        for p in group["params"]:
            p.grad = None


def sigmoid_to_tanh(x: torch.Tensor):
    """[0,1] -> [-1,+1]"""
    out = x * 2.0 - 1.0
    return out


def tanh_to_sigmoid(x: torch.Tensor):
    """[-1,+1] -> [0,1]"""
    out = (x + 1.0) / 2.0
    return out


def get_device(cuda: bool):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        for i in range(torch.cuda.device_count()):
            print("device {}: {}".format(i, torch.cuda.get_device_name(i)))
    else:
        print("device: CPU")
    return device


def noise(tensor: torch.Tensor, std: float = 0.1):
    noise = tensor.clone().normal_(0, std)
    return tensor + noise


def print_gc():
    # https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(type(obj), obj.size())
        except:
            pass


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


@torch.no_grad()
def setup(model_path, config_path, ema=True, fix_noise=True, cuda=True):
    device = get_device(cuda)

    # project_dir = "/".join(model_path.split("/")[:-2])
    # config_path = osp.join(project_dir, ".hydra/config.yaml")

    cfg = OmegaConf.load(config_path)
    cfg.model.gen.shape = cfg.dataset.shape
    cfg.model.dis.shape = cfg.dataset.shape

    assert ".pth" in model_path
    checkpoint = torch.load(model_path, map_location="cpu")
    if ema:
        G_state_dict = checkpoint["G_ema"]
    else:
        G_state_dict = checkpoint["G"]
    print("#iterations:", checkpoint["step"])

    # Model
    G = define_G(cfg)
    G.eval()
    G.load_state_dict(G_state_dict)
    G.to(device)

    if fix_noise:

        def set_gumbel_noise(m, i):
            if m.fixed_noise is None:
                m.fixed_noise = m.logistic_noise(i[0])[[0]]

        for m in G.modules():
            if isinstance(m, GumbelSigmoid):
                m.register_forward_pre_hook(set_gumbel_noise)

    lidar = LiDAR(
        num_ring=cfg.dataset.shape[0],
        num_points=cfg.dataset.shape[1],
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        angle_file=osp.join(cfg.dataset.root, "angles.pt"),
    )
    lidar.to(device)

    return cfg, G, lidar, device


def postprocess(synth, lidar, tol=1e-8, normal_mode="closest"):

    out = {}
    for key, value in synth.items():
        if key == "depth":
            out["depth"] = tanh_to_sigmoid(synth["depth"]).clamp_(0, 1)
        elif key == "depth_orig":
            out["depth_orig"] = tanh_to_sigmoid(synth["depth_orig"]).clamp_(0, 1)
        elif key == "confidence":
            out["confidence"] = torch.sigmoid(synth["confidence"])
        else:
            out[key] = value

    out["points"] = lidar.inv_to_xyz(out["depth"], tol)
    out["normals"] = xyz_to_normal(out["points"], mode=normal_mode)

    return out


def save_videos(frames, filename, fps=30.0):
    N = len(frames)
    H, W, C = frames[0].shape
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename + ".mp4", codec, fps, (W, H))
    for frame in tqdm(frames, desc="Writing..."):
        writer.write(frame[..., ::-1])
    writer.release()
    cv2.destroyAllWindows()
    print("Saved:", filename)


def colorize(tensor, cmap="turbo"):
    if tensor.ndim == 4:
        B, C, H, W = tensor.shape
        assert C == 1, f"expected (B,1,H,W) tensor, but got {tensor.shape}"
        tensor = tensor.squeeze(1)
    assert tensor.ndim == 3, f"got {tensor.ndim}!=3"

    device = tensor.device

    colors = eval(f"cm.{cmap}")(np.linspace(0, 1, 256))[:, :3]
    color_map = torch.tensor(colors, device=device, dtype=tensor.dtype)  # (256,3)

    tensor = tensor.clamp_(0, 1)
    tensor = tensor * 255.0
    index = torch.round(tensor).long()

    return F.embedding(index, color_map).permute(0, 3, 1, 2)


def flatten(tensor_BCHW):
    return tensor_BCHW.flatten(2).permute(0, 2, 1).contiguous()


def xyz_to_normal(xyz, mode="closest"):
    normals = -estimate_surface_normal(xyz, mode=mode)
    normals[normals != normals] = 0.0
    normals = tanh_to_sigmoid(normals).clamp_(0.0, 1.0)
    return normals


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

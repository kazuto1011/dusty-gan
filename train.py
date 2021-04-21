import os
import warnings

import hydra
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from trainers import dcgan_trainer
from utils import denorm_range
from utils.render import colorize, make_bev, make_normal

warnings.simplefilter("ignore")

# save images to tensorboard
def save_img(writer, tensor, tag, step, color=True):
    grid = make_grid(tensor.detach(), nrow=4)
    grid = grid.cpu().numpy()  # CHW
    if color:
        grid = grid[0]  # HW
        grid = colorize(grid).transpose(2, 0, 1)  # CHW
    writer.add_image(tag, grid, step)


def main_worker(gpu, cfg):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    ngpus = torch.cuda.device_count()

    # setup for distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend=cfg.dist_backend, world_size=ngpus, rank=gpu)
    print("init device {}: {}".format(gpu, torch.cuda.get_device_name(gpu)))

    # determine the batchsize for this worker
    assert cfg.solver.batch_size % ngpus == 0
    local_batch_size = int(cfg.solver.batch_size / ngpus)
    assert local_batch_size % cfg.solver.num_accumulation == 0
    local_batch_size = int(local_batch_size / cfg.solver.num_accumulation)

    local_cfg = OmegaConf.create(
        {
            "gpu": gpu,
            "ngpus": ngpus,
            "batch_size": local_batch_size,
            "num_workers": int((cfg.num_workers + ngpus - 1) / ngpus),
        }
    )

    # setup trainer
    if cfg.trainer == "dcgan":
        trainer = dcgan_trainer.Trainer(cfg, local_cfg)
    else:
        raise NotImplementedError

    total_img = cfg.solver.total_kimg * 1000
    total_iteration = int(total_img / cfg.solver.batch_size)
    current_img = lambda i: int(i * cfg.solver.batch_size)
    best_score = 1e10

    if dist.get_rank() == 0:
        # init tensorboatd
        writer = SummaryWriter()
        # real data
        inv, _ = trainer.fetch_reals(next(trainer.loader))
        inv = denorm_range(inv).clamp_(0, 1)
        xyz = trainer.lidar.inv_depth_to_points(inv)
        normals = make_normal(xyz)
        bev = make_bev(xyz, normals)
        save_img(writer, inv, "real/inv", 1)
        save_img(writer, normals, "real/normal", 1, False)
        save_img(writer, bev, "real/bev", 1, False)

    # training loop
    for i in tqdm(
        range(1, total_iteration + 1),
        desc="iteration",
        dynamic_ncols=True,
        disable=not dist.get_rank() == 0,
    ):
        # training
        scalars = trainer.step(i)

        # logging
        if dist.get_rank() == 0:

            step = current_img(i)

            # save scalars
            if i % cfg.solver.checkpoint.save_stats == 0:
                # training stats
                for key, scalar in scalars.items():
                    writer.add_scalar(key, scalar, step)
                # system stats
                memory = psutil.virtual_memory().used / 1024 ** 3
                writer.add_scalar("system/cpu/ram (GB)", memory, step)
                memory = psutil.swap_memory().used / 1024 ** 3
                writer.add_scalar("system/cpu/swap (GB)", memory, step)
                usage = psutil.cpu_percent()
                writer.add_scalar("system/cpu/utilization (%)", usage, step)
                for j in range(torch.cuda.device_count()):
                    device = torch.cuda.device(j)
                    memory = torch.cuda.memory_allocated(device) / 1024 ** 3
                    writer.add_scalar(f"system/gpu{j}/allocated (GB)", memory, step)
                    memory = torch.cuda.memory_reserved(device) / 1024 ** 3
                    writer.add_scalar(f"system/gpu{j}/reserved (GB)", memory, step)

            # save images
            if i % cfg.solver.checkpoint.save_image == 0:
                synth = trainer.generate()
                if "depth" in synth:
                    inv = synth["depth"]
                    inv = denorm_range(inv).clamp_(0, 1)
                    xyz = trainer.lidar.inv_depth_to_points(inv)
                    normals = make_normal(xyz)
                    bev = make_bev(xyz, normals)
                    save_img(writer, inv, "synth/inv", step)
                    save_img(writer, normals, "synth/normal", step, False)
                    save_img(writer, bev, "synth/bev", step, False)
                if "depth_orig" in synth:
                    inv = synth["depth_orig"]
                    inv = denorm_range(inv).clamp_(0, 1)
                    save_img(writer, inv, "synth/inv/orig", step)
                if "confidence" in synth:
                    conf = torch.sigmoid(synth["confidence"])
                    if conf.shape[1] == 2:
                        save_img(writer, conf[:, [0]], "synth/confidence/pix", step)
                        save_img(writer, conf[:, [1]], "synth/confidence/img", step)
                    elif conf.shape[1] == 1:
                        save_img(writer, conf[:, [0]], "synth/confidence/pix", step)
                if "mask" in synth:
                    mask = synth["mask"]
                    if mask.shape[1] == 2:
                        save_img(writer, mask[:, [0]], "synth/mask/pix", step, False)
                        save_img(writer, mask[:, [1]], "synth/mask/img", step, False)
                        mask = mask[:, [0]] * mask[:, [1]]
                        save_img(writer, mask, "synth/mask", step, False)
                    elif mask.shape[1] == 1:
                        save_img(writer, mask, "synth/mask", step, False)

            # validation
            if i % cfg.solver.checkpoint.test == 0:
                scores = trainer.validation()
                for key, scalar in scores.items():
                    writer.add_scalar("score/" + key, scalar, step)

                if scores["1-nn-accuracy-cd"] <= best_score:
                    best_score = scores["1-nn-accuracy-cd"]
                    trainer.save_models("best", int(step))

            # save models
            if i % cfg.solver.checkpoint.save_model == 0:
                suffix = "{:010d}".format(int(step))
                trainer.save_models(suffix, int(step))

    # save the final model
    if dist.get_rank() == 0:
        step = current_img(total_iteration)
        suffix = "{:010d}".format(int(step))
        trainer.save_models(suffix, int(step))


@hydra.main(config_path="configs/config.yaml")
def main(cfg):

    print(cfg.pretty())

    os.makedirs("models", exist_ok=True)
    mp.spawn(main_worker, args=(cfg,), nprocs=torch.cuda.device_count())


if __name__ == "__main__":
    main()

import os
import tempfile

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

import utils
from trainers import dcgan_amp
from utils.render import colorize, render_point_clouds

try:
    import wandb

    publish_wandb = True
except:
    publish_wandb = False

scale = 1 / 0.4  # for visibility

# save images to tensorboard
def log_imgs(writer, tensor, tag, step, color=True):
    grid = make_grid(tensor.detach(), nrow=4)
    grid = grid.cpu().numpy()  # CHW
    if color:
        grid = grid[0]  # HW
        grid = colorize(grid).transpose(2, 0, 1)  # CHW
    writer.add_image(tag, grid, step)


def main_worker(gpu, cfg, temp_dir):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    ngpus = torch.cuda.device_count()

    # setup for distributed training
    init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
    torch.distributed.init_process_group(
        init_method=f"file://{init_file}",
        backend=cfg.dist_backend,
        world_size=ngpus,
        rank=gpu,
    )
    print("rank {}: {} gpu".format(gpu, torch.cuda.get_device_name(gpu)))

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
    trainer = dcgan_amp.Trainer(cfg, local_cfg)

    total_img = cfg.solver.total_kimg * 1000
    total_iteration = int(total_img / cfg.solver.batch_size)
    iteration_to_imgs = lambda i: int(i * cfg.solver.batch_size)

    if dist.get_rank() == 0:
        # init wandb
        if publish_wandb and cfg.publish_wandb:
            wandb.init(
                project="dusty-gan-iros2021",
                config=OmegaConf.to_container(cfg),
            )
            wandb.tensorboard.patch(save=False)
        # init tensorboatd
        writer = SummaryWriter()
        # real data
        inv_real, mask_real = trainer.fetch_reals(next(trainer.loader))
        real = trainer.postprocess({"depth": inv_real, "mask": mask_real})
        real_aug = trainer.postprocess({"depth": trainer.A(inv_real)})
        bev = render_point_clouds(
            utils.flatten(real["points"]),
            utils.flatten(real["normals"]),
            t=torch.tensor([0, 0, 0.5], device=trainer.device),
        )
        log_imgs(writer, real["depth"] * scale, "real/inv", 1)
        log_imgs(writer, real_aug["depth"] * scale, "real/inv_aug", 1)
        log_imgs(writer, real["normals"], "real/normal", 1, False)
        log_imgs(writer, bev, "real/bev", 1, False)

        print("iteration start:", trainer.start_iteration + 1)
        print("iteration total:", total_iteration)

    # training loop
    for i in tqdm(
        range(trainer.start_iteration + 1, total_iteration + 1),
        desc="iteration",
        dynamic_ncols=True,
        disable=not dist.get_rank() == 0,
    ):
        # training
        scalars = trainer.step(i)
        step = iteration_to_imgs(i)

        # logging
        if dist.get_rank() == 0:

            # log scalars
            if i % cfg.solver.checkpoint.save_stats == 0:
                # training stats
                for key, scalar in scalars.items():
                    writer.add_scalar(key, scalar, step)

            # log images
            if i % cfg.solver.checkpoint.save_image == 0:
                out = trainer.generate()
                if "depth" in out:
                    bev = render_point_clouds(
                        utils.flatten(out["points"]),
                        utils.flatten(out["normals"]),
                        t=torch.tensor([0, 0, 0.5], device=trainer.device),
                    )
                    log_imgs(writer, out["depth"] * scale, "synth/inv", step)
                    log_imgs(writer, out["normals"], "synth/normal", step, False)
                    log_imgs(writer, bev, "synth/bev", step, False)
                if "depth_orig" in out:
                    log_imgs(writer, out["depth_orig"] * scale, "synth/inv/orig", step)
                if "confidence" in out:
                    conf = out["confidence"]
                    if conf.shape[1] == 2:
                        log_imgs(writer, conf[:, [0]], "synth/confidence/pix", step)
                        log_imgs(writer, conf[:, [1]], "synth/confidence/img", step)
                    elif conf.shape[1] == 1:
                        log_imgs(writer, conf[:, [0]], "synth/confidence", step)
                if "mask" in out:
                    mask = out["mask"]
                    if mask.shape[1] == 2:
                        log_imgs(writer, mask[:, [0]], "synth/mask/pix", step, False)
                        log_imgs(writer, mask[:, [1]], "synth/mask/img", step, False)
                        mask = torch.prod(mask, dim=1, keepdim=True)
                        log_imgs(writer, mask, "synth/mask", step, False)
                    elif mask.shape[1] == 1:
                        log_imgs(writer, mask, "synth/mask", step, False)

            # validation
            if i % cfg.solver.checkpoint.test == 0:
                scores = trainer.validation()
                for key, scalar in scores.items():
                    writer.add_scalar("score/" + key, scalar, step)

            # save models
            if i % cfg.solver.checkpoint.save_model == 0:
                trainer.save_models("{:010d}".format(int(step)), int(step))

    # save the final model
    if dist.get_rank() == 0:
        step = iteration_to_imgs(total_iteration)
        trainer.save_models("{:010d}".format(int(step)), int(step))

    if publish_wandb and cfg.publish_wandb:
        wandb.finish()


@hydra.main(config_path="configs", config_name="config")
def main(cfg):

    print(OmegaConf.to_yaml(cfg))

    # path settings
    os.makedirs("models", exist_ok=True)
    if cfg.dataset.root[0] != "/":
        cfg.dataset.root = os.path.join(get_original_cwd(), cfg.dataset.root)
    if cfg.resume is not None and cfg.resume[0] != "/":
        cfg.resume = os.path.join(get_original_cwd(), cfg.resume)

    # run distributed training
    with tempfile.TemporaryDirectory() as temp_dir:
        mp.spawn(main_worker, args=(cfg, temp_dir), nprocs=torch.cuda.device_count())


if __name__ == "__main__":
    main()

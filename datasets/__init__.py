from datasets import kitti, mpo


def define_dataset(cfg, phase: str = "train", modality=["depth"]):
    if cfg.name == "kitti_odometry":
        dataset = kitti.KITTIOdometry(
            root=cfg.root,
            split=phase,
            shape=cfg.shape,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            flip=cfg.flip and phase == "train",
            modality=modality,
        )
    elif cfg.name == "sparse_mpo":
        dataset = mpo.SparseMPO(
            root=cfg.root,
            split=phase,
            shape=cfg.shape,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            flip=cfg.flip and phase == "train",
            modality=modality,
        )
    else:
        raise NotImplementedError(cfg.name)
    return dataset

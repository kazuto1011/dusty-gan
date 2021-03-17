from datasets import kitti, mpo


def define_dataset(cfg, phase: str = "train", modality=["depth"]):
    if cfg.name == "kitti_odometry":
        dataset = kitti.KITTIOdometry(
            root=cfg.root,
            split=phase,
            shape=cfg.shape,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            flip=cfg.flip,
            modality=modality,
        )
    elif cfg.name == "mpo_sparse":
        dataset = mpo.MPOSparse(
            root=cfg.root,
            split=phase,
            shape=cfg.shape,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            flip=cfg.flip,
            modality=modality,
        )
    else:
        raise NotImplementedError
    return dataset

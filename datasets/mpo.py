import os.path as osp
import random
from glob import glob

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

CONFIG = {
    "split": {
        "train": [0, 1, 2, 3, 4, 5, 6],
        "val": [7],
        "test": [8, 9, 10],
    },
}


class MPOSparse(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split,
        shape=(32, 256),
        min_depth=0.9,
        max_depth=120.0,
        flip=False,
        config=CONFIG,
        modality=("depth",),
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.config = config
        self.subsets = np.asarray(self.config["split"][split])
        self.shape = tuple(shape)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.flip = flip
        assert "depth" in modality, '"depth" is required'
        self.modality = modality
        self.datalist = None
        self.load_datalist()

    def load_datalist(self):
        datalist = []
        for subset in self.subsets:
            # e.g. class0_set001_scan00001.npy
            filename = "*_set{}_*.npy".format(str(subset).zfill(3))
            point_paths = sorted(glob(osp.join(self.root, filename)))
            datalist += list(point_paths)
        self.datalist = datalist

    def preprocess(self, out):
        out["depth"] = np.linalg.norm(out["xyz"], ord=2, axis=2)
        mask = (
            (out["depth"] > 0.0)
            & (out["depth"] > self.min_depth)
            & (out["depth"] < self.max_depth)
        )
        out["depth"] -= self.min_depth
        out["depth"] /= self.max_depth - self.min_depth
        out["mask"] = mask
        out["xyz"] /= self.max_depth  # unit space
        for key in out.keys():
            out[key][~mask] = 0
        return out

    def transform(self, out):
        flip = self.flip and random.random() > 0.5
        for k, v in out.items():
            v = TF.to_tensor(v)
            if flip:
                v = TF.hflip(v)
            v = TF.resize(v, self.shape, Image.NEAREST)
            out[k] = v
        return out

    def __getitem__(self, index):
        points_path = self.datalist[index]
        points = np.load(points_path).astype(np.float32)
        out = {}
        out["xyz"] = points[..., :3]
        if "reflectance" in self.modality:
            out["reflectance"] = points[..., [3]]
        out = self.preprocess(out)
        out = self.transform(out)
        return out

    def __len__(self):
        return len(self.datalist)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Root location: {}".format(self.root))
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)

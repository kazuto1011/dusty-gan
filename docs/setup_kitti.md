# Setup KITTI Odometry

1. Download [KITTI Odometry dataset (80GB)](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

```s
└── dataset/
    └── sequences/
        ├── 00/
        │   ├── velodyne/
        │   │   ├── 000000.bin
        │   │   └── ...
        │   ├── calib.txt
        │   ├── poses.txt
        │   └── times.txt
        └── ... 
```

2. Run the following command to generate 2D projection data.

```sh
$ python process_kitti.py --source-dir /path/to/your/directory
```

```s
├── dataset/
│   └── sequences/
│       ├── 00/
│       │   ├── velodyne/
│       │   │   ├── 000000.bin
│       │   │   └── ...
│       │   ├── calib.txt
│       │   ├── poses.txt
│       │   └── times.txt
│       └── ...
└── dusty-gan/ # NEW!
    ├── sequences/
    │   ├── 00/
    │   │   └── velodyne/
    │   │       ├── 000000.npy
    │   │       └── ...
    │   └── ...
    └── angles.pt
```

3. Set the path to `configs/dataset/kitti_odometry.yaml`.

```yaml
dataset:
  root: ......./dusty-gan
```
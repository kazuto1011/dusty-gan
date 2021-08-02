# Setup KITTI

1. Download [KITTI Odometry dataset (80GB)](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

```s
KITTI
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
$ python process_kitti.py --root-dir <path to "KITTI">
```

```s
KITTI
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

3. Make a simlink to the dataset.

```sh
$ ln -s <path to "KITTI/dusty-gan"> data/kitti_odometry
```
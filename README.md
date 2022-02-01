# Learning to Drop Points for LiDAR Scan Synthesis (IROS 2021)

This repository provides the official PyTorch implementation of the following paper:

**[Learning to Drop Points for LiDAR Scan Synthesis](https://arxiv.org/abs/2102.11952)**<br>
[Kazuto Nakashima](https://kazuto1011.github.io/) and [Ryo Kurazume](https://robotics.ait.kyushu-u.ac.jp/kurazume/en/)<br>
In IROS 2021<br>

**Overview:** We propose a noise-aware GAN for generative modeling of 3D LiDAR data on a projected 2D representation (aka spherical projection). Although the 2D representation has been adopted in many LiDAR processing tasks, generative modeling is non-trivial due to the discrete dropout noises caused by LiDAR’s lossy measurement. Our GAN can effectively learn the LiDAR data by representing such discrete data distribution as a composite of two modalities: an underlying complete depth and the corresponding reflective uncertainty.

![](docs/model.svg)

**Generation:** The learned depth map (left), the measurability map for simulating the point-drops (center), and the final depth map (right).

<img src='https://user-images.githubusercontent.com/9032347/127803595-6e378bab-f709-4476-a528-73460a503e76.gif' width='100%' style='image-rendering: pixelated;'>

**Reconstruction:** The trained generator can be used as a scene prior to restore LiDAR data.

<img src='docs/example.svg' width='100%' style='image-rendering: pixelated;'>

## Requirements

To setup with anaconda:

```sh
$ conda env create -f environment.yaml
$ conda activate dusty-gan
```

## Datasets

Please follow [the instruction for KITTI Odometry](datasets/README.md). This step produces the azimuth/elevation coordinates (`angles.pt`) aligned 
with the image grid, which are required for all the following steps.

| ![kitti_scene_1](https://user-images.githubusercontent.com/9032347/151912330-992a53fc-848e-42dd-a250-c3cad1834eae.gif) | ![kitti_scene_2](https://user-images.githubusercontent.com/9032347/151912335-695a8b1a-40d2-4674-803f-773de641a5a8.gif) |
| :--------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------: |

## Pretrained Models

|    Dataset     |      LiDAR       |  Method  |                                               Weight                                               |                                          Configuration                                          |
| :------------: | :--------------: | :------: | :------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------: |
| KITTI Odometry | Velodyne HDL-64E | Baseline | [Download](https://github.com/kazuto1011/dusty-gan/releases/download/v1.0/baseline_0025000000.pth) | [Download](https://github.com/kazuto1011/dusty-gan/releases/download/v1.0/baseline_config.yaml) |
|                |                  | DUSty-I  |  [Download](https://github.com/kazuto1011/dusty-gan/releases/download/v1.0/dusty1_0025000000.pth)  |  [Download](https://github.com/kazuto1011/dusty-gan/releases/download/v1.0/dusty1_config.yaml)  |
|                |                  | DUSty-II |  [Download](https://github.com/kazuto1011/dusty-gan/releases/download/v1.0/dusty2_0025000000.pth)  |  [Download](https://github.com/kazuto1011/dusty-gan/releases/download/v1.0/dusty2_config.yaml)  |

## Training

Please use `train.py` with `dataset=`, `solver=`, and `model=` options.
The default configurations are defined under `configs/` and can be overridden via a console ([reference](https://hydra.cc/docs/advanced/override_grammar/basic)).

```sh
$ python train.py dataset=kitti_odometry solver=nsgan_eqlr model=dcgan_eqlr        # baseline
$ python train.py dataset=kitti_odometry solver=nsgan_eqlr model=dusty1_dcgan_eqlr # DUSty-I (ours)
$ python train.py dataset=kitti_odometry solver=nsgan_eqlr model=dusty2_dcgan_eqlr # DUSty-II (ours)
```

Each run creates a unique directory with saved weights and log file.

```sh
outputs/logs
└── dataset=<DATASET>
    └── model=<MODEL>
        └── solver=<SOLVER>
            └── <DATE>
                └── <TIME>
                    ├── .hydra
                    │   └── config.yaml # all configuration parameters
                    ├── models
                    │   ├── ...
                    │   └── checkpoint_0025000000.pth # trained weights
                    └── runs
                        └── <TENSORBOARD FILES> # logged losses, scores, and images
```

To monitor losses, scores, and images, run the following command to launch TensorBoard.

```sh
$ tensorboard --logdir outputs/logs
```

## Evaluation

To evaluate synthesis performance, run:

```sh
$ python evaluate_synthesis.py --model-path $MODEL_PATH --config-path $CONFIG_PATH
```

`MODEL_PATH` and `CONFIG_PATH` are `.pth` file and the corresponding `.hydra/config.yaml` file, respectively.
A relative tolerance to detect the point-drop can be changed via `--tol` option (default `0`).

To evaluate reconstruction performance, run:

```sh
$ python evaluate_reconstruction.py --model-path $MODEL_PATH --config-path $CONFIG_PATH
```

To optimize the relative tolerance for the baseline in our paper, run:

```sh
$ python tune_tolerance.py --model-path $MODEL_PATH --config-path $CONFIG_PATH
```

We used `0.008` for KITTI.

## Demo

```sh
$ streamlit run demo.py $MODEL_PATH $CONFIG_PATH
```

|               Synthesis                |             Reconstruction             |
| :------------------------------------: | :------------------------------------: |
| ![](docs/dusty-gan_demo_synthesis.png) | ![](docs/dusty-gan_demo_inversion.png) |

## Citation

If you find this code helpful, please cite our paper:

```bibtex
@inproceedings{nakashima2021learning,
    title     = {Learning to Drop Points for LiDAR Scan Synthesis},
    author    = {Nakashima, Kazuto and Kurazume, Ryo},
    booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    pages     = {222--229},
    year      = 2021
}
```
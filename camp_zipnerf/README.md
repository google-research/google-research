# CamP Zip-NeRF: A Code Release for CamP and Zip-NeRF

*This is not an officially supported Google product.*

This repository contains [JAX](https://github.com/jax-ml/jax) code for two papers: [Zip-NeRF](https://jonbarron.info/zipnerf/) and [CamP](https://camp-nerf.github.io/).  This is research code, and should be treated accordingly.

## Setup

```
# Clone the repo.
git clone https://github.com/google-research/google-research.git
cd google_research/camp_zipnerf

# Make a conda environment.
conda create --name camp_zipnerf python=3.11
conda activate camp_zipnerf

# Prepare pip.
conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap

# Confirm that all the unit tests pass.
./scripts/run_all_unit_tests.sh
```

You'll probably also need to [update your JAX installation to support GPUs or TPUs](https://jax.readthedocs.io/en/latest/installation.html).

## Running

This code is designed to run on the [NeRF Blender](https://www.matthewtancik.com/nerf) and [Mip-NeRF 360 Datasets](https://jonbarron.info/mipnerf360/). If you want to onboard a new dataset, check out the instructions for the [MultiNeRF code release](https://github.com/google-research/multinerf), they should roughly apply here.

### Running Zip-NeRF

To reproduce all of the Zip-NeRF results on the two datasets, just run`scripts/zipnerf/all.sh` --- be prepared to wait a long time! Unless you are at barron's workstation, you'll need to change some paths in the scripts that point to data folders. This script should recover a Zip-NeRF for all scenes, render out all test set images, render a few nice videos for the 360 dataset, and generate latex for tables. Because the paper's results were on 8 V100s and barron only has a GeForce RTX 3080 in his workstation, we dropped the batch size and learning rate by 8x while increasing the number of iterations by 8x (as per the linear scaling rule). Results are comparable to the paper (see the Zip-NeRF Results section below).

Note that we have results on the "single scale" mip-NeRF 360 Dataset (aka,"the mip-NeRF 360 dataset", see `scripts/zipnerf/360_*.sh`), as well as the "multi-scale" variant that Zip-NeRF presented in order to evaluate performance on multi-scale scenes, or scenes where the camera may be very far away (see `scripts/zipnerf/ms360_*.sh`). Unless you're explicitly interested in this multi-scale setting, you probably just want to use the "360_*.sh" scripts --- especially since they're the only ones that make nice videos.

Also note that the config we used in the Zip-NeRF paper to produce still images
for evaluating error metrics (`configs/zipnerf/360.gin`) is different from the
config we used to produce videos (`configs/zipnerf/360_aglo128.gin`). The latter
includes the "affine GLO" per-image embedding that accounts for changes in
exposure and white balance across images (see the supplement). Using this
reduces floaters in the video renderings significantly, but also decreases
test-set performance on these benchmarks. If you want maximally-accurate still
images, use `360.gin`, but if you want pretty videos, use `360_aglo128.gin`.

### Running CamP

There are multiple configs for CamP depending on the type of dataset (360 scene
vs. object) and level of uncertainty (refining COLMAP poses vs. high noise).

*   `camera_optim.gin` contains the base CamP configuration.
*   `camera_optim_perturbed.gin` contains the CamP configuration for the
    perturbed 360 scene experiments. This will configure the data loader to add
    a significant amount of noise to the training camera parameters, and the
    training process will have to correct for this noise during scene fitting.
    Compared to be base config, the coarse-to-fine schedule is more aggressive
    to avoid falling into local minima.
*   `camera_optim_perturbed_blender.gin` contains the CamP configuration for the
    perturbed Blender scene experiments.
*   `camera_optim_arkit.gin` contains the CamP configuration for the ARKit
    experiments (note that these datasets have not been released at this time).

You can mix and match these configurations with the base NeRF configs (e.g.,
`360.gin`). For example, to add camera refinement to Zip-NeRF, run the training
pipeline like this:

```bash
python -m train \
    --gin_configs=configs/zipnerf/360.gin \
    --gin_configs=configs/camp/camera_optim.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
    --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}/${SCENE}'"
```

A script is available in `scripts/camp/360_train.sh` that runs Zip-NeRF + CamP
on all of the Mip-NeRF 360 scenes.

To run different camera parameterizations, modify `Config.camera_delta_cls` to
one of the following:

*   `@camera_delta.FocalPoseCameraDelta`: The Focal Pose parameterization.
*   `@camera_delta.IntrinsicFocalPoseCameraDelta`: The Focal Pose
    parameterization with radial distortion refinement.
*   `@camera_delta.SE3CameraDelta`: The SE3 parameterization, similar to BARF.
*   `@camera_delta.SE3WithFocalCameraDelta`: The SE3 parameterization with focal
    length refinement.
*   `@camera_delta.IntrinsicSE3WithFocalCameraDelta`: The SE3 parameterization
    with radial distortion and focal length refinement.
*   `@camera_delta.SCNeRFCameraDelta`: The SCNeRF parameterization.

    You can set these by modifying the config files, or by setting e.g.,
    `--gin_bindings="Config.config = '@camera_delta.FocalPoseCameraDelta'"` in
    the launch command.

## Zip-NeRF Results

For completeness we re-ran all results from the Zip-NeRF paper using this code repo, locally on barron's workstation. We've also linked to zip files of each rendered test-set, to allow others to evaluate their own error metrics.

---

### Single-Scale mip-NeRF 360 Dataset Results:

#### Average:
| | PSNR | SSIM |
|-|-|-|
|Code Release | 28.56 | 0.828 |
|Paper | 28.54 | 0.828 |

#### Per-Scene
|| | bicycle | flowers | garden | stump| treehill | room | counter | kitchen | bonsai |
|-|-|-|-|-|-|-|-|-|-|-|
| PSNR | Code Release | 25.85 | 22.33 | 28.22 | 27.35 | 23.95 | 33.04 | 29.12 | 32.36 | 34.79 |
|| Paper | 25.80 | 22.40 | 28.20 | 27.55 | 23.89 | 32.65 | 29.38 | 32.50 | 34.46 |
| SSIM | Code Release | 0.772 | 0.637 | 0.863 | 0.788 | 0.674 | 0.929 | 0.905 | 0.929 | 0.952 |
|| Paper | 0.769 | 0.642 | 0.860 | 0.800 | 0.681 | 0.925 | 0.902 | 0.928 | 0.949 |

[Download Results](http://storage.googleapis.com/gresearch/refraw360/zipnerf_results_360.zip)

---

### Multi-Scale mip-NeRF 360 Dataset Results (Average):
|| PSNR 1x | PSNR 2x | PSNR 4x | PSNR 8x | SSIM 1x | SSIM 2x | SSIM 4x | SSIM 8x |
|-|-|-|-|-|-|-|-|-|
Code Release | 28.15 | 29.90 | 31.43 | 32.38 | 0.822 | 0.890 | 0.931 | 0.952 |
Paper | 28.25 | 30.00 | 31.57 | 32.52 | 0.822 | 0.892 | 0.933 | 0.954 |

[Download Results](http://storage.googleapis.com/gresearch/refraw360/zipnerf_results_ms360.zip)

---

### Single-Scale Per-Scene Blender Dataset Results:

||| chair | drums | ficus | hotdog | lego | materials | mic | ship |
|-|-|-|-|-|-|-|-|-|-|
|PSNR | Code Release | 35.78 | 25.91 | 34.72 | 38.05 | 35.79 | 31.05 | 35.92 | 32.33 |
| | Paper | 34.84 | 25.84 | 33.90 | 37.14 | 34.84 | 31.66 | 35.15 | 31.38 |
|SSIM | Code Release | 0.987 | 0.948 | 0.987 | 0.987 | 0.983 | 0.968 | 0.992 | 0.937 |
| | Paper | 0.983 | 0.944 | 0.985 | 0.984 | 0.980 | 0.969 | 0.991 | 0.929 |

[Download Results](http://storage.googleapis.com/gresearch/refraw360/zipnerf_results_blender.zip)

--------------------------------------------------------------------------------

## Citation

If you use this software package, please cite whichever constituent paper(s) you build upon, or feel free to cite this entire codebase as:
```
@misc{campzipnerf2024,
  title={{CamP Zip-NeRF}: {A} {Code} {Release} for {CamP} and {Zip-NeRF}},
  author={Jonathan T. Barron and Keunhong Park and Ben Mildenhall and John Flynn and Dor Verbin and Pratul Srinivasan and Peter Hedman and Philipp Henzler and Ricardo Martin-Brualla}
  year={2024},
  url={https://github.com/google-research/google-research/tree/master/camp_zipnerf},
}
# Omnimatte3D: Associating Objects and their Effects in Unconstrained Monocular Video

This is a JAX/Flax implementation of our CVPR 2023 paper "Omnimatte3D: Associating Objects and their Effects in Unconstrained Monocular Video"

## Installation
The following code snippet clones the repo and installs dependencies.

```python
ENV_DIR=~/venvs/gpnr

# Clone this repo.
sudo apt install subversion
svn export --force
https://github.com/google-research/google-research/trunk/omnimatte3D

# Setup virtualenv.
python3 -m venv $ENV_DIR
source $ENV_DIR/bin/activate

# Install dependencies.
pip install -r omnimatte3D/requirements.txt
# For training with GPUs, you might have to change this to your
# system's CUDA version. Please check the ouput of nvcc --version and change the
# version accordingly.
pip install --upgrade "jax[cuda]" -f
https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Dataset
Please download the DAVIS dataset from their [webpage](https://davischallenge.org/).
Depths for the scenes in davis dataset can be dowloaded from [here](https://storage.googleapis.com/omnimatte3d/davis_casual_slam/casual_slam_all.zip). These depth were obtained using
[Casual-SAM](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930020.pdf).

## Training

To run the DAVIS experiments, use the script below.

```python
python -m omnimatte3D.main \
  --workdir=/tmp/train_run \
  --is_train=True \
  --config.dataset.name=davis \
  --config.dataset.basedir=/path/to/datset/ \
  --config.dataset.scene=scene_name \
  --config.dataset.batch_size=4 \
  --config.model.name=ldi \
  --config.train.log_loss_every_steps=2000 \
  --config.train.checkpoint_every_steps=1000 \
  --config.train.max_steps=30000 \
  --config.train.switch_steps=100 \
  --config.train.scheduler=cosine \
  --config.train.lr_init=5.e-4 \
  --config.train.weight_decay=0.0000 \
  --config.train.crop_projection=False \
  --config.loss.disp_layer_alpha=1.0 \
  --config.loss.disp_smooth_alpha=0.5 \
  --config.loss.src_rgb_recon_alpha=1.0 \
  --config.loss.proj_far_rgb_alpha=1.0 \
  --config.loss.fg_mask_alpha=0.01 \
  --config.loss.shadow_smooth_alpha=0.01 \
  --config.loss.fg_alpha_reg_l0_alpha=0.001 \
  --config.loss.fg_alpha_reg_l1_alpha=0.0005
```

To run the evaluation code that saves the prediction use `--is_train=False`.
To render video for the layers use `--is_render=True`.

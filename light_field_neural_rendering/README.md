# Light Field Neural Rendering
### [Project Page](https://light-field-neural-rendering.github.io/) | [Paper](https://arxiv.org/pdf/2112.09687.pdf) | [Video](Coming Soon)

This is a JAX/Flax implementation of the paper Suhail et al, "Light Field Neural
Rendering".

## Installation
The following code snippet clones the repo and installs dependencies.

```python
ENV_DIR=~/venvs/lfnr

# Clone this repo.
sudo apt install subversion
svn export --force https://github.com/google-research/google-research/trunk/light_field_neural_rendering

# Setup virtualenv.
python3 -m venv $ENV_DIR
source $ENV_DIR/bin/activate

# Install dependencies.
pip install -r light_field_neural_rendering/requirements.txt
# For training with GPUs, you might have to change this to your
# system's CUDA version. Please check the ouput of nvcc --version and change the
# version accordingly.
pip install --upgrade "jax[cuda110]==0.2.19" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Dataset

You will have to download the dataset from [NeX official Onedrive](https://vistec-my.sharepoint.com/personal/pakkapon_p_s19_vistec_ac_th/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpakkapon%5Fp%5Fs19%5Fvistec%5Fac%5Fth%2FDocuments%2Fpublic%2FVLL%2FNeX%2Fshiny%5Fdatasets%2Fshiny) or [Nex offical Google Drive](https://drive.google.com/corp/drive/folders/1kYGyIJI6AduHC-bM312N41WPjAoYf8Um).

Depending on where you place the dataset you will have to change the `dataset.base_dir` config variable in your experiments.

## Demo Run on Toy Task
To test the working of the installations you can run a toy experiment on the
`Crest` scene from Shiny downsized by a factor of 16 from the original
resolution. The script below will take around 3 hours to train on a single V100
GPU.
```python
python -m light_field_neural_rendering.main \
  --workdir=/tmp/toy_task \
  --is_train=True \
  --ml_config=light_field_neural_rendering/configs/defaults.py \
  --ml_config.dataset.base_dir=/path/to/you/dataset/dirctory/with/scenes \
  --ml_config.dataset.scene=name_of_scene \
  --ml_config.dataset.batch_size=32 \
  --ml_config.dataset.factor=16 \
  --ml_config.model.num_projections=64 \
  --ml_config.train.lr_init=3.0e-5 \
  --ml_config.train.warmup_steps=5000 \
  --ml_config.train.render_every_steps=500000 \
  --ml_config.model.transformer_layers=4 \
  --ml_config.model.conv_feature_dim=\(8,\) \
  --ml_config.eval.chunk=4096 \
  --ml_config.model.ksize1=5
```
Once training is done, to evaluate run the script below.
```python
python -m light_field_neural_rendering.main \
  --workdir=/tmp/toy_task \
  --is_train=False \
  --ml_config=light_field_neural_rendering/configs/defaults.py \
  --ml_config.dataset.base_dir=/path/to/you/dataset/dirctory/with/scenes \
  --ml_config.dataset.scene=name_of_scene \
  --ml_config.dataset.factor=16 \
  --ml_config.dataset.batch_size=32 \
  --ml_config.model.num_projections=64 \
  --ml_config.train.lr_init=3.0e-5 \
  --ml_config.train.warmup_steps=5000 \
  --ml_config.model.transformer_layers=4 \
  --ml_config.model.conv_feature_dim=\(8,\) \
  --ml_config.eval.chunk=4096 \
  --ml_config.model.ksize1=5 \
  --ml_config.eval.eval_once=True
```
You should obtain a PSNR of ~27dB. 

## Training
To reproduce the results in the paper you will need to run the following script.
```python
python -m light_field_neural_rendering.main \
  --workdir=/tmp/train_run \
  --is_train=True \
  --ml_config=light_field_neural_rendering/configs/defaults.py \
  --ml_config.dataset.base_dir=/path/to/you/dataset/dirctory/with/scenes \
  --ml_config.dataset.scene=name_of_scene \
  --ml_config.dataset.name=ff_epipolar  \
  --ml_config.dataset.batch_size=4096  \
  --ml_config.model.num_projections=191 \
  --ml_config.train.lr_init=3.0e-4 \
  --ml_config.train.warmup_steps=5000 \
  --ml_config.train.render_every_steps=500000 \
  --ml_config.model.transformer_layers=8 \
  --ml_config.eval.chunk=4096 \
  --ml_config.model.ksize1=5 \
  --ml_config.model.init_final_precision=HIGHEST
```
Note that according to the number of devices available you will need to adjust
the `batch_size`. When changing `batch_size` please scale the `lr_init`
accordingly. We suggest a linear scaling i.e. if you halve the batch size, halve
the learning rate and double the `max_steps`.

Similarly, evaluation can be done by running the script below.
```python
python -m light_field_neural_rendering.main \
  --workdir=/tmp/train_run \
  --is_train=False \
  --ml_config=light_field_neural_rendering/configs/defaults.py \
  --ml_config.dataset.base_dir=/path/to/you/dataset/dirctory/with/scenes \
  --ml_config.dataset.scene=name_of_scene \
  --ml_config.dataset.name=ff_epipolar  \
  --ml_config.dataset.batch_size=4096  \
  --ml_config.model.num_projections=191 \
  --ml_config.train.lr_init=3.0e-4 \
  --ml_config.train.warmup_steps=5000 \
  --ml_config.train.render_every_steps=500000 \
  --ml_config.model.transformer_layers=8 \
  --ml_config.eval.chunk=4096 \
  --ml_config.model.ksize1=5 \
  --ml_config.model.init_final_precision=HIGHEST \
  --ml_config.eval.eval_once=True
```

**Note:** For the CD and the Lab scene you will have to also set the following 
variables in the config.
```python
  --ml_config.dataset.factor=-1 \
  --ml_config.dataset.image_height=567 \
```

To render a video just run the evaluation script with
`--ml_config.dataset.render_path=True`.

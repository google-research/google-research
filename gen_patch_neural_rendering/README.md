# Generalizable Patch-Based Neural Rendering

This is a JAX/Flax implementation of our ECCV-2022 oral paper "Generalizable Patch-Based Neural Rendering".
### [Project Page](https://mohammedsuhail.net/gen_patch_neural_rendering/) | [Paper](https://arxiv.org/abs/2207.10662)

## Installation
The following code snippet clones the repo and installs dependencies.

```python
ENV_DIR=~/venvs/gpnr

# Clone this repo.
sudo apt install subversion
svn export --force https://github.com/google-research/google-research/trunk/gen_patch_neural_rendering

# Setup virtualenv.
python3 -m venv $ENV_DIR
source $ENV_DIR/bin/activate

# Install dependencies.
pip install -r gen_patch_neural_rendering/requirements.txt
# For training with GPUs, you might have to change this to your
# system's CUDA version. Please check the ouput of nvcc --version and change the
# version accordingly.
pip install --upgrade "jax[cuda110]==0.2.19" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Dataset

Download the IBRNet scenes from [here](https://drive.google.com/file/d/1rkzl3ecL3H0Xxf5WTyc2Swv30RIyr1R_/view) and [here](https://drive.google.com/file/d/1Uxw0neyiIn3Ve8mpRsO6A06KfbqNrWuq/view).
Move all the scenes into a single directory.
Depending on where you place the dataset you will have to change the `dataset.ff_base_dir` config variable in your experiments.

## Training
To reproduce the results in the paper you will need to run the following script.
```python
python -m gen_patch_neural_rendering.main \
  --workdir=/tmp/train_run \
  --is_train=True \
  --ml_config=gen_patch_neural_rendering/configs/defaults.py \
  --ml_config.dataset.ff_base_dir=/path/to/you/dataset/dirctory/with/scenes \
  --ml_config.dataset.name=ff_epipolar  \
  --ml_config.dataset.batch_size=4096  \
  --ml_config.lightfield.max_deg_point=4 \
  --ml_config.train.lr_init=3.0e-4 \
  --ml_config.train.warmup_steps=5000 \
  --ml_config.train.render_every_steps=500000 \
  --ml_config.dataset.normalize=True \
  --ml_config.model.init_final_precision=HIGHEST
```
Note that according to the number of devices available you will need to adjust
the `batch_size`. When changing `batch_size` please scale the `lr_init`
accordingly. We suggest a linear scaling i.e. if you halve the batch size, halve
the learning rate and double the `max_steps`.

Similarly, evaluation can be done by running the script below.
```python
python -m gen_patch_neural_rendering.main \
  --workdir=/tmp/train_run \
  --is_train=False \
  --ml_config=gen_patch_neural_rendering/configs/defaults.py \
  --ml_config.dataset.eval_llff_dir=/path/to/you/dataset/dirctory/with/scenes \
  --ml_config.dataset.eval_dataset=llff  \
  --ml_config.dataset.batch_size=4096  \
  --ml_config.lightfield.max_deg_point=4 \
  --ml_config.train.lr_init=3.0e-4 \
  --ml_config.train.warmup_steps=5000 \
  --ml_config.train.render_every_steps=500000 \
  --ml_config.eval.chunk=4096 \
  --ml_config.dataset.normalize=True \
  --ml_config.model.init_final_precision=HIGHEST \
  --ml_config.eval.eval_once=True
```

For LLFF scene please set `dataset.eval_llff_dir` appropriately.
For shiny scenes please set `dataset.eval_dataset=shiny-6` and `dataset.eval_ff_dir` appropriately.

To render a video just run the evaluation script with
`--ml_config.dataset.render_path=True`.

## Citation
```
@inproceedings{suhail2022generalizable,
  title={Generalizable Patch-Based Neural Rendering},
  author={Suhail, Mohammed and Esteves, Carlos and Sigal, Leonid and Makadia, Ameesh},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}}
```

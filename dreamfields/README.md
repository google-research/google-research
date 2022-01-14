# Zero-Shot Text-Guided Object Generation with Dream Fields
by **Ajay Jain**, **Ben Mildenhall**, **Jonathan T. Barron**, **Pieter Abbeel**, and **Ben Poole**

[Project website](https://ajayj.com/dreamfields) | [arXiv paper](https://arxiv.org/abs/2112.01455)


[![Watch the video](https://img.youtube.com/vi/1Fke6w46tv4/hqdefault.jpg)](https://www.youtube.com/watch?v=1Fke6w46tv4)

## Overview
This code implements Dream Fields, a way to synthesize 3D objects from natural language prompts.

**Abstract:** We combine neural rendering with multi-modal image and text representations to synthesize diverse 3D objects solely from natural language descriptions. Our method, Dream Fields, can generate the geometry and color of a wide range of objects without 3D supervision. Due to the scarcity of diverse, captioned 3D data, prior methods only generate objects from a handful of categories, such as ShapeNet. Instead, we guide generation with image-text models pre-trained on large datasets of captioned images from the web. Our method optimizes a Neural Radiance Field from many camera views so that rendered images score highly with a target caption according to a pre-trained CLIP model. To improve fidelity and visual quality, we introduce simple geometric priors, including sparsity-inducing transmittance regularization, scene bounds, and new MLP architectures. In experiments, Dream Fields produce realistic, multi-view consistent object geometry and color from a variety of natural language captions.

## Running with Docker
We provide a Dockerfile based on a NVIDIA NGC container. Pull the base container:
```
docker pull nvcr.io/nvidia/tensorflow:21.11-tf2-py3
```
Run at low quality on all GPUs:
```
bash run_docker.sh all "matte painting of a bonsai tree; trending on artstation."
```
Videos will be written to `results/`. To specify a subset of GPUs, use:
```
bash run_docker.sh '"device=0,1,2,3"' "matte painting of a bonsai tree; trending on artstation."
```

To monitor training, run Tensorboard with:
```
bash run_tensorboard.sh 6006
```

We provide three configuration files. `config/config_lq.py` provides low quality, faster (~30 minute) text-to-3D synthesis. `config/config_mq.py` and `config/config_hq.py` provide higher quality by rendering higher resolutions and using more augmentations during training. For low quality results, 4 16GB GPUs should be enough. Modify `run_docker.sh` to use these configs if sufficient resources are available. If you run out of memory, lower `render_width` and `crop_width` or `n_local_aug` in `config/config_lq.py`.

## Running in a virtual environment

Python 3 is required. Create and activate a virtual environment:
```
python -m venv env
source env/bin/activate
```

Install JAX with GPU or TPU support following [the JAX docs](https://github.com/google/jax#installation), depending on the accelerator you have available.
For example, for CUDA 11.1:
```
pip install --upgrade pip
pip install --upgrade jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
Test your installation with:
```
python -c "print(__import__('jax').local_devices())"
```
Then, install dependencies:
```
pip install -r requirements.txt
```
To run on all visible GPUs:
```
python run.py --config=config/config_lq.py --query="bouquet of flowers sitting in a clear glass vase."
```

## Citation
Please cite our paper if you find this code or research relevant:
```
@article{jain2021dreamfields,
  author = {Jain, Ajay and Mildenhall, Ben and Barron, Jonathan T. and Abbeel, Pieter and Poole, Ben},
  title = {Zero-Shot Text-Guided Object Generation with Dream Fields},
  journal = {arXiv},
  month = {December},
  year = {2021},
}
```


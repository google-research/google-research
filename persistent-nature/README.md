# Persistent Nature
[Project Page](https://chail.github.io/persistent-nature) | [Paper](http://arxiv.org/abs/2303.13515) | [Bibtex](#citation)

Persistent Nature: A Generative Model of Unbounded 3D Worlds.\
_CVPR 2023_ \
[Lucy Chai](http://people.csail.mit.edu/lrchai/), [Richard Tucker](https://research.google/people/RichardTucker/), [Zhengqi Li](https://zhengqili.github.io/), [Phillip Isola](http://web.mit.edu/phillipi/), [Noah Snavely](https://www.cs.cornell.edu/~snavely/)

*Please note that this is not an officially supported Google product.*

## Prerequisites
- Linux
- gcc-7
- Python 3
- NVIDIA GPU + CUDA CuDNN

**Table of Contents:**
1. [Colab](#colab) - run it in your browser without installing anything locally
2. [Setup](#setup) - download pretrained models and resources
3. [Pretrained Models](#pretrained) - quickstart with pretrained models
4. [Interactive Notebooks](#notebooks) - jupyter notebooks for interactive navigation
5. [Videos](#videos) - export flying videos

<br>

<p float="left">
<img src='img/0017.gif'><img src='img/0005.gif'><img src='img/0009_triplane.gif'>
</p>

<a name="colab"/>

## Colab

Interactive Demo: Try our interactive demo here! Does not require local installations.
- [Layout Model](https://colab.research.google.com/github/google-research/google-research/blob/master/persistent-nature/interactive-layout-colab.ipynb)
- [Triplane Model](https://colab.research.google.com/github/google-research/google-research/blob/master/persistent-nature/interactive-triplane-colab.ipynb)

<a name="setup"/>

## Setup

- Clone this repo ([reference](https://stackoverflow.com/questions/600079/how-do-i-clone-a-subdirectory-only-of-a-git-repository/52269934#52269934)):
```bash
git clone --depth 1 --filter=blob:none --sparse \
         https://github.com/google-research/google-research.git
cd google-research
git sparse-checkout set persistent-nature
```

- Install dependencies:
	- gcc-7 or above is required for installation. Update gcc following [these steps](https://gist.github.com/jlblancoc/99521194aba975286c80f93e47966dc5).
	- We provide a Conda `environment.yml` file listing the dependencies. You can create a Conda environment with the dependencies using:
```bash
conda env create -f environment.yml
```

- Apply patch files: we provide a script for downloading associated resources applying the patch files
```bash
bash patch.sh
```

- Download the pretrained models:
```
bash download.sh
```

<a name="pretrained"/>

## Quickstart with pretrained models

See the notebook `basic.ipynb` for an example of the most basic usage. Examples for layout and triplane models are included. 

The layout model is the model used in the main paper, while the triplane model is in the supplementary --  the layout model is better at FID, but the triplane model is more consistent and faster to render.

Note: remember to add the conda environment to jupyter kernels:
```bash
python -m ipykernel install --user --name persistentnature
```
You may need to also enable widgets in notebook extensions for the next section:
```bash
jupyter nbextension enable --py widgetsnbextension
```

<a name="notebooks"/>

## Notebooks

We provide example notebooks `interactive-layout.ipynb` and `interactive-triplane.ipynb` for running inference on pretrained models (both layout and triplane models)
<a name="videos"/>

## Videos

The script `video.sh` demonstrates how to generate a figure 8 flight pattern with autopilot height adjustment on both models

## Training

1. Data Preprocessing: see the script `preprocess.sh` 
2. Model Training: See the scripts `train_layout.sh` and `train_triplane.sh` 

### Acknowledgements

This code is adapted from [Stylegan3](https://github.com/NVlabs/stylegan3) ([license](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt)), [GSN](https://github.com/apple/ml-gsn) ([license](https://github.com/apple/ml-gsn/blob/main/LICENSE)), [EG3D](https://github.com/NVlabs/eg3d) ([license](https://github.com/NVlabs/eg3d/blob/main/LICENSE.txt)). Remaining changes are covered under [Apache License v2](https://choosealicense.com/licenses/apache-2.0/)

### Contact

For any questions related to our paper,
please email lrchai@mit.edu.

<a name="citation"/>

### Citation
If you use this code for your research, please cite our paper:
```
@inproceedings{chai2023persistentnature,
    title={Persistent Nature: A Generative Model of Unbounded 3D Worlds},
    author={Chai, Lucy and Tucker, Richard and Li, Zhengqi and Isola, Phillip and Snavely, Noah},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023}
}
```

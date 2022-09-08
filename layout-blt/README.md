# Bidirectional Layout Transformer (BLT)

This repository contains the code and models for the following paper:

**[BLT: Bidirectional Layout Transformer for Controllable Layout Generation](https://arxiv.org/abs/2112.05112)** In ECCV'22.


If you find this code useful in your research then please cite

```
@article{kong2021blt,
  title={BLT: Bidirectional Layout Transformer for Controllable Layout Generation},
  author={Kong, Xiang and Jiang, Lu and Chang, Huiwen and Zhang, Han and Hao, Yuan and Gong, Haifeng and Essa, Irfan},
  journal={arXiv preprint arXiv:2112.05112},
  year={2021}
}
```

*Please note that this is not an officially supported Google product.*


# Introduction

Automatic generation of such layouts is important as we seek scale-able and diverse visual designs. We introduce BLT, a bidirectional layout transformer. BLT differs from autoregressive decoding as it first generates a draft layout that satisfies the user inputs and then refines the layout iteratively.


## Set up environment

```
conda env create -f environment.yml
conda activate layout
```
or
```
pip install -r requirement.txt
```

## Datasets

Please download the public datasets at the following webpages and prepare to the JSON format.

1. [COCO](https://cocodataset.org/)
2. [RICO](https://interactionmining.org/rico)
3. [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)
4. [Magazine](https://xtqiao.com/projects/content_aware_layout/)



## Running

```
# Training a model
python  main.py --config configs/${config} --workdir ${model_dir}
# Testing a model
python  main.py --config configs/${config} --workdir ${model_dir} --mode 'test'
```

## Pretrained models on public benchmarks

* [RICO]()
* [PubLayNet]()

# SMUG
Paper: [Scaling Symbolic Methods using Gradients for Neural Model Explanation](https://arxiv.org/abs/2006.16322)

By [Subham S. Sahoo](https://research.google/people/SubhamSekharSahoo/), [Subhashini Venugopalan](https://vsubhashini.github.io), [Li Li](https://research.google/people/LiLi/), [Rishabh Singh](https://research.google/people/RishabhSingh/), [Patrick Riley](https://research.google/people/PatrickRiley/)

# Introduction

This repository contains the implementation of the SMUG saliency method as proposed in the paper [Scaling Symbolic Methods using Gradients for Neural Model Explanation](https://arxiv.org/abs/2006.16322), **ICLR 2021**. SMUG combines gradient-based methods with symbolic techniques to generate explanations for a Neural Network's predictions.

# Dependencies
* python>=3.5
* numpy>=1.19.4
* saliency>=0.0.5
* scikit-image>=0.18.1
* scikit-learn>=0.24.1
* scipy>=1.6.0
* tensorflow>=2.4.1
* tensorflow-datasets>=4.2.0
* tensorflow-estimator>=2.4.0
* tensorflow-hub>=0.11.0
* z3-solver==4.8.8

> **_NOTE:_**  Avoid later versions of z3-solver as we observe performance issues with them.

# Usage
1. Download the repository.
  * `git clone --depth 1  --filter=blob:none  --sparse  https://github.com/google-research/google-research.git`
  * `cd google-research`
  * `git sparse-checkout init --cone`
  * `git sparse-checkout set smug_saliency`
2. Run the following commands to ensure that all the tests pass -
  * `chmod+x smug_saliency/run.sh`
  * `./smug_saliency/run.sh`
3. Notebook examples.
  * [**image_saliency.ipynb**] (image_saliency.ipynb) shows how to run SMUG on CNN models for image saliency. In the notebook we demonstrate how to load an [inception model (v1 and v3)](https://github.com/tensorflow/models/tree/master/research/slim), configure hyperparameters for SMUG, and generate saliency maps. Our paper used a slightly different variant of the inception v1 model (without batch norm), so visualizations and results may differ.

  * [**text_saliency.ipynb**] (text_saliency.ipynb) shows how to run SMUG on text models for text saliency. In the notebook we train a 1D CNN model on [beer reviews dataset](http://people.csail.mit.edu/taolei/beer/), configure hyperparameters for SMUG, and generate saliency maps.

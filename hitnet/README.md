# HITNET
This package includes the code used to run the experiments in the HITNET paper [HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching](https://arxiv.org/abs/2007.12140).

## Setup
The scripts will download the models and install prerequisites, however datasets have to be downloaded manually and scripts have to be modified to point at the download location. Currently models for 3 datasets are shared.

## FlyingThings
Two models are provided with maximum disparity set to 320: one for cleanpass, one for finalpass version of the dataset. The scripts will run in evaluation mode by default and will compute PSM EPE and other metrics.

## Middlebury
Three models are provided, with maximum disparity set to 160, 288, 400. The models are trained and designed to run on half-resolution input images.

## ETH3D
A model is provided with maximum disparity set to 128.

## Performance
The released models are using default tensoflow ops which causes them to be 3X slower than the same models that use custom CUDA ops. The custom CUDA ops and a version of the models that use them may be released later.

## Disparity Range
A version of the models with a variable disparity range may be released later.

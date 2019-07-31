# Depth from Video in the Wild: Unsupervised Monocular Depth Learning from Unknown Cameras

This repository contains a preliminary release of code for the paper bearing the
title above, at https://arxiv.org/abs/1904.04998, and to appear at ICCV 2019.
The code is based on the Struct2depth [repository]
(https://github.com/tensorflow/models/tree/master/research/struct2depth)
(see the respective paper [here](https://arxiv.org/abs/1811.06152)), 
and utilizes the same data format.

This release supports training a depth and motion prediciton model, with either
learned or specified camera intrinsics. The motion model produces 6 degrees of
freedom of camera motion, and a dense translation vector field for every pixel
in the scene. As an input, the code needs triplets of RGB frames with
possibly-moving objects masked out.

Sample command line:

```bash
python -m depth_from_video_in_the_wild.train \
   --checkpoint_dir=$MY_CHECKPOINT_DIR \
   --data_dir=$MY_DATA_DIR
```

`MY_CHECKPOINT_DIR` is where the trained model checkpoints are to be saved.

`MY_DATA_DIR` is where the training data (in Struct2depth's format) is stored.
The `data_example` folder contains a single training example expressed in this
format.


A command line for running a single training step on the single example in
`data_example` (for testing):

```bash
python -m depth_from_video_in_the_wild.train \
  --data_dir=depth_from_video_in_the_wild/data_example \
  --checkpoint_dir=/tmp/my_experiment --train_steps=1
```

To use the given intrinsics instead of learning them, add
`--nolearn_intrinsics` to the coomand.


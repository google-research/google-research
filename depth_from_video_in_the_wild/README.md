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

## Pretrained checkpoints and respective metrics
The table below provides checkpoints trained on Cityscapes, KITTI and their
mixture, with the respective Absolute Relative depth error metrics. The metrics
slightly differ from the results in Table A3 in the paper because for the latter
we averaged the metrics over multiple checkpoints, whereas in the table below
the metrics relate to a specific checkpoint. All checkpoints were harvested
after training on nearly 4M images (since the datasets are much smaller than 4M,
this of course means multiple epochs).


<center>

|Trained on |Intirinsics|Abs Rel on Cityscapes       |Abs Rel on KITTI|Checkpoint|
|:----------|:---------:|:------:|:-------:|:-------:|
|Cityscapes|Learned| 0.1279|0.1729| [download](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Fcheckpoints%2Fcityscapes_learned_intrinsics.zip?generation=1566493765410932&alt=media)|
|KITTI|Learned| 0.1679|0.1262| [download](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Fcheckpoints%2Fkitti_learned_intrinsics.zip?generation=1566493768934649&alt=media)|
|Cityscapes + KITTI | Learned | 0.1196 | 0.1231 | [download](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Fcheckpoints%2Fcityscapes_kitti_learned_intrinsics.zip?generation=1566493762028542&alt=media)

</center>

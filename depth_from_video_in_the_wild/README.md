# Depth from Video in the Wild: Unsupervised Monocular Depth Learning from Unknown Cameras
This repository contains a preliminary release of code for the paper bearing the
title above, [published at ICCV 2019](http://openaccess.thecvf.com/content_ICCV_2019/html/Gordon_Depth_From_Videos_in_the_Wild_Unsupervised_Monocular_Depth_Learning_ICCV_2019_paper.html).
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
   --data_dir=$MY_DATA_DIR \
   --imagenet_ckpt=$MY_IMAGENET_CHECKPOINT
```

`MY_CHECKPOINT_DIR` is where the trained model checkpoints are to be saved.

`MY_DATA_DIR` is where the training data (in Struct2depth's format) is stored.
The `data_example` folder contains a single training example expressed in this
format.

`MY_IMAGENET_CHECKPOINT` is a path to a pretreained ImageNet checkpoint to
intialize the encoder of the depth prediction model.

On Cityscapes we used the default batch size (4), for KITTI we used a batch
size of 16 (add `--batch_size=16` to the training command).

A command line for running a single training step on the single example in
`data_example` (for testing):

```bash
python -m depth_from_video_in_the_wild.train \
  --data_dir=depth_from_video_in_the_wild/data_example \
  --checkpoint_dir=/tmp/my_experiment --train_steps=1
```

To use the given intrinsics instead of learning them, add
`--nolearn_intrinsics` to the coomand.

## Pretrained checkpoints and respective depth metrics
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

## Pretrained checkpoints and respective odometry results
The command for generating a trajectory from a checkpoint given an odometry test
set is:

```bash
python -m depth_from_video_in_the_wild.trajectory_inference \
  --checkpoint_path=$YOUR_CHECKPOINT_PATH \
  --odometry_test_set_dir=$DIRECTORY_WHERE_YOU_STORE_THE_ODOMETRY_TEST_SET \
  --output_dir=$DIRECTORY_WHERE_THE_TRAJECTORIES_WILL_BE_SAVED \
  --alsologtostderr
```

We observed that odometry generally took longer to converge. The table below
lists the checkpoints used to evaluate odometry on in the paper. All checkpoints
were trained on KITTI. The training batch size was 16, and the learning rate and
number of training steps is given in the table.

<center>

|Intirinsics|Learning rate|Training steps|Checkpoint|Seq. 09|Seq. 10|
|:---------|:------:|:-------:|:-------:|:---:|:---:|
|Given| 3e-5|480377| [download](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Fcheckpoints%2Fcityscapes_learned_intrinsics.zip?generation=1566493765410932&alt=media)|[trajectory](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Fodometry%2Fgiven_intrinsics_trajectory_odo09.txt?generation=1568247377779913&alt=media) | [trajectory](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Fodometry%2Flearned_intrinsics_trajectory_odo10.txt?generation=1568247378745091&alt=media)
|Learned| 1e-4|413174| [download](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Fcheckpoints%2Fkitti_odometry_learned_intrinsics.zip?generation=1568245497722898&alt=media)| [trajectory](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Fodometry%2Flearned_intrinsics_trajectory_odo09.txt?generation=1568247378516045&alt=media) | [trajectory](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Fodometry%2Flearned_intrinsics_trajectory_odo10.txt?generation=1568247378745091&alt=media)
| Learned & corrected |  --- same ---| --- as --- | ---- above --- |[trajectory](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Fodometry%2Fcorrected_intrinsics_trajectory_odo09.txt?generation=1568247377030930&alt=media) |[trajectory](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Fodometry%2Fcorrected_intrinsics_trajectory_odo10.txt?generation=1568247377401528&alt=media)|
</center>

The code for generating "Learned & corrected" is not yet publically available.


## YouTube8M IDs of the videos used in the paper
`1ofm 2Ffk 2Gc7 2hdG 4Kdy 4gbW 70eK 77cq 7We1 8Eff 8W2O 8bfg 9q4L A8cd AHdn Ai8q
B8fJ BfeT C23C C4be CP6A EOdA Gu4d IdeB Ixfs Kndm L1fF M28T M92S NSbx NSfl NT57
Q33E Qu62 U4eP UCeG VRdE W0ch WU6A WWdu WY2M XUeS YLcc YkfI ZacY aW8r bRbL d79L
d9bU eEei ePaw iOdz iXev j42G j97W k7fi kxe2 lIbd lWeZ mw3B nLd8 olfE qQ8k qS6J
sFb2 si9H uofG yPeZ zger`

The YouTube8M [website](https://research.google.com/youtube8m/) provides the
instructions for mapping them you YouTube IDs. Two consecutive frames were
sampled off of each video every second.


## Checkpoints trained on EuRoC MAV
The checkpoint used to obtain the results in Table 4 in the paper is given
[here](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Feuroc_ckpt%2FMachineHallAll.zip?generation=1581119531504792&alt=media).
It was trained on all the "Machine Hall" sequences jointly, with
learned intrinsics. In addition, we trained a model on each sequence
separatelty, the results were used to obtain the numbers in Table 5. The
respective checkpoints are given in the table below. All models in this section
were trained with a resolution of 256x384.

<center>

|Room|01|02|03|04|
|:---------|:------:|:------:|:------:|:------:|:------:|
|Machine Hall | [checkpoint](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Feuroc_ckpt%2FMachineHall01.zip?generation=1581119518041994&alt=media) | [checkpoint](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Feuroc_ckpt%2FMachineHall02.zip?generation=1581119521439219&alt=media) | [checkpoint](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Feuroc_ckpt%2FMachineHall03.zip?generation=1581119524820502&alt=media) | [checkpoint](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Feuroc_ckpt%2FMachineHall04.zip?generation=1581119528160751&alt=media) |
|Vicon Room 1| [checkpoint](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Feuroc_ckpt%2FViconRoom1-01.zip?generation=1581119546027773&alt=media) | [checkpoint](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Feuroc_ckpt%2FViconRoom1-02.zip?generation=1581119549454887&alt=media) | [checkpoint](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Feuroc_ckpt%2FViconRoom1-03.zip?generation=1581119552830334&alt=media) ||
|Vicon Room 2 | [checkpoint](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Feuroc_ckpt%2FViconRoom2-01.zip?generation=1581119556122637&alt=media) | [checkpoint](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Feuroc_ckpt%2FViconRoom2-02.zip?generation=1581119572784082&alt=media) | [checkpoint](https://www.googleapis.com/download/storage/v1/b/gresearch/o/depth_from_video_in_the_wild%2Feuroc_ckpt%2FViconRoom2-03.zip?generation=1581119589325131&alt=media) ||

</center>


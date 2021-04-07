## KeyPose: Pose Estimation with Keypoints

This repository contains Tensorflow 2 models and a small set of
labeled transparent object data from the
[KeyPose project](https://sites.google.com/corp/view/keypose/).  There
are sample programs for displaying the data, running the models to
predict keypoints on the data, and training from data.

The full dataset can be downloaded using the directions at the end of this README.  It contains stereo and depth image sequences (720p) of 15 small transparent objects in 5 categories (ball, bottle, cup, mug, heart, tree), against 10 different textured backgrounds, with 4 poses for each object.  There are a total of 600 sequences with approximately 48k stereo and depth images.  The depth images are taken with both transparent and opaque objects in the exact same position.  All RGB and depth images are registered for pixel correspondence, the camera parameters and pose are given, and keypoints are labeled in each image and in 3D.

## Setup and sample programs

To install required python libraries (running in directory
above `keypose/`):
```
pip3 install -r keypose/requirements.txt
```


To look at images and ground-truth keypoints (running in directory
above `keypose/`):
```
$ python3 -m keypose.show_keypoints keypose/data/bottle_0/texture_5_pose_0/ keypose/objects/bottle_0.obj
```

To predict keypoints from a model (running in directory above
`keypose/`), first download the models:
```
keypose/download_models.sh
```
Then run the `predict` command:
```
$ python3 -m keypose.predict keypose/models/bottle_0_t5/ keypose/data/bottle_0/texture_5_pose_0/ keypose/objects/bottle_0.obj
```

## Repository layout

- top level: contains the sample programs.
- `configs/` contains training configuration files.
- `data/` contains the transparent object data.
- `models/` contains the trained Tensorflow models.
- `objects/` contains simplified vertex CAD files for each object, for use in display.
- `tfrecords/` contains tfrecord structures for training, created from
  transparent object data

### Data directory structure and files.

The directory structure for the data divides into one directory for each object, with sub-directories
for each texture/pose sequence.  Each sequence has about 80 images, numbered sequentially with a prefix.
```
  bottle_0/
      texture_0_pose_0/
           000000_L.png        - Left image (reference image)
           000000_L.pbtxt      - Left image parameters
           000000_R.png        - Right image
           000000_R.pbtxt      - Right image parameters
           000000_border.png   - Border of the object in the left image (grayscale)
           000000_mask.png     - Mask of the object in the left image (grayscale)
           000000_Dt.exr       - Depth image for the transparent object
           000000_Do.exr       - Depth image for the opaque object
           ...
```

### Model naming conventions.

In the `models/` directory, there are a set of sub-directories containing TF KeyPose models trained for individual and category predictions.
```
  bottle_0_t5/          - Trained on bottle_0, leaving out texture_5_pose_* data
  bottle_1_t5/
  ...
  bottles_t5/           - Trained on all bottles, leaving out texture_5_pose_* data
  bottles_cups_t5/
  mugs_t5/

  mugs_m0/              - Trained on all mugs except mug_0
```

So, for example, you can use the `predict.py` program to run the `mugs_m0` model against any of the sequences in `data/mug_0/`,
to show how the model performs against a mug it has not seen before.  Similarly, running the model `bottle_0_t5` against
any sequence in `data/bottle_0/texture_5_pose_*` will do predictions against the test set.

## Downloading data and models.

Data and models are available publicly on Google Cloud Storage.

To download all the models, use:
```
keypose/download_models.sh
```
This will populate the `models/` directory.

The image files are large, and divided up by object.  To get any individual object, use:
```
wget https://storage.googleapis.com/keypose-transparent-object-dataset/<object>.zip
```
Then unzip at the `keypose/` directory, and it will populate the appropriate `data/` directories.
These are the 15 available objects:
```
ball_0
bottle_0
bottle_1
bottle_2
cup_0
cup_1
mug_0
mug_1
mug_2
mug_3
mug_4
mug_5
mug_6
heart_0
tree_0
```

## Training a model.

To train a model, the data must be put into tfrecord format, and then grouped
according to train and test records.  There are programs below that will do
this.  For a quick start, you can download a set of tfrecords for training
a bottle_0 model that uses texture 5 as the test set.

To get the sample tfrecords (3.4 GB), use the following:
```
wget https://storage.googleapis.com/keypose-transparent-object-dataset/tfrecords.zip
```
Then unzip at the `keypose/` directory, and it will populate the appropriate
`tfrecords/` directories.

To train, use this command (running in directory above `keypose/`):
```
$ python3 -m keypose.trainer configs/bottle_0_t5 /tmp/model
```

This will run the training using Estimators, and put checkpoints and saved
models into `/tmp/model`.  It helps to have a good GPU with lots of memory.
If you run out of memory on the GPU, reduce the batch size in
`configs/default.yaml` from 32 to 16 or 8.  Periodically the trainer will
write out models in the saved_model format.

## Generating tfrecords from image directories

To put images and associated metadata into tfrecord format, suitable
for training, use the following command (running in directory above `keypose/`):
```
$ python3 -m keypose.gen_tfrecords configs/tfset_bottle_0_t5
``

The configuration file `configs/tfset_bottle_0_t5.pbtxt` is a protobuf
that names the tfrecord output directory, as well as the directory
paths for inputs for train and val.  It also lets you resize the
original images -- for example, the sample tfrecords used in training
are resized to 756x424, with a camera focal length of 400 pixels.

The output of the process is in `keypose/tfrecords/<name>/tfrecords`.
This directory is suitable for input to `trainer.py`.

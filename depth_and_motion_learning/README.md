# Unsupervised Monocular Depth and Motion Learning

This release supports joint training a depth prediction model and a motion
prediction model using only pairs of RGB images. The depth model infers a dense
depth map from a single image. The motion model infers a 6 degree-of-freedom
camera motion and a 3D dense motion field for every pixel from a pair of images.
The approach does not need any auxiliary semantic information from the images,
and the camera intrinsics can be either specified or learned.

Sample command line:

```bash
python -m depth_and_motion_learning.depth_motion_field_train \
  --model_dir=$MY_CHECKPOINT_DIR \
  --param_overrides='{
    "model": {
      "input": {
        "data_path": "$MY_DATA_DIR"
      }
    },
    "trainer": {
      "init_ckpt": "$MY_IMAGENET_CHECKPOINT",
      "init_ckpt_type": "imagenet"
    }
  }'
```

`MY_CHECKPOINT_DIR` is where the trained model checkpoints are to be saved.

`MY_DATA_DIR` is where the training data (in Struct2depth's format) is stored.
The `data_example` folder in project `depth_from_video_in_the_wild` contains a
single training example expressed in this format.

`MY_IMAGENET_CHECKPOINT` is a path to a pretreained ImageNet checkpoint to
intialize the encoder of the depth prediction model.

## Citation
If you use any part of this code in your research, please cite our
[paper](https://arxiv.org/abs/2010.16404):

```
@article{li2020unsupervised,
  title={Unsupervised Monocular Depth Learning in Dynamic Scenes},
  author={Li, Hanhan and Gordon, Ariel and Zhao, Hang and Casser, Vincent and Angelova, Anelia},
  journal={arXiv preprint arXiv:2010.16404},
  year={2020}
}
```

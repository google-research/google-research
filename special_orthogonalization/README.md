## An Analysis of SVD for Deep Rotation Estimation

This repository will contain the TensorFlow code for experiments contained in an upcoming NeurIPS 2020 paper:

**An Analysis of SVD for Deep Rotation Estimation** \
Jake Levinson, Carlos Esteves, Kefan Chen, Noah Snavely, Angjoo Kanazawa, Afshin Rostamizadeh, and Ameesh Makadia \
To appear in the *34th Conference on Neural Information Processing Systems
(NeurIPS 2020)*. \
[arXiv](https://arxiv.org/abs/2006.14616)



### Sample Code
Below is sample code to use SVD orthogonalization to generate 3D rotation
matrices from 9D inputs.


**TensorFlow:**

```
def symmetric_orthogonalization(x):
  """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

  x: should have shape [batch_size, 9]

  Returns a [batch_size, 3, 3] tensor, where each inner 3x3 matrix is in SO(3).
  """
  m = tf.reshape(x, (-1, 3, 3))
  _, u, v = tf.svd(m)
  det = tf.linalg.det(tf.matmul(u, v, transpose_b=True))
  r = tf.matmul(
      tf.concat([u[:, :, :-1], u[:, :, -1:] * tf.reshape(det, [-1, 1, 1])], 2),
      v, transpose_b=True)
  return r
```



**PyTorch:**

```
def symmetric_orthogonalization(x):
  """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

  x: should have size [batch_size, 9]

  Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
  """
  m = x.view(-1, 3, 3)
  u, s, v = torch.svd(m)
  vt = torch.transpose(v, 1, 2)
  det = torch.det(torch.matmul(u, vt))
  det = det.view(-1, 1, 1)
  vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
  r = torch.matmul(u, vt)
  return r
```



### Citation

```
@inproceedings{levinson20neurips,
  title = {An Analysis of {SVD} for Deep Rotation Estimation},
  author = {Jake Levinson, Carlos Esteves, Kefan Chen, Noah Snavely, Angjoo Kanazawa, Afshin Rostamizadeh, and Ameesh Makadia},
  booktitle = {Advances in Neural Information Processing Systems 34},
  year = {2020},
  note = {To appear in}
}
```


## Experiments

### Point Cloud Alignment
This experiment evaluates symmetric orthogonalization for point cloud alignemnt.
The code here is a TensorFlow version of the experiment described in *On the
Continuity of Rotation Representations in Neural Networks* (Zhou et al, CVPR19,
see also their [original PyTorch code](https://github.com/papagina/RotationContinuity)).

**Modify the original test data for our TF code:**

```bash
# From google-research/

# Set this to the path to the test .pts files provided with the original PyTorch
# package.
IN_FILES=/shapenet/data/pc_plane/points_test/*.pts

# Set this to the directory where the new modifed test files will be written.
# This directory should not contain any other .pts files.
NEW_TEST_FILES_DIR=/shapenet/data/pc_plane/points_test_modified

# This variable determines the distribution for random rotations. If True, uses
# the same axis-angle sampling as in the PyTorch code, otherwise rotations are
# sampled uniformly (Haar measure).
AXANG_SAMPLING=True

python -m special_orthogonalization.gen_pt_test_data --input_test_files=$IN_FILES --output_directory=$NEW_TEST_FILES_DIR --random_rotation_axang=$AXANG_SAMPLING
```


**Train and evaluate:**

```bash
# Set this to the location of the original training data.
TRAIN_FILES=/shapenet/data/pc_plane/points/*.pts

# NEW_TEST_FILES_DIR should be the same as used above.
TEST_FILES=$NEW_TEST_FILES_DIR/*.pts

# Specify the rotation representation method, e.g. 'svd', 'svd-inf', or 'gs'.
METHOD=svd

# Where the checkpoints, summaries, eval results, etc are stored.
OUT_DIR=/path/to/model

python -m special_orthogonalization.main_point_cloud --method=$METHOD --checkpoint_dir=$OUT_DIR --log_step_count=200 --save_summaries_steps=25000 --pt_cloud_train_files=$TRAIN_FILES --pt_cloud_test_files=$TEST_FILES --train_steps=2600000 --save_checkpoints_steps=100000 --eval_examples=39900
```


**Generate statistics over all test examples:**

```bash
# Mean, median, std, and percentiles will be printed.
python -m special_orthogonalization.main_point_cloud --method=$METHOD --checkpoint_dir=$OUT_DIR --pt_cloud_test_files=$TEST_FILES --predict_all_test=True
```

### Inverse Kinematics
In this experiment we simply replace the 3D rotation regression layer with the
symmetric orthogonalization (SVD) layer. See the few code changes needed
[here](https://github.com/amakadia/svd_for_pose#inverse-kinematics).


### Single Image Depth and Pose Prediction on KITTI
In this experiment we simply replace the 3D rotation regression layer with the
symmetric orthogonalization (SVD) layer. See the few code changes needed
[here](https://github.com/amakadia/svd_for_pose#single-image-depth-prediction-on-kitti).

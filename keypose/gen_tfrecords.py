# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A script to generate a tfrecord file from a folder containing the renderings.

See data.proto for protobuf definitions.

Format of the files in the folder:
1. Stereo images, with filenames ######_L/R.png
   - Format is RGBA, where the alpha channel indicates the object mask, if
     available.
   - Right images may be missing, for non-stereo data.

2. Camera / object transform and keypoints for each stereo pair in
   "######_L/R.pbtxt"
   - Protobuf (in text format) containing the transform, camera, and kps
   - 4x4 Transform takes 3D points in homogeneous form from the object to the
     camera frame.
   - Camera parameters containing focal length, center,
     stereo displacement.
   - Cameras are assumed to be rectified and aligned, with the same camera
     matrix.
   - Keypoints are ordered, and all files should have the same number of
     keypoints.  If a keypoint is to be skipped, it should be [-1, -1]

4. Target image camera / object transform and keypoints in files
   "target###.pbtxt"

Example usage (from directory above keypose/):
  python3 -m keypose.gen_tfrecords configs/tfset_bottle_0_t5
"""
import glob
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

from keypose import utils


def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def generate_tfrecords(image_dirs, resize_str, params, num_samples_per_rec=20):
  """Generate tfrecords in the image directories.

  tfrecords are of the form NNNN.tfrecord, in a tfrecord/ subdir.
  Images are resized according to params.

  Args:
    image_dirs: directories with image files.
    resize_str: subdir based on image resizing, e.g., 756x424_fl400.
    params: DataParams for the image size and camera.
    num_samples_per_rec: number of samples per tfrecord.
  """
  print('Image folders:', image_dirs)

  # Iterate over image folders, processing each.
  cam = params.camera
  for image_dir in image_dirs:
    print('Processing folder %s' % image_dir)
    # Get data params, set new ones.
    _, _, num_kp, cam_orig = utils.read_data_params(
        os.path.join(image_dir, 'data_params.pbtxt'))
    cam.baseline = cam_orig.baseline
    params.num_kp = num_kp
    out_dir = os.path.join(image_dir, resize_str)
    os.makedirs(out_dir, exist_ok=True)
    utils.write_to_text_file(os.path.join(out_dir, 'data_params.pbtxt'), params)

    # Read image files and generate tfrecords.
    filenames_left = glob.glob(os.path.join(image_dir, '*_L.png'))
    filenames_right = glob.glob(os.path.join(image_dir, '*_R.png'))
    filenames_target_left = glob.glob(os.path.join(image_dir, '*_L.pbtxt'))
    filenames_target_right = glob.glob(os.path.join(image_dir, '*_R.pbtxt'))
    if not filenames_right:
      filenames_right = filenames_left
      filenames_target_right = filenames_target_left
    else:
      print('Have stereo images')
    assert len(filenames_left) == len(filenames_target_left)
    assert len(filenames_target_left) == len(filenames_target_right)
    assert len(filenames_left) == len(filenames_right)
    filenames_left.sort()
    filenames_right.sort()
    filenames_target_left.sort()
    filenames_target_right.sort()
    total = len(filenames_left)
    print('Found %d total image examples' % total)

    j = 0
    while j < total:
      jj = min(total, j + num_samples_per_rec)
      record_name = ('%04d' % j) + '.tfrecord'
      print('Record file %s' % record_name)
      path = os.path.join(out_dir, record_name)
      print('Generating examples %d to %d' % (j, jj))
      generate(path, cam_orig, cam, filenames_left[j:jj], filenames_right[j:jj],
               filenames_target_left[j:jj], filenames_target_right[j:jj])
      j = jj


def generate(path, cam_orig, cam_new, fnames_left, fnames_right,
             fnames_target_left, fnames_target_right):
  """Generate tfrecords for a set of images and their metadata."""
  with tf.io.TFRecordWriter(path) as tfrecord_writer:
    with tf.Graph().as_default():
      im0 = tf.compat.v1.placeholder(dtype=tf.uint8)
      im1 = tf.compat.v1.placeholder(dtype=tf.uint8)
      encoded0 = tf.image.encode_png(im0)
      encoded1 = tf.image.encode_png(im1)

      with tf.compat.v1.Session() as sess:
        for fleft, fright, ftleft, ftright in zip(fnames_left, fnames_right,
                                                  fnames_target_left,
                                                  fnames_target_right):
          assert (os.path.basename(
              os.path.splitext(fleft)[0]) == os.path.basename(
                  os.path.splitext(ftleft)[0]))
          assert (os.path.basename(
              os.path.splitext(fright)[0]) == os.path.basename(
                  os.path.splitext(ftright)[0]))
          print(fleft)
          image_left = utils.read_image(fleft)
          image_right = utils.read_image(fright)
          targs_left = utils.read_target_pb(ftleft)
          targs_right = utils.read_target_pb(ftright)
          image_left = utils.resize_image(image_left, cam_new, cam_orig,
                                          targs_left)
          image_right = utils.resize_image(image_right, cam_new, cam_orig,
                                           targs_right)
          st0, st1 = sess.run([encoded0, encoded1],
                              feed_dict={
                                  im0: image_left,
                                  im1: image_right
                              })
          feats = {'img_L': bytes_feature(st0), 'img_R': bytes_feature(st1)}
          feats.update(make_features(targs_left, 'L'))
          feats.update(make_features(targs_right, 'R'))
          example = tf.train.Example(features=tf.train.Features(feature=feats))
          tfrecord_writer.write(example.SerializeToString())


def make_features(targs_pb, pf):
  """Make tfrecord featurs from image metadata."""
  camera, to_uvd, to_world, keys_uvd, _, visible, _ = utils.get_contents_pb(
      targs_pb.kp_target)
  num_kp = len(keys_uvd)
  # Restrict to max projection targets
  proj_targs = [
      utils.get_contents_pb(targ_pb) for targ_pb in targs_pb.proj_targets
  ][:utils.MAX_TARGET_FRAMES]
  targets_keys_uvd = []
  targets_to_uvd = []
  for proj_targ in proj_targs:
    _, to_uvd, _, keys_uvd, _, _, _ = proj_targ
    targets_keys_uvd.append(keys_uvd)
    targets_to_uvd.append(to_uvd)
    # Add dummy targets if necessary.
  num_targets = len(proj_targs)
  for _ in range(utils.MAX_TARGET_FRAMES - num_targets):
    targets_keys_uvd.append(utils.dummy_keys_uvd(num_kp))
    targets_to_uvd.append(utils.dummy_to_uvd())

  def feat_int(num):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[num]))

  def feat_floats(floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=floats))

  feats = {
      'to_world_' + pf:
          feat_floats(to_world.flatten()),
      'to_uvd_' + pf:
          feat_floats(to_uvd.flatten()),
      'camera_' + pf:
          feat_floats(utils.cam_pb_to_array(camera)),
      'keys_uvd_' + pf:
          feat_floats(np.array(keys_uvd).flatten()),
      'visible_' + pf:
          feat_floats(visible),
      'num_kp_' + pf:
          feat_int(num_kp),
      'num_targets_' + pf:
          feat_int(num_targets),
      'targets_to_uvd_' + pf:
          feat_floats(np.array(targets_to_uvd).flatten()),
      'targets_keys_uvd_' + pf:
          feat_floats(np.array(targets_keys_uvd).flatten()),
      'mirrored':
          feat_int(int(targs_pb.mirrored)),
  }
  return feats


def group_tfrecords(train_dirs, val_dirs, tfset, data_params):
  """Group all relevant tfrecords into a new tfrecord/ directory."""
  out_dir = os.path.join(utils.KEYPOSE_PATH,
                         os.path.join('tfrecords', tfset.name))
  records_out_dir = os.path.join(out_dir, 'tfrecords')
  os.makedirs(records_out_dir, exist_ok=True)

  start = 0
  train_names = []
  for dr in train_dirs:
    tfrecord_dir = os.path.join(dr, tfset.image_size)
    for tfname in glob.glob(os.path.join(tfrecord_dir, '*.tfrecord')):
      rname = ('%04d' % start) + '.tfrecord'
      fname = os.path.join(records_out_dir, rname)
      start += 1
      print('Copying %s to %s' % (tfname, fname))
      shutil.copy(tfname, fname)
      train_names.append(rname)

  val_names = []
  for dr in val_dirs:
    tfrecord_dir = os.path.join(dr, tfset.image_size)
    for tfname in glob.glob(os.path.join(tfrecord_dir, '*.tfrecord')):
      rname = ('%04d' % start) + '.tfrecord'
      fname = os.path.join(records_out_dir, rname)
      start += 1
      print('Copying %s to %s' % (tfname, fname))
      shutil.copy(tfname, fname)
      val_names.append(rname)

  tfset_pb = utils.make_tfset(train_names, val_names, tfset.name)
  utils.write_to_text_file(
      os.path.join(records_out_dir, 'tfset.pbtxt'), tfset_pb)
  utils.write_to_text_file(
      os.path.join(records_out_dir, 'data_params.pbtxt'), data_params)
  utils.write_to_text_file(os.path.join(out_dir, 'tfset_def.pbtxt'), tfset)


# Exclude directories from a list.
def exclude_dirs(dirs, exclude):
  return [dir for dir in dirs if dir not in exclude]


def main(argv):
  if not len(argv) >= 2:
    print(
        'Usage: ./trainer.py <config_file (configs/tfset_bottle_0_t5)> [tfset_only]'
    )
    exit(0)

  config_file = argv[1]
  tfset_only = False
  if len(argv) > 2:
    if argv[2] == 'tfset_only':
      tfset_only = True
    else:
      print('Second argument must be tfset_only')
      exit(0)

  # Check for tfset file.
  fname = os.path.join(utils.KEYPOSE_PATH, config_file + '.pbtxt')
  tfset = utils.read_tfset(fname)
  common_dir = tfset.common_dir
  print('common dir:', common_dir)
  print('train:', tfset.train)
  print('val:', tfset.val)
  print('exclude:', tfset.exclude)

  common_dir = os.path.join(utils.KEYPOSE_PATH, common_dir)
  # pylint: disable=g-complex-comprehension
  train_dirs = [
      dr for tfdir in tfset.train
      for dr in glob.glob(os.path.join(common_dir, tfdir))
  ]
  val_dirs = [
      dr for tfdir in tfset.val
      for dr in glob.glob(os.path.join(common_dir, tfdir))
  ]
  exc_dirs = [
      dr for tfdir in tfset.exclude
      for dr in glob.glob(os.path.join(common_dir, tfdir))
  ]
  train_dirs = exclude_dirs(train_dirs, exc_dirs)
  val_dirs = exclude_dirs(val_dirs, exc_dirs)
  train_dirs = exclude_dirs(train_dirs, val_dirs)
  train_dirs.sort()
  val_dirs.sort()
  all_dirs = train_dirs + val_dirs
  all_dirs.sort()

  # Set up data_params for the new image size.
  size, focal_length = tfset.image_size.split('_')
  resx, resy = size.split('x')
  resx = int(resx)
  resy = int(resy)
  fx = int(focal_length[2:])
  cam = utils.cam_array_to_pb([fx, fx, resx / 2, resy / 2, 0.0, resx, resy])
  data_params = utils.make_data_params(resx, resy, 0, cam)

  if not tfset_only:
    generate_tfrecords(all_dirs, tfset.image_size, data_params)

  group_tfrecords(train_dirs, val_dirs, tfset, data_params)


if __name__ == '__main__':
  main(sys.argv)

# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Dataset input functions for KeyPose estimator."""

import glob
import os

import numpy as np
import tensorflow as tf

from keypose import utils


def get_tfrecords(dset_dir, split='train'):
  """Get tfrecords that match training, validation, and test sets.

  Works off the TfSet protobuf.

  Args:
    dset_dir: dataset directory of tfrecords.
    split: train or eval.

  Returns:
    Relevant tfrecords from dset_dir.

  Raises:
    IOError: if tfset.pbtxt protobuf file not found.
  """
  tfset_fname = os.path.join(dset_dir, 'tfset.pbtxt')
  if not os.path.isfile(tfset_fname):
    raise IOError('%s not found' % tfset_fname)
  tfset_pb = utils.read_tfset(tfset_fname)

  fnames = []
  if split == 'train':
    fnames = tfset_pb.train
  elif split == 'val':
    fnames = tfset_pb.val
  elif split == 'test':
    fnames = tfset_pb.test
  elif split == 'all':
    fnames = glob.glob(os.path.join(dset_dir, '*.tfrecord'))
    print('Found %d tfrecord files' % len(fnames))
    return fnames

  return [os.path.join(dset_dir, f) for f in fnames]


# Return True if all kps are visible.
def filter_nonvisible(_, labels):
  """Check visibility of keypoints.

  Args:
    labels: labels of the tfrecord.

  Returns:
    True if all keypoints are visible.
  """
  visible = labels['visible_L']
  isallowed = tf.equal(visible, tf.constant(0.))
  reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
  visible_r = labels['visible_R']
  isallowed_r = tf.equal(visible_r, tf.constant(0.))
  reduced_r = tf.reduce_sum(tf.cast(isallowed_r, tf.float32))
  return tf.equal(tf.add(reduced, reduced_r), tf.constant(0.))


def permute_order(tensor, order, mask, is_target=False):
  """Reduces <tensor> of shape [order, ...] by picking out order dim.

  Args:
    tensor: input tensor, with orderings.
    order: order permutation.
    mask: which order to use.
    is_target: True if tensor is a target (???).

  Returns:
    Reduced tensor.
  """
  if is_target:
    tensor = tf.stack(
        # order is list of lists, so pylint: disable=not-an-iterable
        [tf.stack([tensor[:, x, Ellipsis] for x in ord], axis=1) for ord in order],
        axis=1)
    res = tf.multiply(tensor, mask)
    return tf.reduce_sum(res, axis=1)
  else:
    tensor = tf.stack(
        # order is list of lists, so pylint: disable=not-an-iterable
        [tf.stack([tensor[x, Ellipsis] for x in ord], axis=0) for ord in order],
        axis=0)
    res = tf.multiply(tensor, mask)
    return tf.reduce_sum(res, axis=0)


def parser(serialized_example, resx, resy, num_kp, sym=None, return_float=True):
  """Parses a single tf.Example into image and label tensors.

  Args:
    serialized_example: tfrecord sample.
    resx: image x resolution, pixels.
    resy: image y resolution, pixles.
    num_kp: number of keypoints.
    sym: list representing keypoint symmetry.
    return_float: True if we want a floating-point return.

  Returns:
    Feature set.
  """
  if sym is None:
    sym = [0]
  fs = tf.io.parse_single_example(
      serialized_example,
      features={
          'img_L':
              tf.io.FixedLenFeature([], tf.string),
          'img_R':
              tf.io.FixedLenFeature([], tf.string),
          'to_world_L':
              tf.io.FixedLenFeature([4, 4], tf.float32),
          'to_world_R':
              tf.io.FixedLenFeature([4, 4], tf.float32),
          'to_uvd_L':
              tf.io.FixedLenFeature([4, 4], tf.float32),
          'to_uvd_R':
              tf.io.FixedLenFeature([4, 4], tf.float32),
          'camera_L':
              tf.io.FixedLenFeature([7], tf.float32),
          'camera_R':
              tf.io.FixedLenFeature([7], tf.float32),
          'keys_uvd_L':
              tf.io.FixedLenFeature([num_kp, 4], tf.float32),
          'keys_uvd_R':
              tf.io.FixedLenFeature([num_kp, 4], tf.float32),
          'visible_L':
              tf.io.FixedLenFeature([num_kp], tf.float32),
          'visible_R':
              tf.io.FixedLenFeature([num_kp], tf.float32),
          'num_kp_L':
              tf.io.FixedLenFeature([], tf.int64),
          'num_kp_R':
              tf.io.FixedLenFeature([], tf.int64),
          'num_targets_L':
              tf.io.FixedLenFeature([], tf.int64),
          'num_targets_R':
              tf.io.FixedLenFeature([], tf.int64),
          'mirrored':
              tf.io.FixedLenFeature([], tf.int64),
          'targets_to_uvd_L':
              tf.io.FixedLenFeature([utils.MAX_TARGET_FRAMES, 4, 4],
                                    tf.float32),
          'targets_to_uvd_R':
              tf.io.FixedLenFeature([utils.MAX_TARGET_FRAMES, 4, 4],
                                    tf.float32),
          'targets_keys_uvd_L':
              tf.io.FixedLenFeature([utils.MAX_TARGET_FRAMES, num_kp, 4],
                                    tf.float32),
          'targets_keys_uvd_R':
              tf.io.FixedLenFeature([utils.MAX_TARGET_FRAMES, num_kp, 4],
                                    tf.float32),
      })

  fs['img_L'] = tf.image.decode_png(fs['img_L'], 4)
  if return_float:
    fs['img_L'] = tf.image.convert_image_dtype(fs['img_L'], tf.float32)
  fs['img_R'] = tf.image.decode_png(fs['img_R'], 4)
  if return_float:
    fs['img_R'] = tf.image.convert_image_dtype(fs['img_R'], tf.float32)

  fs['img_L'].set_shape([resy, resx, 4])
  fs['img_R'].set_shape([resy, resx, 4])

  # Check for randomizing keypoint symmetry order.
  if len(sym) > 1:
    order = utils.make_order(sym, num_kp)
    mask = [0.0 for x in sym]
    mask[0] = 1.0
    mask = tf.random.shuffle(mask)  # [order]
    mask2 = tf.expand_dims(mask, -1)  # [order, 1]
    mask2 = tf.expand_dims(mask2, -1)  # [order, 1, 1]
    mask3 = tf.stack(
        [mask2] * utils.MAX_TARGET_FRAMES, axis=0)  # [targets, order, 1, 1]
    fs['keys_uvd_L'] = permute_order(fs['keys_uvd_L'], order, mask2)
    fs['keys_uvd_R'] = permute_order(fs['keys_uvd_R'], order, mask2)
    fs['targets_keys_uvd_L'] = permute_order(fs['targets_keys_uvd_L'], order,
                                             mask3, True)
    fs['targets_keys_uvd_R'] = permute_order(fs['targets_keys_uvd_R'], order,
                                             mask3, True)
  print('Targets to_uvd shape [num_targs, 4, 4]:', fs['targets_to_uvd_L'].shape)
  print('Targets keys_uvd shape [num_targs, num_kp, 4]:',
        fs['targets_keys_uvd_L'].shape)

  return fs


def create_input_fn(params,
                    split='val',
                    keep_order=False,
                    filenames=None,
                    late_fusion=False):
  """Returns input_fn for tf.estimator.Estimator.

  Reads tfrecords and constructs input_fn for either training or eval. All
  tfrecords not in val.txt will be assigned to training set.
  train.txt and val.txt have the base filenames for their tfrecords, one per
  line.

  Args:
    params: ConfigParams structure.  There are various elements used --
      keep_order - True if order of images should be preserved. dset_dir -
      Directory containing tfrecord data of original images. batch_size - The
      batch size! occ_fraction - Fraction of object to occlude. noise - Random
      noise to add to an image, in pixel values; (a,b) is bounds. blur - Random
      gaussian blur to add, sigma in pixels (low, high). motion - Random motion
      blur to add (move_pixels_min, move_pixels_max, angle_deg_min,
      angle_deg_max). rot - (rotx, roty, rotz) for camera in degrees. gamma -
      Gamma value to modify image; (a,b) is bounds. num_kp - Number of
      keypoints.
      resx, resy - Size of input images.  NB: using 'crop' causes output size to
        change. crop - Crop image and offset right crop, e.g., [280, 180, 20].
        sym - keypoint symmetry list, [0] is no symmetry. kp_occ_radius - Radius
        (m) around a keypoint for occlusion.
    split: A string indicating the split. Can be either 'train' or 'val'.
    keep_order: keep tfrecord order, for prediction of a sequence.
    filenames: set of tfrecord filenames.
    late_fusion: True if left / right images are handled separately by the net.

  Returns:
    input_fn for tf.estimator.Estimator.

  Raises:
    IOError: If val.txt is not found.
  """

  print('Input function called for split <%s>' % split)

  mparams = params.model_params
  dset_dir = params.dset_dir
  batch_size = params.batch_size
  occ_fraction = mparams.occ_fraction
  noise = mparams.noise
  blur = mparams.blur
  motion = mparams.motion
  rotation = mparams.rot
  gamma = mparams.gamma
  num_kp = mparams.num_kp
  resx = mparams.resx
  resy = mparams.resy
  crop = mparams.crop
  shear = mparams.shear
  scale = mparams.scale
  flip = mparams.flip
  sym = mparams.sym
  input_sym = mparams.input_sym
  dither = mparams.dither
  visible_only = mparams.visible

  if not filenames:
    filenames = get_tfrecords(dset_dir, split)
  if not filenames:
    raise IOError('No tfrecord files found!')
  print('Found %d files' % len(filenames))
  bg_filenames = ['']  # No filename, just use a dummy image.

  def input_fn():
    """input_fn for tf.estimator.Estimator."""

    def zip_parser(sample, bg_filename):
      """Modifies the input training images.

      Args:
        sample: A dict of features from the TFRecord files.
        bg_filename: A string, filename of a background scene.

      Returns:
        A sample with the images modified by image augmentation.
      """
      image = sample['img_L']
      image2 = sample['img_R']
      rgba_shape = image.get_shape()
      final_shape = rgba_shape
      to_world = sample['to_world_L']
      to_world = tf.reshape(to_world, [4, 4])
      keys_uvd = sample['keys_uvd_L']
      keys_uvd = tf.reshape(keys_uvd, [-1, 4])
      visible = sample['visible_L']
      keys_uvd_r = sample['keys_uvd_R']
      keys_uvd_r = tf.reshape(keys_uvd_r, [-1, 4])
      camera = sample['camera_L']
      mirrored = sample['mirrored']
      # uvd offset based on cropping.
      sample['offsets'] = tf.constant([0.0, 0.0, 0.0])

      # Causes issues for rotation (keypoint changes).
      if split == 'train' and (rotation[0] or rotation[1] or rotation[2]):
        # Random rotation around camera center.

        image, image2, transform, keys_uvd, visible = tf.numpy_function(
            func=utils.do_rotation,
            inp=[image, image2, to_world, camera, keys_uvd, visible, rotation],
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
        image.set_shape(rgba_shape)
        image2.set_shape(rgba_shape)
        transform.set_shape(to_world.get_shape())
        keys_uvd.set_shape(keys_uvd.get_shape())
        visible.set_shape(sample['visible_L'].get_shape())
        sample['to_world_L'] = transform
        sample['keys_uvd_L'] = keys_uvd
      else:
        image = utils.image_uint8_to_float(image)
        image2 = utils.image_uint8_to_float(image2)

      final_shape = [crop[1], crop[0], 4]
      # Will filter if visible returns a 0.
      image, image2, offsets, visible = tf.numpy_function(
          func=utils.do_occlude_crop,
          inp=[
              image, image2, keys_uvd, keys_uvd_r, crop, visible, dither,
              late_fusion
          ],
          Tout=[tf.float32, tf.float32, tf.float32, tf.float32])
      sample['offsets'] = offsets
      sample['visible_L'] = visible
      image.set_shape(final_shape)
      image2.set_shape(final_shape)
      if occ_fraction and split == 'train':
        image, image2 = tf.numpy_function(
            func=utils.do_occlude,
            inp=[image, image2, occ_fraction],
            Tout=[tf.float32, tf.float32])
        image.set_shape(final_shape)
        image2.set_shape(final_shape)

      # Check for mirrored sample.
      image, image2, hom = tf.numpy_function(
          func=utils.do_vertical_flip,
          inp=[image, image2, mirrored],
          Tout=[tf.float32, tf.float32, tf.float32])
      image.set_shape(final_shape)
      image2.set_shape(final_shape)
      hom.set_shape((3, 3))

      # 2D geometric transform.
      image, image2, hom = tf.numpy_function(
          func=utils.do_2d_homography,
          inp=[image, image2, scale, shear, flip, mirrored, split],
          Tout=[tf.float32, tf.float32, tf.float32])
      image.set_shape(final_shape)
      image2.set_shape(final_shape)
      hom.set_shape((3, 3))

      sample['hom'] = hom

      # Compute ground truth spatial probabilities.
      # TODO(konolige): Make variance a parameter.
      probs = tf.numpy_function(
          func=utils.do_spatial_prob,
          inp=[keys_uvd, hom, offsets, 20.0, crop[:2]],
          Tout=tf.float32)
      probs.set_shape([num_kp, final_shape[0], final_shape[1]])
      sample['prob_label'] = probs

      # Composite with background.
      image = tf.numpy_function(
          func=utils.do_composite,
          inp=[image, '', blur, motion, (0.0, noise),
               gamma],  # Use gray background.
          Tout=tf.float32)
      # Shape is [h, w, 3]
      image.set_shape((final_shape[0], final_shape[1], 3))
      image2 = tf.numpy_function(
          func=utils.do_composite,
          inp=[image2, bg_filename, blur, motion, (0.0, noise), gamma],
          Tout=tf.float32)
      # Shape is [h, w, 3]
      image2.set_shape((final_shape[0], final_shape[1], 3))

      if split == 'train':
        # Introduce photometric randomness for generalization.
        brightness = 32.0
        contrast = (0.7, 1.2)
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_saturation(image, 0.6, 1.2)
        image = tf.image.random_contrast(image, contrast[0], contrast[1])
        image = tf.image.random_brightness(image, max_delta=brightness / 255.0)
        image = tf.clip_by_value(image, 0.0, 1.0)
        image2 = tf.image.random_hue(image2, max_delta=0.1)
        image2 = tf.image.random_saturation(image2, 0.6, 1.2)
        image2 = tf.image.random_contrast(image2, contrast[0], contrast[1])
        image2 = tf.image.random_brightness(
            image2, max_delta=brightness / 255.0)
        image2 = tf.clip_by_value(image2, 0.0, 1.0)

      features = {}
      features['img_L'] = tf.image.convert_image_dtype(image, tf.float32)
      features['img_R'] = tf.image.convert_image_dtype(image2, tf.float32)
      features['offsets'] = sample['offsets']
      features['hom'] = sample['hom']
      features['to_world_L'] = sample['to_world_L']
      del sample['img_L']
      del sample['img_R']
      return features, sample  # That is, (features, labels) for Estimator.

    # Main code of create_input_fn here.
    pcount = 8
    rcount = pcount
    if split == 'train':
      np.random.shuffle(filenames)
    else:
      filenames.sort()

    np.random.shuffle(bg_filenames)
    bg_dataset = tf.data.Dataset.from_tensor_slices(bg_filenames)
    bg_dataset = bg_dataset.shuffle(200).repeat()

    if keep_order:
      rcount = 1
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=rcount)
    parse_sym = sym
    if len(input_sym) > 1:
      parse_sym = input_sym  # Mix up keypoints on input.

    def parse_it(x):
      return parser(x, resx, resy, num_kp, sym=parse_sym, return_float=True)

    dataset = dataset.map(parse_it, num_parallel_calls=pcount)

    if split == 'train':
      dataset = dataset.shuffle(1200)
      if not keep_order:
        dataset = dataset.repeat()
    elif batch_size > 1 and keep_order:
      dataset = dataset.flat_map(
          lambda x: tf.data.Dataset.from_tensors(x).repeat(batch_size))

    # Add a random background and composite it in.
    dataset = tf.data.Dataset.zip((dataset, bg_dataset))
    dataset = dataset.map(zip_parser, num_parallel_calls=pcount)
    if visible_only:
      dataset = dataset.filter(filter_nonvisible)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=1)

    return dataset

  return input_fn

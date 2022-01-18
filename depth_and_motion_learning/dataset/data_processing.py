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

"""Functions for data processing in struct2depth readers."""

import six
from six.moves import range
import tensorflow.compat.v1 as tf

from depth_and_motion_learning.dataset import data_processing_util as util


def _no_op(x, *unused_args, **unused_kwargs):
  return x


_NO_OP = {'crop': _no_op, 'resize': _no_op, 'flip': _no_op}

# This dictionary maps a data type (rgb image, depth image, intrinsics, etc.)
# and an operation (resize, crop, flip) to a function that executes the
# operation.
(_FUNCTION_DICT) = dict(
    rgb={
        'crop': util.crop_image,
        'resize': util.resize_area,
        'flip': util.flip_left_right,
    },
    mask={
        'crop': util.crop_image,
        # A segmentation mask encodes object IDs are RGB values. Therefore we do
        # not interpolate but use a nearest-neighbor resizing.
        'resize': util.resize_nearest_neighbor,
        'flip': util.flip_left_right,
    },
    validity_mask={
        'crop': util.crop_image,
        # A validity mask encodes regions with meanigful values.
        # Therefore we do not interpolate but use a nearest-neighbor resizing.
        'resize': util.resize_nearest_neighbor,
        'flip': util.flip_left_right,
    },
    depth={
        'crop': util.crop_image,
        'resize': util.resize_area,
        'flip': util.flip_left_right,
    },
    ground_truth_mask={
        'crop': util.crop_image,
        # A segmentation mask encodes object IDs are RGB values. Therefore we do
        # not interpolate but use a nearest-neighbor resizing.
        'resize': util.resize_nearest_neighbor,
        'flip': util.flip_left_right,
    },
    depth_confidence={
        'crop': util.crop_image,
        # A mask encodes the confidence of depth.
        'resize': util.resize_area,
        'flip': util.flip_left_right,
    },
    normals={
        'crop': util.crop_image,
        'resize': util.resize_area,
    },
    intrinsics={
        'crop': util.crop_intrinsics,
        'resize': util.resize_intrinsics,
        'flip': util.flip_intrinsics
    },
    egomotion_mat={
        'crop': util.crop_egomotion,
        'resize': util.resize_egomotion,
        'flip': util.flip_egomotion
    },
    video_index=_NO_OP,
)


def make_intrinsics_mat(intrinsics):
  """Generates an intrinsics matrix from a 1-d input array of intrinsics.

  Args:
    intrinsics: 1-d array containing w, h, fx, fy, x0, y0.

  Returns:
    An intrinsics matrix represented as 3x3 tensor.
  """
  _, _, fx, fy, x0, y0 = tf.unstack(intrinsics)
  with tf.name_scope('make_intrinsics_mat'):
    return tf.convert_to_tensor([[fx, 0.0, x0], [0.0, fy, y0], [0.0, 0.0, 1.0]])


def crop(endpoints, offset_height, offset_width, target_height, target_width):
  """Produces an endpoint dictionary of cropped images and camera intrinsics.

  Args:
    endpoints: a dictinonary containing a rgb and depth image and camera
      intrinsics. It is expected that the rgb and depth image have the same
      shape.
    offset_height: amount of offset in y direction.
    offset_width: amount of offset in x direction.
    target_height: target height of images.
    target_width: target width of images.

  Returns:
    An endpoint dictionary of cropped images and camera intrinsics.
  """
  with tf.name_scope('crop'):
    ep = dict(endpoints)
    for key in ep:
      processing_fn = _FUNCTION_DICT[key]['crop']
      ep[key] = processing_fn(ep[key], offset_height, offset_width,
                              target_height, target_width)
    return ep


def resize(endpoints, size):
  """Produces a tuple of endpoint dictionaries of resized images and intrinsics.

  Args:
    endpoints: a tuple of dictionaries each containing a rgb and depth image and
      camera intrinsics. It is expected that all dictionaries in the tuple have
      the same keys and each image in the dictionary has the same shape.
    size: a tuple containing the target image size.

  Returns:
    A tuple of endpoint dictionaries of resized images and camera intrinsics.
  """
  with tf.name_scope('resize'):
    output_ep = []
    for ep in endpoints:
      ep = dict(ep)
      for key in ep:
        processing_fn = _FUNCTION_DICT[key]['resize']
        ep[key] = processing_fn(ep[key], size)
      output_ep.append(ep)
    return output_ep


def flip(endpoints):
  """Produces an endpoint dictionary of flipped images and camera intrinsics.

  Args:
    endpoints: a dictinoary containing a rgb and depth image and camera
      intrinsics. It is expected that the rgb and depth image have the same
      shape.

  Returns:
    An endpoint dictionary of flipped images and camera intrinsics.
  """
  with tf.name_scope('flip'):
    ep = dict(endpoints)
    for key in ep:
      processing_fn = _FUNCTION_DICT[key]['flip']
      ep[key] = processing_fn(ep[key])
    return ep


def random_crop(endpoints,
                target_height,
                target_width,
                random_uniform_fn=tf.random.uniform):
  """Randomly crops images and transforms intrinsics accordingly.

  Args:
    endpoints: a tuple of dictionaries each containing a rgb and depth image and
      camera intrinsics. It is expected that all dictionaries in the tuple have
      the same keys and each image in the dictionary has the same shape.
    target_height: target height of images.
    target_width: target width of images.
    random_uniform_fn: a function to generate uniformly distributed random
                       numbers, set to tf.random.unfiorm per default. Intended
                       to be replaced by a function that returns deterministic
                       values for testing purposes.

  Returns:
    A tuple of endpoint dictionaries of randomly cropped images and camera
    intrinsics; rgb and depth images are cropped with the same random
    parameters.
  """
  with tf.name_scope('random_crop'):
    h, w, _ = tf.unstack(tf.shape(endpoints[0]['rgb']))
    offset_height = random_uniform_fn([],
                                      0,
                                      h - target_height + 1,
                                      dtype=tf.dtypes.int32)
    offset_width = random_uniform_fn([],
                                     0,
                                     w - target_width + 1,
                                     dtype=tf.dtypes.int32)
    output_ep = []
    for ep in endpoints:
      output_ep.append(
          crop(ep, offset_height, offset_width, target_height, target_width))

    return tuple(output_ep)


def random_crop_and_size(endpoints,
                         min_height,
                         min_width,
                         random_uniform_fn=tf.random.uniform):
  """Randomly crops and resizes images and adjusts camera intrinsics.

  Args:
    endpoints: a tuple of dictionaries containing a rgb and depth image and
      camera intrinsics. It is expected that the rgb and depth image have the
      same shape.
    min_height: minimum height of images.
    min_width: minimum width of images.
    random_uniform_fn: a function to generate uniformly distributed random
      numbers, set to tf.random.unfiorm as default. Intended to be replaced by a
      function that returns deterministic values for testing purposes.

  Returns:
    A tuple of endpoint dictionaries of randomly cropped images and camera
    intrinsics; rgb and depth images are cropped with the same random
    parameters.
  """
  with tf.name_scope('random_crop'):
    h, w, _ = tf.unstack(tf.shape(endpoints[0]['rgb']))
    target_height = random_uniform_fn([],
                                      min_height,
                                      h + 1,
                                      dtype=tf.dtypes.int32)
    target_width = random_uniform_fn([],
                                     min_width,
                                     w + 1,
                                     dtype=tf.dtypes.int32)

    return random_crop(endpoints, target_height, target_width,
                       random_uniform_fn)


def random_flip(endpoints, random_uniform_fn=tf.random.uniform):
  """Randomly flips images and adjusts camera intrinsics.

  Args:
    endpoints: a tuple of dictionaries each containing a rgb and depth image and
      camera intrinsics. It is expected that the rgb and depth image have the
      same shape.
    random_uniform_fn: a function to generate uniformly distributed random
      numbers, set to tf.random.unfiorm as default. Intended to be replaced by a
      function that returns deterministic values for testing purposes.

  Returns:
    A tuple of endpoint dictionaries of randomly flipped images and camera
    intrinsics; rgb and depth images are flipped jointly.
  """
  with tf.name_scope('random_flip'):
    should_flip = tf.cast(
        random_uniform_fn([], 0, 2, dtype=tf.dtypes.int32), tf.dtypes.bool)

    def flip_all_ep(endpoints):
      return [flip(ep) for ep in endpoints]

    return tf.cond(
        should_flip,
        true_fn=lambda: flip_all_ep(endpoints),
        false_fn=lambda: endpoints)


def to_tuple_of_dicts(endpoints):
  """Converts a dictionary of batches into a tuple of dictionaries.

  Args:
    endpoints: a dictionary containing tuples of endpoints. It is expected that
      all batches in the dictionary have the same length and that all images
      have the same shape.

  Returns:
    A tuple of dictionaries, each containing rgb and depth images and
    intrinsics.
  """
  with tf.name_scope('unstack_fn'):
    data = list()
    keys = list()
    for key in endpoints:
      keys.append(key)
      unstacked_data = tuple(tf.unstack(endpoints[key]))
      data.append(unstacked_data)

    it = iter(data)
    num_frames = len(next(it))
    if not all(len(lst) == num_frames for lst in it):
      raise ValueError('Image and intrinsics tuples not of same length\n'
                       'Keys: %s\n Values: %s' % (keys, data))

    tuple_of_dicts = [dict() for _ in range(num_frames)]
    for i in range(len(keys)):
      for j in range(num_frames):
        tuple_of_dicts[j][keys[i]] = data[i][j]

    return tuple(tuple_of_dicts)


def to_dict_of_tuples(endpoints):
  """Converts a tuple of dictionaries into a dictionary of tuples.

  Args:
    endpoints: a tuple of dictionaries containing endpoints. It is expected that
      all images in the dictionaries have the same keys and all images have the
      same shape.

  Returns:
    A dictionary of tuples, containing rgb and depth images and camera
    intrinsics.
  """
  with tf.name_scope('stack_fn'):
    output_ep = dict()

    for ep in endpoints:
      ep = dict(ep)
      for key, value in six.iteritems(ep):
        if key in output_ep:
          output_ep[key].append(value)
        else:
          output_ep.setdefault(key, []).append(value)

    for key in output_ep:
      output_ep[key] = tuple(output_ep[key])

    return output_ep


def maybe_add_intrinsics_matrices(endpoints):
  """Resolves a 1-d camera intrinsics tensor into intrinsic matrices.

  Args:
    endpoints: a tuple of dictionaries of rgb (and depth) images and camera
    intrinsics.

  Returns:
    A tuple of dictionaries, each containing rgb (and depth) images and
    intrinsics and inverse intrinsics matrices.
  """
  with tf.name_scope('add_intrinsics_matrices'):
    extended_ep = []
    for ep in endpoints:
      ep = dict(ep)
      if 'intrinsics' in ep:
        intrinsics = ep['intrinsics']
        intrinsics_mat = make_intrinsics_mat(intrinsics)
        ep['intrinsics_mat'] = intrinsics_mat
        ep['intrinsics_mat_inv'] = tf.linalg.inv(intrinsics_mat)
        del ep['intrinsics']
      extended_ep.append(ep)
    return extended_ep


def random_crop_and_resize_pipeline(endpoints,
                                    target_height,
                                    target_width,
                                    enable_flipping=True):
  """Random crop and resize pipeline.

  It is assumed that the provided dictionaries contained in the
  provided tuple hold rgb (and optionally depth) images and a 1-d array with
  camera intrinsics values.

  Args:
    endpoints: a dictionary of batches of size S, each containing a rgb (and
      depth) image and camera intrinsics.
               'rgb': a tensor of shape [S, H, W, 3] representing a pair of RGB
                 images.
               'depth': a tensor of shape [S, H, W, 3] representing a pair of
                 depth images (optional).
               'intrinsics': a tensor of shape [S, 6] representing a pair of
                 camera intrinsics.
    target_height: target height of fetched images.
    target_width: target width of fetched images.
    enable_flipping: whether or not to enable vertical flipping of images.

  Returns:
    mod_endpoints: a tuple of dictionaries, each containing rgb and depth image
    and intrinsics and inverse intrinsics matrices.
  """

  # Convert dictionary of tuples into a tuple of dictionaries. Conversion is
  # necessary as looping over tuples raises the exception "Trying to capture a
  # tensor from an inner function...".
  mod_endpoints = to_tuple_of_dicts(endpoints)

  # Crop and resize rgb and depth image jointly and adjust camera intrinsics.
  mod_endpoints = random_crop_and_size(mod_endpoints, target_height,
                                       target_width)

  # Resize area of images and adjust camera intrinsics.
  mod_endpoints = resize(mod_endpoints, (target_height, target_width))

  if enable_flipping:
    # Randomly flip rgb and depth image. We do not want to flip normal images.
    mod_endpoints = random_flip(mod_endpoints)

  # Replace 1-d camera intrinsics tensor with intrinsics matrix and its inverse
  # if camera intrinsics are present.
  mod_endpoints = maybe_add_intrinsics_matrices(mod_endpoints)

  # Convert tuple of dictionaries back to dictionary of tuples.
  mod_endpoints = to_dict_of_tuples(mod_endpoints)

  return mod_endpoints


def resize_pipeline(endpoints, target_height, target_width):
  """Pipeline that only resizes images.

  It is assumed that the provided dictionaries contained in the
  provided tuple hold rgb (and optionally depth) images and a 1-d array with
  camera intrinsics values.

  Args:
    endpoints: a dictionary of batches of size S, each containing a rgb (and
      depth) image and camera intrinsics.
               'rgb': a tensor of shape [S, H, W, 3] representing a pair of RGB
                 images.
               'depth': a tensor of shape [S, H, W, 3] representing a pair of
                 depth images (optional).
               'intrinsics': a tensor of shape [S, 6] representing a pair of
                 camera intrinsics.
    target_height: target height of fetched images.
    target_width: target width of fetched images.

  Returns:
    mod_endpoints: a dictionary of tuples, each containing rgb and depth image
    and intrinsics and inverse intrinsics matrices.
  """

  # Convert dictionary of tuples into a tuple of dictionaries. Conversion is
  # necessary as looping over tuples raises the exception "Trying to capture a
  # tensor from an inner function...".
  mod_endpoints = to_tuple_of_dicts(endpoints)

  # Resize area of images and adjust camera intrinsics.
  mod_endpoints = resize(mod_endpoints, (target_height, target_width))

  # Replace 1-d camera intrinsics tensor with intrinsics matrix and its inverse.
  mod_endpoints = maybe_add_intrinsics_matrices(mod_endpoints)

  # Convert tuple of dictionaries back to dictionary of tuples.
  mod_endpoints = to_dict_of_tuples(mod_endpoints)

  return mod_endpoints


def no_op_pipeline(endpoints):
  """Function to convert endpoints of the reader to dictionary of tuples.

  This function is supposed to be used by readers that take care of converting
  data to the proper format already.

  Args:
    endpoints: a tuple of dictionaries of rgb (and depth) images and camera
      intrinsics.

  Returns:
    A tuple of dictionaries, each containing rgb (and depth) images and
    intrinsics and inverse intrinsics matrices.
  """

  # Convert dictionary of tuples into a tuple of dictionaries. Conversion is
  # necessary as looping over tuples raises the exception "Trying to capture a
  # tensor from an inner function...".
  mod_endpoints = to_tuple_of_dicts(endpoints)

  # Convert tuple of dictionaries back to dictionary of tuples.
  mod_endpoints = to_dict_of_tuples(mod_endpoints)

  return mod_endpoints

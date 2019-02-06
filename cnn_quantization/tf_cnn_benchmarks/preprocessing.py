# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Image pre-processing utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from cnn_quantization.tf_cnn_benchmarks import cnn_util
from cnn_quantization.tf_cnn_benchmarks import mlperf
from tensorflow.contrib.data.python.ops import threadpool
from tensorflow.contrib.image.python.ops import distort_image_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.framework import function
from tensorflow.python.layers import utils
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile


def parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields:

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox, features['image/class/text']


_RESIZE_METHOD_MAP = {
    'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    'bilinear': tf.image.ResizeMethod.BILINEAR,
    'bicubic': tf.image.ResizeMethod.BICUBIC,
    'area': tf.image.ResizeMethod.AREA
}


def get_image_resize_method(resize_method, batch_position=0):
  """Get tensorflow resize method.

  If resize_method is 'round_robin', return different methods based on batch
  position in a round-robin fashion. NOTE: If the batch size is not a multiple
  of the number of methods, then the distribution of methods will not be
  uniform.

  Args:
    resize_method: (string) nearest, bilinear, bicubic, area, or round_robin.
    batch_position: position of the image in a batch. NOTE: this argument can
      be an integer or a tensor
  Returns:
    one of resize type defined in tf.image.ResizeMethod.
  """

  if resize_method != 'round_robin':
    return _RESIZE_METHOD_MAP[resize_method]

  # return a resize method based on batch position in a round-robin fashion.
  resize_methods = list(_RESIZE_METHOD_MAP.values())
  def lookup(index):
    return resize_methods[index]

  def resize_method_0():
    return utils.smart_cond(batch_position % len(resize_methods) == 0,
                            lambda: lookup(0), resize_method_1)

  def resize_method_1():
    return utils.smart_cond(batch_position % len(resize_methods) == 1,
                            lambda: lookup(1), resize_method_2)

  def resize_method_2():
    return utils.smart_cond(batch_position % len(resize_methods) == 2,
                            lambda: lookup(2), lambda: lookup(3))

  # NOTE(jsimsa): Unfortunately, we cannot use a single recursive function here
  # because TF would not be able to construct a finite graph.

  return resize_method_0()


def decode_jpeg(image_buffer, scope=None):  # , dtype=tf.float32):
  """Decode a JPEG string into one 3-D float image Tensor.

  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  # with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
  # with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
  with tf.name_scope(scope or 'decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3,
                                 fancy_upscaling=False,
                                 dct_method='INTEGER_FAST')

    # image = tf.Print(image, [tf.shape(image)], 'Image shape: ')

    return image


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]


def normalized_image(images):
  # Rescale from [0, 255] to [0, 2]
  images = tf.multiply(images, 1. / 127.5)
  # Rescale to [-1, 1]
  mlperf.logger.log(key=mlperf.tags.INPUT_MEAN_SUBTRACTION, value=[1.0] * 3)
  return tf.subtract(images, 1.0)


def eval_image(image,
               height,
               width,
               batch_position,
               resize_method,
               summary_verbosity=0):
  """Get the image for model evaluation.

  We preprocess the image simiarly to Slim, see
  https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
  Validation images do not have bounding boxes, so to crop the image, we first
  resize the image such that the aspect ratio is maintained and the resized
  height and width are both at least 1.145 times `height` and `width`
  respectively. Then, we do a central crop to size (`height`, `width`).

  Args:
    image: 3-D float Tensor representing the image.
    height: The height of the image that will be returned.
    width: The width of the image that will be returned.
    batch_position: position of the image in a batch, which affects how images
      are distorted and resized. NOTE: this argument can be an integer or a
      tensor
    resize_method: one of the strings 'round_robin', 'nearest', 'bilinear',
      'bicubic', or 'area'.
    summary_verbosity: Verbosity level for summary ops. Pass 0 to disable both
      summaries and checkpoints.
  Returns:
    An image of size (output_height, output_width, 3) that is resized and
    cropped as described above.
  """
  # TODO(reedwm): Currently we resize then crop. Investigate if it's faster to
  # crop then resize.
  with tf.name_scope('eval_image'):
    if summary_verbosity >= 3:
      tf.summary.image(
          'original_image', tf.expand_dims(image, 0))

    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    image_height_float = tf.cast(image_height, tf.float32)
    image_width_float = tf.cast(image_width, tf.float32)

    # This value is chosen so that in resnet, images are cropped to a size of
    # 256 x 256, which matches what other implementations do. The final image
    # size for resnet is 224 x 224, and floor(224 * 1.145) = 256.
    scale_factor = 1.145

    # Compute resize_height and resize_width to be the minimum values such that
    #   1. The aspect ratio is maintained (i.e. resize_height / resize_width is
    #      image_height / image_width), and
    #   2. resize_height >= height * `scale_factor`, and
    #   3. resize_width >= width * `scale_factor`
    max_ratio = tf.maximum(height / image_height_float,
                           width / image_width_float)
    resize_height = tf.cast(image_height_float * max_ratio * scale_factor,
                            tf.int32)
    resize_width = tf.cast(image_width_float * max_ratio * scale_factor,
                           tf.int32)
    mlperf.logger.log_input_resize_aspect_preserving(height, width,
                                                     scale_factor)

    # Resize the image to shape (`resize_height`, `resize_width`)
    image_resize_method = get_image_resize_method(resize_method, batch_position)
    distorted_image = tf.image.resize_images(image,
                                             [resize_height, resize_width],
                                             image_resize_method,
                                             align_corners=False)

    # Do a central crop of the image to size (height, width).
    # MLPerf requires us to log (height, width) with two different keys.
    mlperf.logger.log(key=mlperf.tags.INPUT_CENTRAL_CROP, value=[height, width])
    mlperf.logger.log(key=mlperf.tags.INPUT_RESIZE, value=[height, width])
    total_crop_height = (resize_height - height)
    crop_top = total_crop_height // 2
    total_crop_width = (resize_width - width)
    crop_left = total_crop_width // 2
    distorted_image = tf.slice(distorted_image, [crop_top, crop_left, 0],
                               [height, width, 3])

    distorted_image.set_shape([height, width, 3])
    if summary_verbosity >= 3:
      tf.summary.image(
          'cropped_resized_image', tf.expand_dims(distorted_image, 0))
    image = distorted_image
  return image


def train_image(image_buffer,
                height,
                width,
                bbox,
                batch_position,
                resize_method,
                distortions,
                scope=None,
                summary_verbosity=0,
                distort_color_in_yiq=False,
                fuse_decode_and_crop=False):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    batch_position: position of the image in a batch, which affects how images
      are distorted and resized. NOTE: this argument can be an integer or a
      tensor
    resize_method: round_robin, nearest, bilinear, bicubic, or area.
    distortions: If true, apply full distortions for image colors.
    scope: Optional scope for op_scope.
    summary_verbosity: Verbosity level for summary ops. Pass 0 to disable both
      summaries and checkpoints.
    distort_color_in_yiq: distort color of input images in YIQ space.
    fuse_decode_and_crop: fuse the decode/crop operation.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  # with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
  # with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
  with tf.name_scope(scope or 'distort_image'):
    # A large fraction of image datasets contain a human-annotated bounding box
    # delineating the region of the image containing the object of interest.  We
    # choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    min_object_covered = 0.1
    aspect_ratio_range = [0.75, 1.33]
    area_range = [0.05, 1.0]
    max_attempts = 100
    mlperf.logger.log(key=mlperf.tags.INPUT_DISTORTED_CROP_MIN_OBJ_COV,
                      value=min_object_covered)
    mlperf.logger.log(key=mlperf.tags.INPUT_DISTORTED_CROP_RATIO_RANGE,
                      value=aspect_ratio_range)
    mlperf.logger.log(key=mlperf.tags.INPUT_DISTORTED_CROP_AREA_RANGE,
                      value=area_range)
    mlperf.logger.log(key=mlperf.tags.INPUT_DISTORTED_CROP_MAX_ATTEMPTS,
                      value=max_attempts)

    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
    if summary_verbosity >= 3:
      image = tf.image.decode_jpeg(image_buffer, channels=3,
                                   dct_method='INTEGER_FAST')
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      image_with_distorted_box = tf.image.draw_bounding_boxes(
          tf.expand_dims(image, 0), distort_bbox)
      tf.summary.image(
          'images_with_distorted_bounding_box',
          image_with_distorted_box)

    # Crop the image to the specified bounding box.
    if fuse_decode_and_crop:
      offset_y, offset_x, _ = tf.unstack(bbox_begin)
      target_height, target_width, _ = tf.unstack(bbox_size)
      crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
      image = tf.image.decode_and_crop_jpeg(
          image_buffer, crop_window, channels=3)
    else:
      image = tf.image.decode_jpeg(image_buffer, channels=3,
                                   dct_method='INTEGER_FAST')
      image = tf.slice(image, bbox_begin, bbox_size)

    mlperf.logger.log(key=mlperf.tags.INPUT_RANDOM_FLIP)
    distorted_image = tf.image.random_flip_left_right(image)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected.
    mlperf.logger.log(key=mlperf.tags.INPUT_RESIZE, value=[height, width])
    image_resize_method = get_image_resize_method(resize_method, batch_position)
    distorted_image = tf.image.resize_images(
        distorted_image, [height, width],
        image_resize_method,
        align_corners=False)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])
    if summary_verbosity >= 3:
      tf.summary.image('cropped_resized_maybe_flipped_image',
                       tf.expand_dims(distorted_image, 0))

    if distortions:
      distorted_image = tf.cast(distorted_image, dtype=tf.float32)
      # Images values are expected to be in [0,1] for color distortion.
      distorted_image /= 255.
      # Randomly distort the colors.
      distorted_image = distort_color(distorted_image, batch_position,
                                      distort_color_in_yiq=distort_color_in_yiq)

      # Note: This ensures the scaling matches the output of eval_image
      distorted_image *= 255

    if summary_verbosity >= 3:
      tf.summary.image(
          'final_distorted_image',
          tf.expand_dims(distorted_image, 0))
    return distorted_image


def distort_color(image, batch_position=0, distort_color_in_yiq=False,
                  scope=None):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops based on the position of the image in a batch.

  Args:
    image: float32 Tensor containing single image. Tensor values should be in
      range [0, 1].
    batch_position: the position of the image in a batch. NOTE: this argument
      can be an integer or a tensor
    distort_color_in_yiq: distort color of input images in YIQ space.
    scope: Optional scope for op_scope.
  Returns:
    color-distorted image
  """
  with tf.name_scope(scope or 'distort_color'):

    def distort_fn_0(image=image):
      """Variant 0 of distort function."""
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      if distort_color_in_yiq:
        image = distort_image_ops.random_hsv_in_yiq(
            image, lower_saturation=0.5, upper_saturation=1.5,
            max_delta_hue=0.2 * math.pi)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      return image

    def distort_fn_1(image=image):
      """Variant 1 of distort function."""
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      if distort_color_in_yiq:
        image = distort_image_ops.random_hsv_in_yiq(
            image, lower_saturation=0.5, upper_saturation=1.5,
            max_delta_hue=0.2 * math.pi)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      return image

    image = utils.smart_cond(batch_position % 2 == 0, distort_fn_0,
                             distort_fn_1)
    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


class InputPreprocessor(object):
  """Base class for all model preprocessors."""

  def __init__(self, batch_size, output_shapes):
    self.batch_size = batch_size
    self.output_shapes = output_shapes

  def supports_datasets(self):
    """Whether this preprocessor supports dataset."""
    return False

  def minibatch(self, dataset, subset, params, shift_ratio=-1):
    """Returns tensors representing a minibatch of all the input."""
    raise NotImplementedError('Must be implemented by subclass.')

  # The methods added below are only supported/used if supports_datasets()
  # returns True.
  # TODO(laigd): refactor benchmark_cnn.py and put the logic of
  # _build_input_processing() into InputPreprocessor.

  def parse_and_preprocess(self, value, batch_position):
    """Function to parse and preprocess an Example proto in input pipeline."""
    raise NotImplementedError('Must be implemented by subclass.')

  # TODO(laigd): figure out how to remove these parameters, since the
  # preprocessor itself has self.batch_size, self.num_splits, etc defined.
  def build_multi_device_iterator(self, batch_size, num_splits, cpu_device,
                                  params, gpu_devices, dataset, doing_eval):
    """Creates a MultiDeviceIterator."""
    assert self.supports_datasets()
    assert num_splits == len(gpu_devices)
    with tf.name_scope('batch_processing'):
      if doing_eval:
        subset = 'validation'
      else:
        subset = 'train'
      batch_size_per_split = batch_size // num_splits
      ds = self.create_dataset(
          batch_size,
          num_splits,
          batch_size_per_split,
          dataset,
          subset,
          train=(not doing_eval),
          datasets_repeat_cached_sample=params.datasets_repeat_cached_sample,
          num_threads=params.datasets_num_private_threads,
          datasets_use_caching=params.datasets_use_caching,
          datasets_parallel_interleave_cycle_length=(
              params.datasets_parallel_interleave_cycle_length),
          datasets_sloppy_parallel_interleave=(
              params.datasets_sloppy_parallel_interleave),
          datasets_parallel_interleave_prefetch=(
              params.datasets_parallel_interleave_prefetch))
      multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          ds,
          gpu_devices,
          source_device=cpu_device,
          max_buffer_size=params.multi_device_iterator_max_buffer_size)
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                           multi_device_iterator.initializer)
      return multi_device_iterator

  def create_dataset(self,
                     batch_size,
                     num_splits,
                     batch_size_per_split,
                     dataset,
                     subset,
                     train,
                     datasets_repeat_cached_sample,
                     num_threads=None,
                     datasets_use_caching=False,
                     datasets_parallel_interleave_cycle_length=None,
                     datasets_sloppy_parallel_interleave=False,
                     datasets_parallel_interleave_prefetch=None):
    """Creates a dataset for the benchmark."""
    raise NotImplementedError('Must be implemented by subclass.')

  def create_iterator(self, ds):
    ds_iterator = ds.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                         ds_iterator.initializer)
    return ds_iterator

  def minibatch_fn(self, batch_size, model_input_shapes, num_splits,
                   dataset, subset, train, datasets_repeat_cached_sample,
                   num_threads, datasets_use_caching,
                   datasets_parallel_interleave_cycle_length,
                   datasets_sloppy_parallel_interleave,
                   datasets_parallel_interleave_prefetch):
    """Returns a function and list of args for the fn to create a minibatch."""
    assert self.supports_datasets()
    batch_size_per_split = batch_size // num_splits
    assert batch_size_per_split == model_input_shapes[0][0]
    with tf.name_scope('batch_processing'):
      ds = self.create_dataset(batch_size, num_splits, batch_size_per_split,
                               dataset, subset, train,
                               datasets_repeat_cached_sample, num_threads,
                               datasets_use_caching,
                               datasets_parallel_interleave_cycle_length,
                               datasets_sloppy_parallel_interleave,
                               datasets_parallel_interleave_prefetch)
      ds_iterator = self.create_iterator(ds)

      ds_iterator_string_handle = ds_iterator.string_handle()

      @function.Defun(tf.string)
      def _fn(h):
        remote_iterator = tf.data.Iterator.from_string_handle(
            h, ds_iterator.output_types, ds_iterator.output_shapes)
        input_list = remote_iterator.get_next()
        reshaped_input_list = [
            tf.reshape(input_list[i], shape=model_input_shapes[i])
            for i in range(len(input_list))
        ]
        return reshaped_input_list

      return _fn, [ds_iterator_string_handle]


class BaseImagePreprocessor(InputPreprocessor):
  """Base class for all image model preprocessors."""

  def __init__(self,
               batch_size,
               output_shapes,
               num_splits,
               dtype,
               train,
               distortions,
               resize_method,
               shift_ratio=-1,
               summary_verbosity=0,
               distort_color_in_yiq=True,
               fuse_decode_and_crop=True,
               match_mlperf=False):
    super(BaseImagePreprocessor, self).__init__(batch_size, output_shapes)
    image_shape = output_shapes[0]
    # image_shape is in form (batch_size, height, width, depth)
    self.height = image_shape[1]
    self.width = image_shape[2]
    self.depth = image_shape[3]
    self.num_splits = num_splits
    self.dtype = dtype
    self.train = train
    self.resize_method = resize_method
    self.shift_ratio = shift_ratio
    self.distortions = distortions
    self.distort_color_in_yiq = distort_color_in_yiq
    self.fuse_decode_and_crop = fuse_decode_and_crop
    if self.batch_size % self.num_splits != 0:
      raise ValueError(
          ('batch_size must be a multiple of num_splits: '
           'batch_size %d, num_splits: %d') %
          (self.batch_size, self.num_splits))
    self.batch_size_per_split = self.batch_size // self.num_splits
    self.summary_verbosity = summary_verbosity
    self.match_mlperf = match_mlperf

  def parse_and_preprocess(self, value, batch_position):
    assert self.supports_datasets()
    image_buffer, label_index, bbox, _ = parse_example_proto(value)
    if self.match_mlperf:
      bbox = tf.zeros((1, 0, 4), dtype=bbox.dtype)
      mlperf.logger.log(key=mlperf.tags.INPUT_CROP_USES_BBOXES, value=False)
    else:
      mlperf.logger.log(key=mlperf.tags.INPUT_CROP_USES_BBOXES, value=True)
    image = self.preprocess(image_buffer, bbox, batch_position)
    return (image, label_index)

  def preprocess(self, image_buffer, bbox, batch_position):
    raise NotImplementedError('Must be implemented by subclass.')

  def create_dataset(self,
                     batch_size,
                     num_splits,
                     batch_size_per_split,
                     dataset,
                     subset,
                     train,
                     datasets_repeat_cached_sample,
                     num_threads=None,
                     datasets_use_caching=False,
                     datasets_parallel_interleave_cycle_length=None,
                     datasets_sloppy_parallel_interleave=False,
                     datasets_parallel_interleave_prefetch=None):
    """Creates a dataset for the benchmark."""
    assert self.supports_datasets()
    glob_pattern = dataset.tf_record_pattern(subset)
    file_names = gfile.Glob(glob_pattern)
    if not file_names:
      raise ValueError('Found no files in --data_dir matching: {}'
                       .format(glob_pattern))
    ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=train)
    ds = ds.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=datasets_parallel_interleave_cycle_length or 10,
            sloppy=datasets_sloppy_parallel_interleave,
            prefetch_input_elements=datasets_parallel_interleave_prefetch))
    if datasets_repeat_cached_sample:
      # Repeat a single sample element indefinitely to emulate memory-speed IO.
      ds = ds.take(1).cache().repeat()
    counter = tf.data.Dataset.range(batch_size)
    counter = counter.repeat()
    ds = tf.data.Dataset.zip((ds, counter))
    ds = ds.prefetch(buffer_size=batch_size)
    if datasets_use_caching:
      ds = ds.cache()
    if train:
      buffer_size = 10000
      mlperf.logger.log(key=mlperf.tags.INPUT_SHARD, value=buffer_size)
      ds = ds.apply(
          tf.data.experimental.shuffle_and_repeat(buffer_size=buffer_size))
    else:
      ds = ds.repeat()
    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            map_func=self.parse_and_preprocess,
            batch_size=batch_size_per_split,
            num_parallel_batches=num_splits))
    ds = ds.prefetch(buffer_size=num_splits)
    if num_threads:
      ds = threadpool.override_threadpool(
          ds,
          threadpool.PrivateThreadPool(
              num_threads, display_name='input_pipeline_thread_pool'))
    return ds


class RecordInputImagePreprocessor(BaseImagePreprocessor):
  """Preprocessor for images with RecordInput format."""

  def preprocess(self, image_buffer, bbox, batch_position):
    """Preprocessing image_buffer as a function of its batch position."""
    if self.train:
      image = train_image(image_buffer, self.height, self.width, bbox,
                          batch_position, self.resize_method, self.distortions,
                          None, summary_verbosity=self.summary_verbosity,
                          distort_color_in_yiq=self.distort_color_in_yiq,
                          fuse_decode_and_crop=self.fuse_decode_and_crop)
    else:
      image = tf.image.decode_jpeg(
          image_buffer, channels=3, dct_method='INTEGER_FAST')
      image = eval_image(image, self.height, self.width, batch_position,
                         self.resize_method,
                         summary_verbosity=self.summary_verbosity)
    # Note: image is now float32 [height,width,3] with range [0, 255]

    # image = tf.cast(image, tf.uint8) # HACK TESTING

    if self.match_mlperf:
      mlperf.logger.log(key=mlperf.tags.INPUT_MEAN_SUBTRACTION,
                        value=_CHANNEL_MEANS)
      normalized = image - _CHANNEL_MEANS
    else:
      normalized = normalized_image(image)
    return tf.cast(normalized, self.dtype)

  def minibatch(self,
                dataset,
                subset,
                params,
                shift_ratio=-1):
    if shift_ratio < 0:
      shift_ratio = self.shift_ratio
    with tf.name_scope('batch_processing'):
      # Build final results per split.
      images = [[] for _ in range(self.num_splits)]
      labels = [[] for _ in range(self.num_splits)]
      if params.use_datasets:
        ds = self.create_dataset(
            self.batch_size, self.num_splits, self.batch_size_per_split,
            dataset, subset, self.train,
            datasets_repeat_cached_sample=params.datasets_repeat_cached_sample,
            num_threads=params.datasets_num_private_threads,
            datasets_use_caching=params.datasets_use_caching,
            datasets_parallel_interleave_cycle_length=(
                params.datasets_parallel_interleave_cycle_length),
            datasets_sloppy_parallel_interleave=(
                params.datasets_sloppy_parallel_interleave),
            datasets_parallel_interleave_prefetch=(
                params.datasets_parallel_interleave_prefetch))
        ds_iterator = self.create_iterator(ds)
        for d in xrange(self.num_splits):
          images[d], labels[d] = ds_iterator.get_next()

      # TODO(laigd): consider removing the --use_datasets option, it should
      # always use datasets.
      else:
        record_input = data_flow_ops.RecordInput(
            file_pattern=dataset.tf_record_pattern(subset),
            seed=301,
            parallelism=64,
            buffer_size=10000,
            batch_size=self.batch_size,
            shift_ratio=shift_ratio,
            name='record_input')
        records = record_input.get_yield_op()
        records = tf.split(records, self.batch_size, 0)
        records = [tf.reshape(record, []) for record in records]
        for idx in xrange(self.batch_size):
          value = records[idx]
          (image, label) = self.parse_and_preprocess(value, idx)
          split_index = idx % self.num_splits
          labels[split_index].append(label)
          images[split_index].append(image)

      for split_index in xrange(self.num_splits):
        if not params.use_datasets:
          images[split_index] = tf.parallel_stack(images[split_index])
          labels[split_index] = tf.concat(labels[split_index], 0)
        images[split_index] = tf.reshape(
            images[split_index],
            shape=[self.batch_size_per_split, self.height, self.width,
                   self.depth])
        labels[split_index] = tf.reshape(labels[split_index],
                                         [self.batch_size_per_split])
      return images, labels

  def supports_datasets(self):
    return True


class ImagenetPreprocessor(RecordInputImagePreprocessor):

  def preprocess(self, image_buffer, bbox, batch_position):
    # pylint: disable=g-import-not-at-top
    try:
      from tensorflow_models.official.resnet.imagenet_preprocessing import preprocess_image
    except ImportError:
      tf.logging.fatal('Please include tensorflow/models to the PYTHONPATH.')
      raise
    if self.train:
      image = preprocess_image(
          image_buffer, bbox, self.height, self.width, self.depth,
          is_training=True)
    else:
      image = preprocess_image(
          image_buffer, bbox, self.height, self.width, self.depth,
          is_training=False)
    return tf.cast(image, self.dtype)


class Cifar10ImagePreprocessor(BaseImagePreprocessor):
  """Preprocessor for Cifar10 input images."""

  def _distort_image(self, image):
    """Distort one image for training a network.

    Adopted the standard data augmentation scheme that is widely used for
    this dataset: the images are first zero-padded with 4 pixels on each side,
    then randomly cropped to again produce distorted images; half of the images
    are then horizontally mirrored.

    Args:
      image: input image.
    Returns:
      distorted image.
    """
    image = tf.image.resize_image_with_crop_or_pad(
        image, self.height + 8, self.width + 8)
    distorted_image = tf.random_crop(image,
                                     [self.height, self.width, self.depth])
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    if self.summary_verbosity >= 3:
      tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))
    return distorted_image

  def _eval_image(self, image):
    """Get the image for model evaluation."""
    distorted_image = tf.image.resize_image_with_crop_or_pad(
        image, self.width, self.height)
    if self.summary_verbosity >= 3:
      tf.summary.image('cropped.image', tf.expand_dims(distorted_image, 0))
    return distorted_image

  def preprocess(self, raw_image):
    """Preprocessing raw image."""
    if self.summary_verbosity >= 3:
      tf.summary.image('raw.image', tf.expand_dims(raw_image, 0))
    if self.train and self.distortions:
      image = self._distort_image(raw_image)
    else:
      image = self._eval_image(raw_image)
    normalized = normalized_image(image)
    return tf.cast(normalized, self.dtype)

  def minibatch(self,
                dataset,
                subset,
                params,
                shift_ratio=-1):
    # TODO(jsimsa): Implement datasets code path
    del shift_ratio, params
    with tf.name_scope('batch_processing'):
      all_images, all_labels = dataset.read_data_files(subset)
      all_images = tf.constant(all_images)
      all_labels = tf.constant(all_labels)
      input_image, input_label = tf.train.slice_input_producer(
          [all_images, all_labels])
      input_image = tf.cast(input_image, self.dtype)
      input_label = tf.cast(input_label, tf.int32)
      # Ensure that the random shuffling has good mixing properties.
      min_fraction_of_examples_in_queue = 0.4
      min_queue_examples = int(dataset.num_examples_per_epoch(subset) *
                               min_fraction_of_examples_in_queue)
      raw_images, raw_labels = tf.train.shuffle_batch(
          [input_image, input_label], batch_size=self.batch_size,
          capacity=min_queue_examples + 3 * self.batch_size,
          min_after_dequeue=min_queue_examples)

      images = [[] for i in range(self.num_splits)]
      labels = [[] for i in range(self.num_splits)]

      # Create a list of size batch_size, each containing one image of the
      # batch. Without the unstack call, raw_images[i] would still access the
      # same image via a strided_slice op, but would be slower.
      raw_images = tf.unstack(raw_images, axis=0)
      raw_labels = tf.unstack(raw_labels, axis=0)
      for i in xrange(self.batch_size):
        split_index = i % self.num_splits
        # The raw image read from data has the format [depth, height, width]
        # reshape to the format returned by minibatch.
        raw_image = tf.reshape(raw_images[i],
                               [dataset.depth, dataset.height, dataset.width])
        raw_image = tf.transpose(raw_image, [1, 2, 0])
        image = self.preprocess(raw_image)
        images[split_index].append(image)

        labels[split_index].append(raw_labels[i])

      for split_index in xrange(self.num_splits):
        images[split_index] = tf.parallel_stack(images[split_index])
        labels[split_index] = tf.parallel_stack(labels[split_index])
      return images, labels


class COCOPreprocessor(BaseImagePreprocessor):
  """Preprocessor for COCO dataset input images, boxes, and labels."""

  def minibatch(self,
                dataset,
                subset,
                params,
                shift_ratio=-1):
    del shift_ratio  # Not used when using datasets instead of data_flow_ops
    with tf.name_scope('batch_processing'):
      ds = self.create_dataset(
          self.batch_size, self.num_splits, self.batch_size_per_split,
          dataset, subset, self.train, params.datasets_repeat_cached_sample)
      ds_iterator = self.create_iterator(ds)

      # Training data: 4 tuple
      # Validation data: 5 tuple
      # See get_input_shapes in models/ssd_model.py for details.
      input_len = 4 if subset == 'train' else 5
      input_lists = [[None for _ in range(self.num_splits)]
                     for _ in range(input_len)]
      for d in xrange(self.num_splits):
        input_list = ds_iterator.get_next()
        for i in range(input_len):
          input_lists[i][d] = input_list[i]
      return input_lists

  def preprocess(self, data):
    try:
      from cnn_quantization.tf_cnn_benchmarks import ssd_dataloader  # pylint: disable=g-import-not-at-top
      from cnn_quantization.tf_cnn_benchmarks import ssd_constants  # pylint: disable=g-import-not-at-top
      from tensorflow_models.object_detection.core import preprocessor  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise ImportError('To use the COCO dataset, you must clone the '
                        'repo https://github.com/tensorflow/models and add '
                        'tensorflow/models and tensorflow/models/research to '
                        'the PYTHONPATH, and compile the protobufs')
    image_buffer = data['image_buffer']
    boxes = data['groundtruth_boxes']
    classes = tf.reshape(data['groundtruth_classes'], [-1, 1])
    source_id = tf.string_to_number(data['source_id'])
    raw_shape = data['raw_shape']

    ssd_encoder = ssd_dataloader.Encoder()

    # Only 80 of the 90 COCO classes are used.
    class_map = tf.convert_to_tensor(ssd_constants.CLASS_MAP)
    classes = tf.gather(class_map, classes)
    classes = tf.cast(classes, dtype=tf.float32)

    if self.train:
      image, boxes, classes = ssd_dataloader.ssd_decode_and_crop(
          image_buffer, boxes, classes, raw_shape)
      # ssd_crop resizes and returns image of dtype float32 and does not change
      # its range (i.e., value in between 0--255). Divide by 255. converts it
      # to [0, 1] range. Not doing this before cropping to avoid dtype cast
      # (which incurs additional memory copy).
      image /= 255.

      image, boxes = preprocessor.random_horizontal_flip(
          image=image, boxes=boxes)
      # Random horizontal flip probability is 50%
      # See https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py  # pylint: disable=line-too-long
      mlperf.logger.log(key=mlperf.tags.RANDOM_FLIP_PROBABILITY, value=0.5)

      image = ssd_dataloader.color_jitter(
          image, brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05)
      image = ssd_dataloader.normalize_image(image)
      image = tf.cast(image, self.dtype)

      encoded_returns = ssd_encoder.encode_labels(boxes, classes)
      encoded_classes, encoded_boxes, num_matched_boxes = encoded_returns

      # Shape of image: [width, height, channel]
      # Shape of encoded_boxes: [NUM_SSD_BOXES, 4]
      # Shape of encoded_classes: [NUM_SSD_BOXES, 1]
      # Shape of num_matched_boxes: [1]
      return (image, encoded_boxes, encoded_classes, num_matched_boxes)

    else:
      image = tf.image.decode_jpeg(image_buffer)
      image = tf.image.resize_images(
          image, size=(ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE))
      # resize_image returns image of dtype float32 and does not change its
      # range. Divide by 255 to convert image to [0, 1] range.
      image /= 255.

      image = ssd_dataloader.normalize_image(image)
      image = tf.cast(image, self.dtype)

      def trim_and_pad(inp_tensor):
        """Limit the number of boxes, and pad if necessary."""
        inp_tensor = inp_tensor[:ssd_constants.MAX_NUM_EVAL_BOXES]
        num_pad = ssd_constants.MAX_NUM_EVAL_BOXES - tf.shape(inp_tensor)[0]
        inp_tensor = tf.pad(inp_tensor, [[0, num_pad], [0, 0]])
        return tf.reshape(inp_tensor, [ssd_constants.MAX_NUM_EVAL_BOXES,
                                       inp_tensor.get_shape()[1]])

      boxes, classes = trim_and_pad(boxes), trim_and_pad(classes)

      # Shape of boxes: [MAX_NUM_EVAL_BOXES, 4]
      # Shape of classes: [MAX_NUM_EVAL_BOXES, 1]
      # Shape of source_id: [] (scalar tensor)
      # Shape of raw_shape: [3]
      return (image, boxes, classes, source_id, raw_shape)

  def create_dataset(self,
                     batch_size,
                     num_splits,
                     batch_size_per_split,
                     dataset,
                     subset,
                     train,
                     datasets_repeat_cached_sample,
                     num_threads=None,
                     datasets_use_caching=False,
                     datasets_parallel_interleave_cycle_length=None,
                     datasets_sloppy_parallel_interleave=False,
                     datasets_parallel_interleave_prefetch=None):
    """Creates a dataset for the benchmark."""
    try:
      from cnn_quantization.tf_cnn_benchmarks import ssd_dataloader  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise ImportError('To use the COCO dataset, you must clone the '
                        'repo https://github.com/tensorflow/models and add '
                        'tensorflow/models and tensorflow/models/research to '
                        'the PYTHONPATH, and compile the protobufs')
    assert self.supports_datasets()

    glob_pattern = dataset.tf_record_pattern(subset)
    ds = tf.data.TFRecordDataset.list_files(glob_pattern, shuffle=train)
    # TODO(haoyuzhang): Enable map+filter fusion after cl/218399112 in release
    # options = tf.data.Options()
    # options.experimental_optimization = tf.data.experimental.OptimizationOptions()  # pylint: disable=line-too-long
    # options.experimental_optimization.map_and_filter_fusion = True
    # ds = ds.with_options(options)

    ds = ds.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=datasets_parallel_interleave_cycle_length or 10,
            sloppy=datasets_sloppy_parallel_interleave))
    mlperf.logger.log(key=mlperf.tags.INPUT_ORDER)
    if datasets_repeat_cached_sample:
      # Repeat a single sample element indefinitely to emulate memory-speed IO.
      ds = ds.take(1).cache().repeat()
    ds = ds.prefetch(buffer_size=batch_size)
    if datasets_use_caching:
      ds = ds.cache()
    if train:
      ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10000))
      mlperf.logger.log(key=mlperf.tags.INPUT_SHARD, value=10000)
      mlperf.logger.log(key=mlperf.tags.INPUT_ORDER)
    else:
      ds = ds.repeat()

    ds = ds.map(ssd_dataloader.ssd_parse_example_proto, num_parallel_calls=64)
    ds = ds.filter(
        lambda data: tf.greater(tf.shape(data['groundtruth_boxes'])[0], 0))
    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            map_func=self.preprocess,
            batch_size=batch_size_per_split,
            num_parallel_batches=num_splits,
            drop_remainder=train))
    ds = ds.prefetch(buffer_size=num_splits)
    if num_threads:
      ds = threadpool.override_threadpool(
          ds,
          threadpool.PrivateThreadPool(
              num_threads, display_name='input_pipeline_thread_pool'))
    return ds

  def supports_datasets(self):
    return True


class TestImagePreprocessor(BaseImagePreprocessor):
  """Preprocessor used for testing.

  set_fake_data() sets which images and labels will be output by minibatch(),
  and must be called before minibatch(). This allows tests to easily specify
  a set of images to use for training, without having to create any files.

  Queue runners must be started for this preprocessor to work.
  """

  def __init__(self,
               batch_size,
               output_shapes,
               num_splits,
               dtype,
               train=None,
               distortions=None,
               resize_method=None,
               shift_ratio=0,
               summary_verbosity=0,
               distort_color_in_yiq=False,
               fuse_decode_and_crop=False,
               match_mlperf=False):
    super(TestImagePreprocessor, self).__init__(
        batch_size, output_shapes, num_splits, dtype, train, distortions,
        resize_method, shift_ratio, summary_verbosity=summary_verbosity,
        distort_color_in_yiq=distort_color_in_yiq,
        fuse_decode_and_crop=fuse_decode_and_crop, match_mlperf=match_mlperf)
    self.expected_subset = None

  def set_fake_data(self, fake_images, fake_labels):
    assert len(fake_images.shape) == 4
    assert len(fake_labels.shape) == 1
    num_images = fake_images.shape[0]
    assert num_images == fake_labels.shape[0]
    assert num_images % self.batch_size == 0
    self.fake_images = fake_images
    self.fake_labels = fake_labels

  def minibatch(self,
                dataset,
                subset,
                params,
                shift_ratio=0):
    """Get test image batches."""
    del dataset, params
    if (not hasattr(self, 'fake_images') or
        not hasattr(self, 'fake_labels')):
      raise ValueError('Must call set_fake_data() before calling minibatch '
                       'on TestImagePreprocessor')
    if self.expected_subset is not None:
      assert subset == self.expected_subset

    shift_ratio = shift_ratio or self.shift_ratio
    fake_images = cnn_util.roll_numpy_batches(self.fake_images, self.batch_size,
                                              shift_ratio)
    fake_labels = cnn_util.roll_numpy_batches(self.fake_labels, self.batch_size,
                                              shift_ratio)

    with tf.name_scope('batch_processing'):
      image_slice, label_slice = tf.train.slice_input_producer(
          [fake_images, fake_labels],
          shuffle=False,
          name='image_slice')
      raw_images, raw_labels = tf.train.batch(
          [image_slice, label_slice], batch_size=self.batch_size,
          name='image_batch')
      images = [[] for _ in range(self.num_splits)]
      labels = [[] for _ in range(self.num_splits)]
      for i in xrange(self.batch_size):
        split_index = i % self.num_splits
        raw_image = tf.cast(raw_images[i], self.dtype)
        images[split_index].append(raw_image)
        labels[split_index].append(raw_labels[i])
      for split_index in xrange(self.num_splits):
        images[split_index] = tf.parallel_stack(images[split_index])
        labels[split_index] = tf.parallel_stack(labels[split_index])

      normalized = [normalized_image(part) for part in images]
      return [[tf.cast(part, self.dtype) for part in normalized], labels]


class LibrispeechPreprocessor(InputPreprocessor):
  """Preprocessor for librispeech class for all image model preprocessors."""

  def __init__(self, batch_size, output_shapes, num_splits, dtype, train,
               **kwargs):
    del kwargs
    super(LibrispeechPreprocessor, self).__init__(batch_size, output_shapes)
    self.num_splits = num_splits
    self.dtype = dtype
    self.is_train = train
    if self.batch_size % self.num_splits != 0:
      raise ValueError(('batch_size must be a multiple of num_splits: '
                        'batch_size %d, num_splits: %d') % (self.batch_size,
                                                            self.num_splits))
    self.batch_size_per_split = self.batch_size // self.num_splits

  def create_dataset(self,
                     batch_size,
                     num_splits,
                     batch_size_per_split,
                     dataset,
                     subset,
                     train,
                     datasets_repeat_cached_sample,
                     num_threads=None,
                     datasets_use_caching=False,
                     datasets_parallel_interleave_cycle_length=None,
                     datasets_sloppy_parallel_interleave=False,
                     datasets_parallel_interleave_prefetch=None):
    """Creates a dataset for the benchmark."""
    # TODO(laigd): currently the only difference between this and the one in
    # BaseImagePreprocessor is, this uses map() and padded_batch() while the
    # latter uses tf.data.experimental.map_and_batch(). Try to merge them.
    assert self.supports_datasets()
    glob_pattern = dataset.tf_record_pattern(subset)
    file_names = gfile.Glob(glob_pattern)
    if not file_names:
      raise ValueError('Found no files in --data_dir matching: {}'
                       .format(glob_pattern))
    ds = tf.data.TFRecordDataset.list_files(file_names, shuffle=train)
    ds = ds.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=datasets_parallel_interleave_cycle_length or 10,
            sloppy=datasets_sloppy_parallel_interleave,
            prefetch_input_elements=datasets_parallel_interleave_prefetch))
    if datasets_repeat_cached_sample:
      # Repeat a single sample element indefinitely to emulate memory-speed IO.
      ds = ds.take(1).cache().repeat()
    counter = tf.data.Dataset.range(batch_size)
    counter = counter.repeat()
    ds = tf.data.Dataset.zip((ds, counter))
    ds = ds.prefetch(buffer_size=batch_size)
    if datasets_use_caching:
      ds = ds.cache()
    if train:
      ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10000))
    else:
      ds = ds.repeat()
    ds = ds.map(map_func=self.parse_and_preprocess,
                num_parallel_calls=batch_size_per_split*num_splits)
    ds = ds.padded_batch(
        batch_size=batch_size_per_split,
        padded_shapes=tuple([
            tf.TensorShape(output_shape[1:])
            for output_shape in self.output_shapes
        ]),
        drop_remainder=True)
    ds = ds.prefetch(buffer_size=num_splits)
    if num_threads:
      ds = threadpool.override_threadpool(
          ds,
          threadpool.PrivateThreadPool(
              num_threads, display_name='input_pipeline_thread_pool'))
    return ds

  def minibatch(self, dataset, subset, params, shift_ratio=-1):
    assert params.use_datasets
    # TODO(laigd): unify this with CNNModel's minibatch()
    # TODO(laigd): in distributed mode we use shift_ratio so different workers
    # won't work on same inputs, so we should respect that.
    del shift_ratio
    with tf.name_scope('batch_processing'):
      ds = self.create_dataset(
          self.batch_size,
          self.num_splits,
          self.batch_size_per_split,
          dataset,
          subset,
          self.is_train,
          datasets_repeat_cached_sample=params.datasets_repeat_cached_sample,
          num_threads=params.datasets_num_private_threads,
          datasets_use_caching=params.datasets_use_caching,
          datasets_parallel_interleave_cycle_length=(
              params.datasets_parallel_interleave_cycle_length),
          datasets_sloppy_parallel_interleave=(
              params.datasets_sloppy_parallel_interleave),
          datasets_parallel_interleave_prefetch=(
              params.datasets_parallel_interleave_prefetch))
      ds_iterator = self.create_iterator(ds)

      # The four lists are: input spectrogram feature, labels, input lengths,
      # label lengths
      input_lists = [[None for _ in range(self.num_splits)] for _ in range(4)]
      for d in xrange(self.num_splits):
        input_list = ds_iterator.get_next()
        for i in range(4):
          input_lists[i][d] = input_list[i]

      assert self.output_shapes == [
          input_lists[i][0].shape.as_list() for i in range(4)
      ]
      return tuple(input_lists)

  def supports_datasets(self):
    return True

  def parse_and_preprocess(self, value, batch_position):
    """Parse an TFRecord."""
    del batch_position
    assert self.supports_datasets()
    context_features = {
        'labels': tf.VarLenFeature(dtype=tf.int64),
        'input_length': tf.FixedLenFeature([], dtype=tf.int64),
        'label_length': tf.FixedLenFeature([], dtype=tf.int64),
    }
    sequence_features = {
        'features': tf.FixedLenSequenceFeature([161], dtype=tf.float32)
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=value,
        context_features=context_features,
        sequence_features=sequence_features,
    )

    return [
        # Input
        tf.expand_dims(sequence_parsed['features'], axis=2),
        # Label
        tf.cast(
            tf.reshape(
                tf.sparse_tensor_to_dense(context_parsed['labels']), [-1]),
            dtype=tf.int32),
        # Input length
        tf.cast(
            tf.reshape(context_parsed['input_length'], [1]),
            dtype=tf.int32),
        # Label length
        tf.cast(
            tf.reshape(context_parsed['label_length'], [1]),
            dtype=tf.int32),
    ]

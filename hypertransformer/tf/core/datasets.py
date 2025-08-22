# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Task generator using basic datasets."""
import dataclasses
import functools

from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_addons import image as tfi
import tree

from hypertransformer.tf.core import common_ht

DatasetInfo = common_ht.DatasetInfo

UseLabelSubset = Union[List[int], Callable[[], List[int]]]
LabelGenerator = Generator[Tuple[int, int], None, None]
SupervisedSamples = Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
SupervisedTensors = Tuple[np.ndarray, np.ndarray]
DSBatch = Tuple[tf.Tensor, tf.Tensor, tf.Operation]


# Range of the random alpha value controlling contrast enhancement augmentation.
ALPHA_MIN = 0.5
ALPHA_MAX = 2.5
# Maximum resize fraction in the resize augmentation (0.3 means 70%-100% of the
# original size).
MAX_RESIZE = 0.3

# Default image augmentation parameters
HUE_MAX_DELTA = 0.2
BRIGHTNESS_MAX_DELTA = 0.3
LOWER_CONTRAST = 0.8
UPPER_CONTRAST = 1.2
MIN_CROP_FRACTION = 0.6

_resize_methods = {
    tf.image.ResizeMethod.NEAREST_NEIGHBOR: 'nearest',
    tf.image.ResizeMethod.BILINEAR: 'bilinear',
    'nearest': 'nearest',
    'bilinear': 'bilinear',
}


def get_random_bounding_box(
    hw,
    aspect_ratio=1,
    min_crop_fraction=0.5,
    new_height=None,
    dtype=tf.int32):
  """Random bounding box with aspect_ratio and within (0,0,hw[:,0],hw[:,1])."""
  hw = tf.to_float(hw)
  h = hw[Ellipsis, 0]
  w = hw[Ellipsis, 1]
  if new_height is None:
    min_height = tf.minimum(h * min_crop_fraction,
                            w * min_crop_fraction / aspect_ratio)
    max_height = tf.minimum(h, w / aspect_ratio)
    new_h = tf.random.uniform(
        shape=h.shape.as_list(),
        minval=min_height,
        maxval=max_height,
        dtype=tf.float32)
  else:
    new_h = tf.to_float(new_height)
  new_hw = tf.stack([new_h, new_h / aspect_ratio], axis=-1)
  if dtype in [tf.int32, tf.int64]:
    new_hw = tf.round(new_hw)

  minval = tf.zeros_like(hw)
  maxval = hw - new_hw
  tl = tf.random.uniform(shape=hw.shape, minval=minval,
                         maxval=maxval, dtype=tf.float32)
  if dtype in [tf.int32, tf.int64]:
    tl = tf.round(tl)

  br = tl + new_hw
  boxes = tf.concat((tl, br), axis=-1)
  if dtype in [tf.int32, tf.int64]:
    boxes = tf.round(boxes)
  return tf.cast(boxes, dtype=dtype)


def crop_and_resize(images,
                    bboxes,
                    target_size,
                    methods=tf.image.ResizeMethod.BILINEAR,
                    extrapolation_value=0):
  """Does crop and resize given normalized boxes."""
  bboxes = tf.cast(bboxes, tf.float32)
  if not isinstance(target_size, (tuple, list)):
    target_size = [target_size, target_size]
  def resize_fn(images, resize_method, bboxes):
    """Resizes images according to bboxes and resize method."""
    squeeze = False
    if images.shape.rank == 3:
      images = images[None, Ellipsis]
      bboxes = bboxes[None, Ellipsis]
      squeeze = True
    if callable(resize_method):
      r = resize_method(
          images,
          boxes=bboxes,
          crop_size=target_size,
          extrapolation_value=extrapolation_value)
    else:
      r = tf.image.crop_and_resize(
          images,
          boxes=bboxes,
          method=_resize_methods[resize_method],
          crop_size=target_size,
          box_ind=tf.range(tf.shape(images)[0]),
          extrapolation_value=extrapolation_value)
    if squeeze:
      r = r[0]
    return r

  args = [images]
  if tree.is_nested(methods):
    args.append(methods)
  else:
    resize_fn = functools.partial(resize_fn, resize_method=methods)
  resize_fn = functools.partial(resize_fn, bboxes=bboxes)
  return tree.map_structure(resize_fn, *args)


def random_crop_and_resize(
    images, target_size,
    min_crop_fraction=0.5,
    crop_size=None,
    methods=tf.image.ResizeMethod.BILINEAR):
  """All tensors are cropped to the same size and resized to target size."""
  if not isinstance(target_size, (tuple, list)):
    target_size = [target_size, target_size]
  aspect_ratio = target_size[0] / target_size[1]
  batch_size = tf.shape(tree.flatten(images)[0])[0]
  bboxes = get_random_bounding_box(
      hw=tf.ones((batch_size, 2)),
      aspect_ratio=aspect_ratio, min_crop_fraction=min_crop_fraction,
      new_height=crop_size, dtype=tf.float32)
  return crop_and_resize(images, bboxes, target_size, methods)


def augment_images(images,
                   image_size,
                   augment_individually = False,
                   hue_delta = None,
                   brightness_delta = None,
                   contrast = None,
                   min_crop_fraction = None):
  """Standard image augmentation."""
  assert image_size is not None
  if hue_delta is None:
    hue_delta = HUE_MAX_DELTA
  if brightness_delta is None:
    brightness_delta = BRIGHTNESS_MAX_DELTA
  if contrast is None:
    contrast = (LOWER_CONTRAST, UPPER_CONTRAST)
  if min_crop_fraction is None:
    min_crop_fraction = MIN_CROP_FRACTION

  images = tf.cast((images + 1.0) * 128.0, tf.uint8)

  def _augment(image_tensor):
    out = tf.image.random_flip_left_right(image_tensor)
    out = tf.image.random_hue(out, max_delta=hue_delta)
    out = tf.image.random_brightness(out, max_delta=brightness_delta)
    return tf.image.random_contrast(out, lower=contrast[0], upper=contrast[1])

  if augment_individually:
    batch_size = int(images.shape[0])
    aug_images = [_augment(i) for i in tf.split(images, batch_size, axis=0)]
    images = tf.concat(aug_images, axis=0)
  else:
    images = _augment(images)

  images = tf.cast(images, tf.float32) / 128.0 - 1.0
  return random_crop_and_resize(images, image_size,
                                min_crop_fraction=min_crop_fraction)


def get_dataset_info(dataset_name):
  """Returns basic information about the dataset."""
  if dataset_name == 'emnist':
    return DatasetInfo(num_labels=62, num_samples_per_label=1500,
                       transpose_images=True)
  elif dataset_name == 'fashion_mnist':
    return DatasetInfo(num_labels=10, num_samples_per_label=3000,
                       transpose_images=False)
  elif dataset_name == 'kmnist':
    return DatasetInfo(num_labels=10, num_samples_per_label=3000,
                       transpose_images=False)
  elif dataset_name == 'omniglot':
    return DatasetInfo(num_labels=1623, num_samples_per_label=20,
                       transpose_images=False)
  elif dataset_name == 'miniimagenet':
    return DatasetInfo(num_labels=100, num_samples_per_label=600,
                       transpose_images=False)
  elif dataset_name == 'tieredimagenet':
    return DatasetInfo(num_labels=608,
                       # This dataset has a variable number of samples per class
                       num_samples_per_label=None,
                       transpose_images=False)
  else:
    raise ValueError(f'Dataset "{dataset_name}" is not supported.')


@dataclasses.dataclass
class RandomizedAugmentationConfig:
  """Specification of random augmentations applied to the images."""
  rotation_probability: float = 0.5
  smooth_probability: float = 0.3
  contrast_probability: float = 0.3
  resize_probability: float = 0.0
  negate_probability: float = 0.0
  roll_probability: float = 0.0
  angle_range: float = 180.0
  roll_range: float = 0.3
  rotate_by_90: bool = False


class RandomValue:
  """Random value."""

  value: Optional[tf.Variable] = None

  def create(self, name, dtype=tf.bool):
    """Create a random value of a given type."""
    if self.value is None:
      self.value = tf.get_variable(name, shape=(), dtype=dtype, trainable=False)

  def _random_bool(self, prob):
    return tf.math.less(tf.random.uniform(shape=(), maxval=1.0), prob)

  def assign_bool(self, prob):
    """Operator assigning a random boolean value to `value`."""
    return tf.assign(self.value, self._random_bool(prob))

  def assign_uniform(self, scale):
    """Operator assigning a random uniform value to `value`."""
    rand = tf.random.uniform(shape=(), minval=-1.0, maxval=1.0) * scale
    return tf.assign(self.value, rand)


@dataclasses.dataclass
class AugmentationConfig:
  """Configuration of the image augmentation generator."""
  random_config: Optional[RandomizedAugmentationConfig] = None
  rotate: RandomValue = dataclasses.field(default_factory=RandomValue)
  smooth: RandomValue = dataclasses.field(default_factory=RandomValue)
  contrast: RandomValue = dataclasses.field(default_factory=RandomValue)
  negate: RandomValue = dataclasses.field(default_factory=RandomValue)
  roll: RandomValue = dataclasses.field(default_factory=RandomValue)
  resize: RandomValue = dataclasses.field(default_factory=RandomValue)
  angle: RandomValue = dataclasses.field(default_factory=RandomValue)
  alpha: RandomValue = dataclasses.field(default_factory=RandomValue)
  size: RandomValue = dataclasses.field(default_factory=RandomValue)
  roll_x: RandomValue = dataclasses.field(default_factory=RandomValue)
  roll_y: RandomValue = dataclasses.field(default_factory=RandomValue)
  rotate_90_times: RandomValue = dataclasses.field(default_factory=RandomValue)

  children: List['AugmentationConfig'] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    if self.children:
      return
    with tf.variable_scope(None, default_name='augmentation'):
      for name in ['rotate', 'smooth', 'contrast', 'negate', 'resize', 'roll']:
        getattr(self, name).create(name)
      for name in ['angle', 'alpha', 'size', 'roll_x', 'roll_y',
                   'rotate_90_times']:
        getattr(self, name).create(name, tf.float32)

  def randomize_op(self):
    """Randomizes the augmentation according to the `random_config`."""
    if self.children:
      return tf.group([child.randomize_op() for child in self.children])
    if self.random_config is None:
      return tf.no_op()
    config = self.random_config
    assign_rotate = self.rotate.assign_bool(config.rotation_probability)
    assign_smooth = self.smooth.assign_bool(config.smooth_probability)
    assign_contrast = self.contrast.assign_bool(config.contrast_probability)
    assign_negate = self.negate.assign_bool(config.negate_probability)
    assign_resize = self.resize.assign_bool(config.resize_probability)
    assign_roll = self.roll.assign_bool(config.roll_probability)
    angle_range = config.angle_range / 180.0 * np.pi
    assign_angle = self.angle.assign_uniform(scale=angle_range)
    assign_size = self.size.assign_uniform(scale=1.0)
    assign_alpha = self.alpha.assign_uniform(scale=1.0)
    assign_roll_x = self.roll_x.assign_uniform(scale=config.roll_range)
    assign_roll_y = self.roll_y.assign_uniform(scale=config.roll_range)
    assign_rotate_90 = self.rotate_90_times.assign_uniform(scale=2.0)
    return tf.group(assign_rotate, assign_smooth, assign_contrast, assign_angle,
                    assign_negate, assign_resize, assign_alpha, assign_size,
                    assign_roll, assign_roll_x, assign_roll_y, assign_rotate_90)

  def _normalize(self, image):
    v_min = tf.reduce_min(image, axis=(1, 2), keepdims=True)
    v_max = tf.reduce_max(image, axis=(1, 2), keepdims=True)
    return (image - v_min) / (v_max - v_min + 1e-7)

  def _aug_contrast(self, image):
    """Increses the image contrast."""
    normalized = image / 128.0 - 1
    v_mean = tf.reduce_mean(normalized, axis=(1, 2), keepdims=True)
    mult, shift = (ALPHA_MAX - ALPHA_MIN) / 2, (ALPHA_MAX + ALPHA_MIN) / 2
    alpha = mult * self.alpha.value + shift
    output = tf.math.tanh((normalized - v_mean) * alpha) / alpha
    output += v_mean
    return 255.0 * self._normalize(output)

  def _aug_smooth(self, images):
    """Smooths the image using a 5x5 uniform kernel."""
    depth = int(images.shape[-1])
    image_filter = tf.eye(num_rows=depth, num_columns=depth, dtype=tf.float32)
    image_filter = tf.stack([image_filter] * 3, axis=0)
    image_filter = tf.stack([image_filter] * 3, axis=0)
    output = tf.nn.conv2d(images, image_filter / 9.0, padding='SAME')
    return 255.0 * self._normalize(output)

  def _aug_roll(self, images):
    """Smooths the image using a 5x5 uniform kernel."""
    width, height = images.shape[1], images.shape[2]
    width, height = tf.cast(width, tf.float32), tf.cast(height, tf.float32)
    x = tf.cast(self.roll_x.value * width, tf.int32)
    y = tf.cast(self.roll_y.value * height, tf.int32)
    return tf.roll(images, [x, y], axis=[1, 2])

  def _aug_negate(self, images):
    """Negates the image."""
    return 255.0 - images

  def _aug_resize(self, images):
    """Resizes the image."""
    size = int(images.shape[1])
    re_size = size * (1 - MAX_RESIZE / 2 + MAX_RESIZE * self.size.value / 2)
    images = tf.image.resize(images, [re_size, re_size])
    return tf.image.resize_with_crop_or_pad(images, size, size)

  def _aug_rotate_90(self, images):
    num_rotations = tf.cast(tf.math.floor(self.rotate_90_times.value + 2.0),
                            tf.int32)
    return tf.image.rot90(images, k=num_rotations)

  def process(self, images, index = None
              ):
    """Processes a batch of samples."""
    if index is not None:
      return self.children[index].process(images)
    config = self.random_config
    if config is None:
      raise ValueError('AugmentationConfig is undefined.')
    if config.rotate_by_90:
      images = self._aug_rotate_90(images)
    if config.rotation_probability > 0.0:
      images = tf.cond(self.rotate.value,
                       lambda: tfi.rotate(images, self.angle.value),
                       lambda: tf.identity(images))
    if config.roll_probability > 0.0:
      images = tf.cond(self.roll.value,
                       lambda: self._aug_roll(images),
                       lambda: tf.identity(images))
    if config.resize_probability > 0.0:
      images = tf.cond(self.resize.value,
                       lambda: self._aug_resize(images),
                       lambda: tf.identity(images))
    if config.contrast_probability > 0.0:
      images = tf.cond(self.contrast.value,
                       lambda: self._aug_contrast(images),
                       lambda: tf.identity(images))
    if config.smooth_probability > 0.0:
      images = tf.cond(self.smooth.value,
                       lambda: self._aug_smooth(images),
                       lambda: tf.identity(images))
    if config.negate_probability > 0.0:
      images = tf.cond(self.negate.value,
                       lambda: self._aug_negate(images),
                       lambda: tf.identity(images))
    return images


@dataclasses.dataclass
class TaskGenerator:
  """Task generator using a dictionary of NumPy arrays as input."""

  def __init__(self,
               data,
               num_labels,
               image_size,
               always_same_labels = False,
               use_label_subset = None):
    self.data = data
    self.num_labels = num_labels
    self.image_size = (image_size, image_size)
    self.always_same_labels = always_same_labels
    if use_label_subset is not None:
      self.use_labels = use_label_subset
    else:
      self.use_labels = list(self.data.keys())

  def _labels_per_batch(self, batch_size):
    samples_per_label = batch_size // self.num_labels
    labels_with_extra = batch_size % self.num_labels
    output = []
    for i in range(self.num_labels):
      if i < labels_with_extra:
        output.append(samples_per_label + 1)
      else:
        output.append(samples_per_label)
    return output

  def sample_random_labels(self,
                           labels,
                           batch_size,
                           same_labels = None
                           ):
    """Generator producing random labels and corr. numbers of samples."""
    if same_labels is None:
      same_labels = self.always_same_labels
    if same_labels:
      chosen_labels = labels[:self.num_labels]
    else:
      chosen_labels = np.random.choice(labels,
                                       size=self.num_labels,
                                       replace=False)
    samples_per_label = batch_size // self.num_labels
    labels_with_extra = batch_size % self.num_labels
    for i, label in enumerate(chosen_labels):
      pick = samples_per_label
      if i < labels_with_extra:
        pick += 1
      yield label, pick

  def _images_labels(self,
                     label_generator,
                     unlabeled = 0,
                     ):
    """Produces labels and images from the label generator."""
    images, labels, classes = [], [], []
    consecutive_label = 0
    for label, samples in label_generator():
      sample = self.data[label]
      chosen = np.random.choice(range(sample.shape[0]), size=samples,
                                replace=False)
      images.append(sample[chosen, :, :])
      chosen_labels = np.array([consecutive_label] * samples)
      remove_label = [(i < unlabeled) for i in range(samples)]
      # This indicates that the sample does not have a label.
      chosen_labels[remove_label] = self.num_labels
      classes.append(np.array([label] * samples))
      labels.append(chosen_labels)
      consecutive_label += 1
    return images, labels, classes

  def _make_semisupervised_samples(
      self, batch_sizes, num_unlabeled_per_class
      ):
    """Helper function for creating multiple semi-supervised samples."""
    output = []
    use_labels = self.use_labels
    if callable(use_labels):
      use_labels = use_labels()
    if not self.always_same_labels:
      # Copying to avoid changing the original list
      use_labels = use_labels[:]
      np.random.shuffle(use_labels)
    for batch_size, unlabeled in zip(batch_sizes, num_unlabeled_per_class):
      # Using the same labelset in all batches.
      label_generator = functools.partial(self.sample_random_labels,
                                          use_labels, batch_size,
                                          same_labels=True)
      output.append(self._images_labels(label_generator, unlabeled))

    return output

  def _make_semisupervised_batches(self,
                                   batch_sizes,
                                   num_unlabeled_per_class):
    """Creates batches of semi-supervised samples."""
    batches = self._make_semisupervised_samples(
        batch_sizes, num_unlabeled_per_class)
    output = []
    for images, labels, classes in batches:
      output.extend([image_mat.astype(np.float32) for image_mat in images])
      output.extend([label_mat.astype(np.int32) for label_mat in labels])
      output.extend([class_mat.astype(np.int32) for class_mat in classes])
    return tuple(output)

  def _make_supervised_batch(self, batch_size):
    """Creates a batch of supervised samples."""
    batches = self._make_semisupervised_samples([batch_size], [0])
    images, labels, classes = batches[0]
    labels = np.concatenate(labels, axis=0).astype(np.int32)
    images = np.concatenate(images, axis=0).astype(np.float32)
    classes = np.concatenate(classes, axis=0).astype(np.int32)
    return images, labels, classes

  def get_batches(self,
                  batch_sizes,
                  config,
                  num_unlabeled_per_class,
                  ):
    """Generator producing multiple separate balanced batches of data.

    Arguments:
      batch_sizes: A list of batch sizes for all batches.
      config: Augmentation configuration.
      num_unlabeled_per_class: A list of integers indicating a number of
        "unlabeled" samples per class for each batch.

    Returns:
      A list of (images,labels) pairs produced for each output batch.
    """
    sup_sample = functools.partial(
        self._make_semisupervised_batches,
        num_unlabeled_per_class=num_unlabeled_per_class,
        batch_sizes=batch_sizes)
    # Returned array is [images, ..., labels, ..., images, ..., labels, ...]
    types = [tf.float32] * self.num_labels
    types += [tf.int32] * self.num_labels
    types += [tf.int32] * self.num_labels
    types = types * len(batch_sizes)
    output = tf.py_func(sup_sample, [], types, stateful=True)

    images_labels = []
    some_label = list(self.data.keys())[0]
    offs = 0
    for batch_size in batch_sizes:
      images = output[offs:offs + self.num_labels]
      offs += self.num_labels
      labels = output[offs:offs + self.num_labels]
      offs += self.num_labels
      classes = output[offs:offs + self.num_labels]
      offs += self.num_labels
      # Setting a proper shape for post-processing to work
      samples_per_label = self._labels_per_batch(batch_size)
      for image, num_samples in zip(images, samples_per_label):
        image.set_shape([num_samples] + list(self.data[some_label][0].shape))
      # Processing and combining in batches
      if config.children:
        images = [config.process(image_mat, idx)
                  for idx, image_mat in enumerate(images)]
      else:
        images = [config.process(image_mat) for image_mat in images]
      images_labels.append((tf.concat(images, axis=0),
                            tf.concat(labels, axis=0),
                            tf.concat(classes, axis=0)))

    # Shuffling each batch
    output = []
    for images, labels, classes in images_labels:
      indices = tf.range(start=0, limit=tf.shape(images)[0], dtype=tf.int32)
      shuffled_indices = tf.random.shuffle(indices)
      images = tf.gather(images, shuffled_indices)
      labels = tf.gather(labels, shuffled_indices)
      classes = tf.gather(classes, shuffled_indices)
      output.append((images, labels, classes))
    return output

  def get_batch(self,
                batch_size,
                config,
                num_unlabeled_per_class = 0
                ):
    """Generator producing a single batch of data (meta-train + meta-test)."""
    if num_unlabeled_per_class > 0:
      raise ValueError('Unlabeled samples are currently only supported in '
                       'balanced inputs.')
    sup_sample = functools.partial(self._make_supervised_batch,
                                   batch_size=batch_size)
    images, labels, classes = tf.py_func(
        sup_sample, [], (tf.float32, tf.int32, tf.int32), stateful=True)
    some_label = list(self.data.keys())[0]
    # Setting a proper shape for post-processing to work
    images.set_shape([batch_size] + list(self.data[some_label][0].shape))
    images = config.process(images)

    indices = tf.range(start=0, limit=tf.shape(images)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    images = tf.gather(images, shuffled_indices)
    labels = tf.gather(labels, shuffled_indices)
    classes = tf.gather(classes, shuffled_indices)

    return images, labels, classes


def make_numpy_data(sess,
                    ds,
                    batch_size,
                    num_labels,
                    samples_per_label,
                    image_key = 'image',
                    label_key = 'label',
                    transpose = True,
                    max_batches = None):
  """Makes a label-to-samples dictionary from the TF dataset.

  Arguments:
    sess: Initialized TF session.
    ds: TF dataset.
    batch_size: batch size to use for processing data.
    num_labels: total number of labels.
    samples_per_label: number of samples per label to accumulate.
    image_key: key of the image tensor.
    label_key: key of the label tensor.
    transpose: if True, the image is transposed (XY).
    max_batches: if provided, we process no more than this number of batches.

  Returns:
    Dictionary mapping labels to tensors containing all samples.
  """
  data = tf.data.make_one_shot_iterator(ds.batch(batch_size)).get_next()

  examples = {i: [] for i in range(num_labels)}
  batch_index = 0
  while True:
    value = sess.run(data)
    samples = value[label_key].shape[0]
    for index in range(samples):
      label = int(value[label_key][index])
      if len(examples[label]) < samples_per_label:
        examples[label].append(value[image_key][index])
    # Checking if we accumulated enough.
    min_examples = np.min([len(examples[key]) for key in examples])
    if min_examples >= samples_per_label:
      break
    # Checking if we already processed too many batches.
    batch_index += 1
    if max_batches is not None and batch_index >= max_batches:
      break

  examples = {k: np.stack(v, axis=0) for k, v in examples.items()}
  if transpose:
    examples = {k: np.transpose(v, axes=[0, 2, 1, 3])
                for k, v in examples.items()}
  return examples

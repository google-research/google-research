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

"""Data provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from saccader import utils
tf.disable_v2_behavior()


_IMAGE_SIZE_DICT = {
    "imagenet224": 224,
    "imagenet331": 331,
}

# Mean and stddev after normalizing to 0 - 1 range.
_MEAN_RGB_DICT = {
    "imagenet": [0.485, 0.456, 0.406],
}

_STDDEV_RGB_DICT = {
    "imagenet": [0.229, 0.224, 0.225],
}


# =================  Preprocessing Utility Functions. =====================
def _distorted_bounding_box_crop(image,
                                 bbox,
                                 min_object_covered=0.1,
                                 aspect_ratio_range=(0.75, 1.33),
                                 area_range=(0.05, 1.0),
                                 max_attempts=100,
                                 scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: `Tensor` of image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]` where
      each coordinate is [0, 1) and the coordinates are arranged as `[ymin,
      xmin, ymax, xmax]`. If num_boxes is 0 then use the whole image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped area
      of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image must
      contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional `str` for name scope.

  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf.name_scope(scope, "distorted_bounding_box_crop", [image, bbox]):
    shape = tf.shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x,
                                          target_height, target_width)

    return image


def _random_crop(image, image_size):
  """Make a random crop of size `image_size`."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = _distorted_bounding_box_crop(
      image,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=100,
      scope=None)
  return tf.image.resize_bicubic([image], [image_size, image_size])[0]


def _center_crop(image, crop_padding, image_size):
  """Crops to center of image with padding then scales to `image_size`."""
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + crop_padding)) * tf.cast(
          tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  image = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                        padded_center_crop_size,
                                        padded_center_crop_size)

  image = tf.image.resize_bicubic([image], [image_size, image_size])[0]

  return image


def standardize_image(image, dataset):
  """Normalize the image to zero mean and unit variance."""
  moment_shape = [1] * (len(image.shape) - 1) + [3]
  offset = tf.constant(_MEAN_RGB_DICT[dataset], shape=moment_shape)
  image -= offset

  scale = tf.constant(_STDDEV_RGB_DICT[dataset], shape=moment_shape)
  image /= scale
  return image


def preprocess_imagenet_for_train(image, image_size):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    image_size: size of image.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _random_crop(image, image_size=image_size)
  image = standardize_image(image, "imagenet")
  image = tf.image.random_flip_left_right(image)
  image = tf.reshape(image, [image_size, image_size, 3])
  return image


def preprocess_imagenet_for_eval(image, image_size, crop=True,
                                 standardize=True):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    image_size: size of image.
    crop: If is_training is `False`, determines whether the function should
      extract a central crop of the images (as for standard ImageNet
      evaluation), or rescale the full image without cropping.
    standardize: If `True` (default), standardize to unit variance. Otherwise,
      the returned image is approximately in [0, 1], with some excursions due to
      bicubic resampling.

  Returns:
    A preprocessed image `Tensor`.
  """
  crop_padding = image_size // 10
  image = _center_crop(
      image, crop_padding=crop_padding if crop else 0, image_size=image_size)
  if standardize:
    image = standardize_image(image, "imagenet")
  image = tf.reshape(image, [image_size, image_size, 3])
  return image


def preprocess_imagenet(data,
                        is_training,
                        image_size,
                        crop=True):
  """Preprocesses the given image.

  Args:
    data: `Dictionary` with 'image' representing an image of arbitrary size,
      and 'label' representing image class label.
    is_training: `bool` for whether the preprocessing is for training.
    image_size: size of image.
    crop: If is_training is `False`, determines whether the function should
      extract a central crop of the images (as for standard ImageNet
      evaluation), or rescale the full image without cropping.

  Returns:
    A preprocessed image `Tensor`.
    image label.
    mask to track padded vs reral data.
  """
  # Create a mask variable to track the real vs padded data in the last batch.
  mask = 1.
  image = data["image"]

  # Reserve label 0 for background
  label = data["label"] + 1
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  if is_training:
    return preprocess_imagenet_for_train(
        image, image_size=image_size), label, mask
  else:
    return preprocess_imagenet_for_eval(
        image, image_size=image_size, crop=crop), label, mask


# ========= ImageNet data provider. ============
class ImageNetDataProvider(object):
  """ImageNet Data Provider.

  Attributes:
    images: (4-D tensor) Images of shape (batch, height, width, channels).
    labels: (1-D tensor) Data labels of size (batch,).
    mask: (1-D boolean tensor) Data mask. Used when data is not repeated to
      indicate the fraction of the batch with true data in the final batch.
    num_classes: (Integer) Number of classes in the dataset.
    num_examples: (Integer) Number of examples in the dataset.
    class_names: (List of Strings) ImageNet id for class labels.
  """

  def __init__(self,
               batch_size,
               subset,
               data_dir,
               is_training=False):
    dataset_builder = tfds.builder("imagenet2012", data_dir=data_dir)
    dataset_builder.download_and_prepare(download_dir=data_dir)
    if subset == "train":
      dataset = dataset_builder.as_dataset(split=tfds.Split.TRAIN,
                                           shuffle_files=True)
    elif subset == "validation":
      dataset = dataset_builder.as_dataset(split=tfds.Split.VALIDATION)
    else:
      raise ValueError("subset %s is undefined " % subset)
    preprocess_fn = self._preprocess_fn(is_training)
    dataset = dataset.map(preprocess_fn,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    info = dataset_builder.info
    if is_training:
      # 4096 is ~0.625 GB of RAM. Reduce if memory issues encountered.
      dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.repeat(-1 if is_training else 1)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)

    if not is_training:
      # Pad the remainder of the last batch to make batch size fixed.
      dataset = utils.pad_to_batch(dataset, batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    self.images, self.labels, self.mask = iterator.get_next()
    self.num_classes = info.features["label"].num_classes + 1
    self.class_names = ["unused"] + info.features["label"].names
    self.num_examples = info.splits[subset].num_examples

  def _preprocess_fn(self, is_training):
    return functools.partial(preprocess_imagenet, is_training=is_training,
                             image_size=self.image_size)


class ImageNet224DataProvider(ImageNetDataProvider):
  """ImageNet 224x224 data provider."""
  image_size = _IMAGE_SIZE_DICT["imagenet224"]


class ImageNet331DataProvider(ImageNetDataProvider):
  """ImageNet 331x331 data provider."""
  image_size = _IMAGE_SIZE_DICT["imagenet331"]


# ===== Function that provides data. ======
_DATASETS = {
    "imagenet224": ImageNet224DataProvider,
    "imagenet331": ImageNet331DataProvider,
}


def get_data_provider(dataset_name):
  """Returns dataset by name."""
  return _DATASETS[dataset_name]

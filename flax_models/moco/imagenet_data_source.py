# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet input pipeline.
"""

import collections

import jax

import tensorflow as tf
import tensorflow_datasets as tfds


TRAIN_IMAGES = 1281167
TEST_IMAGES = 50000


MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def normalize_image(image):
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image


def random_crop(image,
                min_object_covered=0.1,
                aspect_ratio_range=(0.75, 1.33),
                area_range=(0.05, 1.0),
                max_attempts=100,):
  """Randomly crop an input image.

  Args:
    image: The image to be cropped.
    min_object_covered: The minimal percentage of the target object that should
      be in the final crop.
    aspect_ratio_range: The cropped area of the image must have an aspect
      ratio = width / height within this range.
    area_range: The cropped area of the image must contain a fraction of the
      input image within this range.
    max_attempts: Number of attempts at generating a cropped region of the image
      of the specified constraints. After max_attempts failures,
      the original image is returned.
  Returns:
    A random crop of the supplied image.
  """
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      tf.shape(image),
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop = tf.image.crop_to_bounding_box(image, offset_y, offset_x,
                                       target_height, target_width)
  return crop


def center_crop(image, image_size, crop_padding=32):
  """Crop an image in the center while preserving aspect ratio.

  Args:
    image: The image to be cropped.
    image_size: the desired crop size.
    crop_padding: minimal distance of the crop from the edge of the image.
  Returns:
    The center crop of the provided image.
  """
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + crop_padding)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                       padded_center_crop_size,
                                       padded_center_crop_size)
  return crop


def colour_jitter(image, greyscale_prob=0.0):
  """Colour jitter augmentation."""
  # Make sure it has 3 channels so random_saturation and random_hue don't
  # fail on greyscale images
  image = image * tf.ones([1, 1, 3], dtype=image.dtype)
  if greyscale_prob > 0.0:
    def f_grey():
      return tf.image.rgb_to_grayscale(image)

    def f_colour():
      image_col = tf.image.random_saturation(image, 0.7, 1.4)
      image_col = tf.image.random_hue(image_col, 0.1)
      return image_col

    p = tf.random.uniform(())

    image = tf.cond(tf.less(p, greyscale_prob), f_grey, f_colour)
  else:
    image = tf.image.random_saturation(image, 0.7, 1.4)
    image = tf.image.random_hue(image, 0.1)
  image = tf.image.random_contrast(image, 0.7, 1.4)
  image = tf.image.random_brightness(image, 0.4)
  return image


def preprocess_train_image(image, greyscale_prob=0.0, image_size=224):
  """Preprocess a raw ImageNet image for training or evaluation.

  Args:
    image: The image to be preprocessed.
    greyscale_prob: Probability of augmentation converting image to greyscale.
    image_size: The target size of the image.
  Returns:
    The pre-processed image.
  """
  image = random_crop(image)
  image = tf.image.resize([image],
                          [image_size, image_size],
                          method=tf.image.ResizeMethod.BICUBIC
                         )[0]
  # Randomly flip the image horizontally.
  image = tf.image.random_flip_left_right(image)

  image = colour_jitter(image, greyscale_prob=greyscale_prob)

  image = normalize_image(image)
  return image


def preprocess_eval_image(image, image_size=224):
  """Preprocess a raw ImageNet image for training or evaluation.

  Args:
    image: The image to be preprocessed.
    image_size: The target size of the image.
  Returns:
    The pre-processed image.
  """
  image = center_crop(image, image_size)
  image = tf.image.resize([image],
                          [image_size, image_size],
                          method=tf.image.ResizeMethod.BICUBIC
                         )[0]
  image = normalize_image(image)
  return image


def _load_tfds_imagenet(split_name, n_total):
  split_size = float(n_total) // jax.host_count()
  start = split_size * jax.host_id()
  end = start + split_size
  start_index = int(round(start))
  end_index = int(round(end))
  split = '{}[{}:{}]'.format(split_name, start_index, end_index)
  return tfds.load('imagenet2012:5.*.*', split=split)


ImageNetDataSource = collections.namedtuple(
    'ImageNetDataSource',
    ['n_train', 'n_test', 'n_classes',
     'train_moco_ds', 'train_clf_ds', 'test_ds'])


def load_imagenet(train_batch_size, eval_batch_size,
                  greyscale_prob=0.0, image_size=224, shuffle_seed=1):
  """Load ImageNet 2012.

  Args:
    train_batch_size: training batch size
    eval_batch_size: evaluation batch size
    greyscale_prob: probability of converting image to greyscale
    image_size: output image size
    shuffle_seed: shuffling random seed

  Returns:
    An ImageNetDataSource namedtuple with the following attributes:
      n_train: number of training samples
      n_test: number of test samples
      n_classes: number of classes
      train_moco_ds: Dataset for training MoCo network
      train_clf_ds: Dataset for training linear classifier
      test_ds: Test dataset
  """
  # Draw unsupervised samples from complete training set
  train_ds = _load_tfds_imagenet('train', TRAIN_IMAGES)

  def _augment_moco_sample(sample):
    aug_sample = {
        'key_image': preprocess_train_image(
            sample['image'], greyscale_prob=greyscale_prob,
            image_size=image_size),
        'query_image': preprocess_train_image(
            sample['image'], greyscale_prob=greyscale_prob,
            image_size=image_size),
    }
    return aug_sample

  def _augment_clf_sample(sample):
    aug_sample = {
        'image': preprocess_train_image(
            sample['image'], greyscale_prob=greyscale_prob,
            image_size=image_size),
        'label': sample['label'],
    }
    return aug_sample

  def _process_eval_sample(sample):
    image = preprocess_eval_image(sample['image'], image_size=image_size)
    eval_sample = {'image': image, 'label': sample['label']}
    return eval_sample

  train_moco_ds = train_ds.repeat()
  train_moco_ds = train_moco_ds.shuffle(16 * train_batch_size,
                                        seed=shuffle_seed)
  train_moco_ds = train_moco_ds.map(_augment_moco_sample,
                                    num_parallel_calls=128)

  train_clf_ds = train_ds.repeat()
  train_clf_ds = train_clf_ds.shuffle(16 * train_batch_size,
                                      seed=shuffle_seed)
  train_clf_ds = train_clf_ds.map(_augment_clf_sample,
                                  num_parallel_calls=128)

  train_moco_ds = train_moco_ds.batch(train_batch_size,
                                      drop_remainder=True)
  train_clf_ds = train_clf_ds.batch(train_batch_size,
                                    drop_remainder=True)

  train_moco_ds = train_moco_ds.prefetch(10)
  train_clf_ds = train_clf_ds.prefetch(10)

  #
  # Test set
  #

  test_ds = _load_tfds_imagenet('validation', TEST_IMAGES)
  test_ds = test_ds.cache()

  test_ds = test_ds.map(_process_eval_sample, num_parallel_calls=128)
  test_ds = test_ds.batch(eval_batch_size, drop_remainder=True)
  test_ds = test_ds.repeat()
  test_ds = test_ds.prefetch(10)

  return ImageNetDataSource(
      n_train=TRAIN_IMAGES, n_test=TEST_IMAGES, n_classes=1000,
      train_moco_ds=train_moco_ds, train_clf_ds=train_clf_ds,
      test_ds=test_ds)

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

"""ImageNet input pipeline.
"""

import os
import pickle
import jax
import numpy as np
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
  """Colour jitter augmentation.

  Args:
    image: The image to be augmented
    greyscale_prob: probability of greyscale conversion

  Returns:
    Augmented image
  """
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

    p = tf.random.uniform([1])

    image = tf.cond(tf.less(p[0], greyscale_prob), f_grey, f_colour)
  else:
    image = tf.image.random_saturation(image, 0.7, 1.4)
    image = tf.image.random_hue(image, 0.1)
  image = tf.image.random_contrast(image, 0.7, 1.4)
  image = tf.image.random_brightness(image, 0.4)
  return image


def preprocess_train_image(image, apply_colour_jitter=False,
                           greyscale_prob=0.0, image_size=224):
  """Preprocess a raw ImageNet image for training or evaluation.

  Args:
    image: The image to be preprocessed.
    apply_colour_jitter: If True, apply colour jitterring.
    greyscale_prob: Probability of converting image to greyscale.
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

  if apply_colour_jitter:
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


_JPEG_ENCODED_FEATURE_DESCRIPTION = {
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image': tf.io.FixedLenFeature([], tf.string),
    'file_name': tf.io.FixedLenFeature([], tf.string),
}


def _filter_tfds_by_file_name(in_ds, subset_filenames):
  kv_init = tf.lookup.KeyValueTensorInitializer(
      np.array(subset_filenames), np.ones((len(subset_filenames),), dtype=int),
      key_dtype=tf.string, value_dtype=tf.int64)
  ht = tf.lookup.StaticHashTable(kv_init, 0)

  def pred_fn(x):
    return tf.equal(ht.lookup(x['file_name']), 1)

  return in_ds.filter(pred_fn)


def _deserialize_and_decode_jpeg(serialized_sample):
  sample = tf.io.parse_single_example(serialized_sample,
                                      _JPEG_ENCODED_FEATURE_DESCRIPTION)
  sample['image'] = tf.io.decode_jpeg(sample['image'])
  return sample


def _deserialize_sample(serialized_sample):
  return tf.io.parse_example(serialized_sample,
                             _JPEG_ENCODED_FEATURE_DESCRIPTION)


def _decode_jpeg(sample):
  image = tf.io.decode_jpeg(sample['image'])
  return dict(label=sample['label'], file_name=sample['file_name'], image=image)


def deserialize_and_decode_image_dataset(ds, batch_size):
  if batch_size is not None and batch_size > 1:
    return ds.batch(batch_size).map(
        _deserialize_sample,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch().map(
            _decode_jpeg, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    return ds.map(_deserialize_and_decode_jpeg,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)


def _load_tfds_imagenet(split_name, n_total):
  """Load ImageNet from TFDS."""
  split_size = float(n_total) // jax.host_count()
  start = split_size * jax.host_id()
  end = start + split_size
  start_index = int(round(start))
  end_index = int(round(end))
  split = '{}[{}:{}]'.format(split_name, start_index, end_index)
  return tfds.load('imagenet2012:5.*.*', split=split)


def _load_custom_imagenet_split(split_path):
  """Load a custom split of the ImageNet dataset."""
  if not tf.io.gfile.exists(split_path):
    raise RuntimeError('Cannot find {}'.format(split_path))
  shard_filenames = tf.io.gfile.listdir(split_path)
  shard_filenames.sort()
  if jax.host_count() > 1:
    n_hosts = jax.host_count()
    host_id = jax.host_id()
    shard_filenames = [f for i, f in enumerate(shard_filenames)
                       if (i % n_hosts) == host_id]
  files_in_split = [os.path.join(split_path, f) for f in shard_filenames]
  ds = tf.data.TFRecordDataset(files_in_split, buffer_size=128 * 1024 * 1024,
                               num_parallel_reads=len(files_in_split))
  # ds = deserialize_and_decode_image_dataset(ds, batch_size=256)
  ds = deserialize_and_decode_image_dataset(ds, batch_size=1)
  return ds


_SUP_PATH_PAT = r'{imagenet_subset_dir}/imagenet_{n_sup}_seed{subset_seed}'
_VAL_TVSPLIT_PATH_PAT = r'{imagenet_subset_dir}/imagenet_tv{n_val}s{val_seed}_split.pkl'
_VAL_PATH_PAT = r'{imagenet_subset_dir}/imagenet_tv{n_val}s{val_seed}_val'
_VAL_SUP_PATH_PAT = r'{imagenet_subset_dir}/imagenet_tv{n_val}s{val_seed}_{n_sup}_seed{subset_seed}'


class ImageNetDataSource(object):
  """ImageNet data source.

  Attributes:
    n_train: number of training samples
    n_sup: number of supervised samples
    n_val: number of validation samples
    n_test: number of test samples
    train_semisup_ds: Semi-supervised training dataset
    train_unsup_ds: Unsupervised training dataset
    train_sup_ds: Supervised training dataset
    val_ds: Validation dataset
    test_ds: Test dataset
    n_classes: Number of classes
  """

  def __init__(self, imagenet_subset_dir, n_val, n_sup, train_batch_size,
               eval_batch_size, augment_twice, apply_colour_jitter=False,
               greyscale_prob=0.0, load_test_set=True, image_size=224,
               subset_seed=12345, val_seed=131):
    if n_val == 0:
      # We are using the complete ImageNet training set for traininig
      # No samples are being held out for validation

      # Draw unsupervised samples from complete training set
      train_unsup_ds = _load_tfds_imagenet('train', TRAIN_IMAGES)
      self.n_train = TRAIN_IMAGES

      if n_sup == -1 or n_sup == TRAIN_IMAGES:
        # All training samples are supervised
        train_sup_ds = train_unsup_ds
        self.n_sup = TRAIN_IMAGES
      else:
        sup_path = _SUP_PATH_PAT.format(
            imagenet_subset_dir=imagenet_subset_dir, n_sup=n_sup,
            subset_seed=subset_seed)
        train_sup_ds = _load_custom_imagenet_split(sup_path)
        self.n_sup = n_sup

      val_ds = None
      self.n_val = 0
    else:
      # A validation set has been requested

      # Load the pickle file that tells us which file names are train / val
      tvsplit_path = _VAL_TVSPLIT_PATH_PAT.format(
          imagenet_subset_dir=imagenet_subset_dir, n_val=n_val,
          val_seed=val_seed)
      with tf.io.gfile.GFile(tvsplit_path, 'rb') as f_tvsplit:
        tvsplit = pickle.load(f_tvsplit)
      train_fn = tvsplit['train_fn']

      # Filter the dataset to select samples in the training set
      trainval_ds = _load_tfds_imagenet('train', TRAIN_IMAGES)
      train_unsup_ds = _filter_tfds_by_file_name(trainval_ds, train_fn)
      self.n_train = len(train_fn)

      # Load the validation set from a custom dataset
      val_path = _VAL_PATH_PAT.format(imagenet_subset_dir=imagenet_subset_dir,
                                      n_val=n_val,
                                      val_seed=val_seed)
      val_ds = _load_custom_imagenet_split(val_path)
      self.n_val = n_val

      if n_sup == -1 or n_sup == len(train_fn):
        # All training samples are supervised
        train_sup_ds = train_unsup_ds
        self.n_sup = len(train_fn)
      else:
        sup_path = _VAL_SUP_PATH_PAT.format(
            imagenet_subset_dir=imagenet_subset_dir, n_val=n_val,
            val_seed=val_seed, n_sup=n_sup,
            subset_seed=subset_seed)
        train_sup_ds = _load_custom_imagenet_split(sup_path)
        self.n_sup = n_sup

    train_sup_ds = train_sup_ds.repeat()
    train_sup_ds = train_sup_ds.shuffle(8 * train_batch_size)

    train_unsup_ds = train_unsup_ds.repeat()
    train_unsup_ds = train_unsup_ds.shuffle(8 * train_batch_size)

    train_semisup_ds = tf.data.Dataset.zip((train_sup_ds, train_unsup_ds))

    def _augment_sup(sup_sample):
      """Augment supervised sample."""
      sample = {
          'sup_image': preprocess_train_image(
              sup_sample['image'], apply_colour_jitter=apply_colour_jitter,
              greyscale_prob=greyscale_prob, image_size=image_size),
          'sup_label': sup_sample['label'],
      }
      return sample

    def _augment_unsup_once(unsup_sample):
      """Augment unsupervised sample, single augmentation."""
      unsup_x0 = preprocess_train_image(
          unsup_sample['image'], apply_colour_jitter=apply_colour_jitter,
          greyscale_prob=greyscale_prob, image_size=image_size)
      sample = {
          'unsup_image0': unsup_x0,
          'unsup_image1': unsup_x0,
      }
      return sample

    def _augment_unsup_twice(unsup_sample):
      """Augment unsupervised sample, two augmentations."""
      sample = {
          'unsup_image0': preprocess_train_image(
              unsup_sample['image'], apply_colour_jitter=apply_colour_jitter,
              greyscale_prob=greyscale_prob, image_size=image_size),
          'unsup_image1': preprocess_train_image(
              unsup_sample['image'], apply_colour_jitter=apply_colour_jitter,
              greyscale_prob=greyscale_prob, image_size=image_size),
      }
      return sample

    def _augment_semisup_once(sup_sample, unsup_sample):
      """Augment semi-supervised sample, single augmentation."""
      unsup_x0 = preprocess_train_image(
          unsup_sample['image'], apply_colour_jitter=apply_colour_jitter,
          greyscale_prob=greyscale_prob, image_size=image_size)
      semisup_sample = {
          'sup_image': preprocess_train_image(
              sup_sample['image'], apply_colour_jitter=apply_colour_jitter,
              greyscale_prob=greyscale_prob, image_size=image_size),
          'sup_label': sup_sample['label'],
          'unsup_image0': unsup_x0,
          'unsup_image1': unsup_x0,
      }
      return semisup_sample

    def _augment_semisup_twice(sup_sample, unsup_sample):
      """Augment semi-supervised sample, two augmentations."""
      semisup_sample = {
          'sup_image': preprocess_train_image(
              sup_sample['image'], apply_colour_jitter=apply_colour_jitter,
              greyscale_prob=greyscale_prob, image_size=image_size),
          'sup_label': sup_sample['label'],
          'unsup_image0': preprocess_train_image(
              unsup_sample['image'], apply_colour_jitter=apply_colour_jitter,
              greyscale_prob=greyscale_prob, image_size=image_size),
          'unsup_image1': preprocess_train_image(
              unsup_sample['image'], apply_colour_jitter=apply_colour_jitter,
              greyscale_prob=greyscale_prob, image_size=image_size),
      }
      return semisup_sample

    def _process_eval_sample(x):
      """Pre-process evaluation sample."""
      image = preprocess_eval_image(x['image'], image_size=image_size)
      batch = {'image': image, 'label': x['label']}
      return batch

    if augment_twice:
      train_semisup_ds = train_semisup_ds.map(_augment_semisup_twice,
                                              num_parallel_calls=128)
      train_unsup_only_ds = train_unsup_ds.map(_augment_unsup_twice,
                                               num_parallel_calls=128)
    else:
      train_semisup_ds = train_semisup_ds.map(_augment_semisup_once,
                                              num_parallel_calls=128)
      train_unsup_only_ds = train_unsup_ds.map(_augment_unsup_once,
                                               num_parallel_calls=128)
    train_sup_only_ds = train_sup_ds.map(_augment_sup,
                                         num_parallel_calls=128)
    train_semisup_ds = train_semisup_ds.batch(train_batch_size,
                                              drop_remainder=True)
    train_unsup_only_ds = train_unsup_only_ds.batch(train_batch_size,
                                                    drop_remainder=True)
    train_sup_only_ds = train_sup_only_ds.batch(train_batch_size,
                                                drop_remainder=True)
    train_semisup_ds = train_semisup_ds.prefetch(10)
    train_unsup_only_ds = train_unsup_only_ds.prefetch(10)
    train_sup_only_ds = train_sup_only_ds.prefetch(10)
    self.train_semisup_ds = train_semisup_ds
    self.train_unsup_ds = train_unsup_only_ds
    self.train_sup_ds = train_sup_only_ds

    #
    # Validation set
    #
    if n_val > 0:
      val_ds = val_ds.cache()
      val_ds = val_ds.map(_process_eval_sample, num_parallel_calls=128)
      val_ds = val_ds.batch(eval_batch_size)
      val_ds = val_ds.repeat()
      val_ds = val_ds.prefetch(10)
      self.val_ds = val_ds
    else:
      self.val_ds = None

    if load_test_set:
      #
      # Test set
      #

      test_ds = _load_tfds_imagenet('validation', TEST_IMAGES)
      test_ds = test_ds.cache()

      test_ds = test_ds.map(_process_eval_sample, num_parallel_calls=128)
      test_ds = test_ds.batch(eval_batch_size)
      test_ds = test_ds.repeat()
      test_ds = test_ds.prefetch(10)
      self.test_ds = test_ds
      self.n_test = TEST_IMAGES
    else:
      self.test_ds = None
      self.n_test = 0

    self.n_classes = 1000


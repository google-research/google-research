# coding=utf-8
"""Utils functions for supporting the code."""

from __future__ import absolute_import
from __future__ import division


from absl import flags
import numpy as np
from numpy.core.numeric import False_  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from ieg.dataset_utils import augmentation_transforms
from ieg.dataset_utils import autoaugment
from ieg.dataset_utils import randaugment

FLAGS = flags.FLAGS

GODD_POLICIES = autoaugment.cifar10_policies()
RANDOM_POLICY_OPS = randaugment.RANDOM_POLICY_OPS
BLUR_OPS = randaugment.BLUR_OPS
_IMAGE_SIZE = 224
_CROP_PADDING = 32
# Does multiprocessing speed things up?
POOL = None


def cifar_process(image, augmentation=True):
  """Map function for cifar dataset.

  Args:
    image: An image tensor.
    augmentation: If True, process train images.

  Returns:
    A processed image tensor.
  """
  # label = tf.cast(label, dtype=tf.int32)
  image = tf.math.divide(tf.cast(image, dtype=tf.float32), 255.0)

  if augmentation:
    image = tf.image.resize_image_with_crop_or_pad(image, 32 + 4, 32 + 4)
    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [32, 32, 3])
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    image = tf.clip_by_value(image, 0, 1)

  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)

  return image


def apply_autoaugment(data, no_policy=False):
  """Python written Autoaugment for preprocessing image.

  Args:
    data: can be
      A list: (3D image, policy)
      A list: (4D image, policy) A numpy image
    no_policy: If True, does not used learned AutoAugment policies

  Returns:
    A 3D or 4D processed images
  """
  if not isinstance(data, (list, tuple)):
    epoch_policy = GODD_POLICIES[np.random.choice(len(GODD_POLICIES))]
    image = data
  else:
    image, epoch_policy = data

  if len(image.shape) == 3:
    images = [image]
  else:
    images = image
  res = []
  for img in images:
    assert img.max() <= 1 and img.min() >= -1
    # ! image is assumed to be normalized to [-1, 1]
    if no_policy:
      final_img = img
    else:
      final_img = augmentation_transforms.apply_policy(epoch_policy, img)

    final_img = augmentation_transforms.random_flip(
        augmentation_transforms.zero_pad_and_crop(final_img, 4))
    final_img = augmentation_transforms.cutout_numpy(final_img)
    res.append(final_img.astype(np.float32))

  res = np.concatenate(res, 0)

  return res


def random_blur(images, magnitude=10, nops=1):
  """Apply random blurs for a batch of data."""
  # using shared policies are better
  policies = [(policy, 0.5, mag) for (policy, mag) in zip(
      np.random.choice(BLUR_OPS, nops), np.random.randint(1, magnitude, nops))]
  policies = [policies] * images.shape[0]
  if POOL is not None:
    jobs = [(image.squeeze(), policy) for image, policy in zip(
        np.split(images.copy(), images.shape[0], axis=0), policies)]
    augmented_images = POOL.map(apply_randomaugment, jobs)
  else:
    augmented_images = []
    for image, policy in zip(images.copy(), policies):
      final_img = apply_randomaugment((image, policy))
      augmented_images.append(final_img)

  augmented_images = np.stack(augmented_images, axis=0)

  return augmented_images


def apply_randomaugment(data):
  """Apply random augmentations."""
  image, epoch_policy = data

  if len(image.shape) == 3:
    images = [image]
  else:
    images = image
  res = []
  for img in images:
    # ! image is assumed to be normalized to [-1, 1]
    final_img = randaugment.apply_policy(epoch_policy, img)
    final_img = randaugment.random_flip(
        randaugment.zero_pad_and_crop(final_img, 4))
    final_img = randaugment.cutout_numpy(final_img)
    res.append(final_img.astype(np.float32))

  res = np.concatenate(res, 0)

  return res


def pool_policy_augmentation(images):
  """Batch AutoAugment.

  Given a 4D numpy tensor of images,
  perform AutoAugment using apply_autoaugment().

  Args:
    images: 4D numpy tensor

  Returns:
    A 4D numpy tensor of processed images.

  """
  # Use the same policy for all batch data seems work better.
  policies = [GODD_POLICIES[np.random.choice(len(GODD_POLICIES))]
             ] * images.shape[0]
  jobs = [(image.squeeze(), policy) for image, policy in zip(
      np.split(images.copy(), images.shape[0], axis=0), policies)]
  if POOL is None:
    jobs = np.split(images.copy(), images.shape[0], axis=0)
    augmented_images = map(apply_autoaugment, jobs)
  else:
    augmented_images = POOL.map(apply_autoaugment, jobs)
  augmented_images = np.stack(augmented_images, axis=0)

  return augmented_images


def random_augmentation(images, magnitude=10, nops=2):
  """Apply random augmentations for a batch of data."""
  # using shared policies are better
  policies = [(policy, 0.5, mag) for (policy, mag) in zip(
      np.random.choice(RANDOM_POLICY_OPS, nops),
      np.random.randint(1, magnitude, nops))]
  policies = [policies] * images.shape[0]
  if POOL is not None:
    jobs = [(image.squeeze(), policy) for image, policy in zip(
        np.split(images.copy(), images.shape[0], axis=0), policies)]
    augmented_images = POOL.map(apply_randomaugment, jobs)
  else:
    augmented_images = []
    for image, policy in zip(images.copy(), policies):
      final_img = apply_randomaugment((image, policy))
      augmented_images.append(final_img)

  augmented_images = np.stack(augmented_images, axis=0)

  return augmented_images


def autoaug_batch_process_map_fn(images, labels):
  """tf.data.Dataset map function to enable python AutoAugmnet with tf.py_func.

  It is usually called after tf.data.Dataset is batched.

  Args:
    images: A 4D tensor of a batch of images.
    labels: labels of images.

  Returns:
    A 5D tensor of processed images [Bx2xHxWx3].
  """
  if FLAGS.aug_type == 'autoaug':
    aa_images = tf.py_func(pool_policy_augmentation, [tf.identity(images)],
                           [tf.float32])
  elif FLAGS.aug_type == 'randaug':
    aa_images = tf.py_func(random_augmentation, [tf.identity(images)],
                           [tf.float32])
  elif FLAGS.aug_type == 'default':
    aa_images = tf.py_func(cifar_process, [tf.identity(images)], [tf.float32])
  else:
    raise NotImplementedError('{} aug_type does not exist'.format(
        FLAGS.aug_type))
  aa_images = tf.reshape(aa_images, [-1] + images.shape.as_list()[1:])
  images = tf.concat([tf.expand_dims(images, 1),
                      tf.expand_dims(aa_images, 1)],
                     axis=1)
  return images, labels


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
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
    cropped image `Tensor`
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image_bytes, bbox]):
    shape = tf.image.extract_jpeg_shape(image_bytes)
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
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

    return image


def _at_least_x_are_equal(a, b, x):
  """Checks if at least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
  """Makes a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      scope=None)
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad, lambda: _decode_and_center_crop(image_bytes, image_size),
      lambda: tf.compat.v1.image.resize(  # pylint: disable=g-long-lambda
          image, [image_size, image_size],
          method=tf.image.ResizeMethod.BILINEAR,
          align_corners=False))
  return image


def _decode_and_center_crop(image_bytes, image_size):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + _CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_size,
      padded_center_crop_size
  ])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return tf.compat.v1.image.resize(
      image, [image_size, image_size],
      method=tf.image.ResizeMethod.BILINEAR,
      align_corners=False)


def _flip(image):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image)
  return image


def preprocess_for_train(image_bytes,
                         use_bfloat16,
                         image_size=_IMAGE_SIZE,
                         autoaugment_name=None):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.
    autoaugment_name: `string` that is the name of the autoaugment policy to
      apply to the image. If the value is `None` autoaugment will not be
      applied.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_random_crop(image_bytes, image_size)
  image = _flip(image)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.cast(image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)

  if autoaugment_name:
    tf.logging.info('Apply AutoAugment policy {}'.format(autoaugment_name))
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, dtype=tf.uint8)
    # Random aug should also work.
    image = autoaugment.distort_image_with_autoaugment(image, autoaugment_name)
    image = tf.cast(image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
  return image


def preprocess_for_eval(image_bytes, use_bfloat16, image_size=_IMAGE_SIZE):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.cast(image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
  return image


def cutout(image, pad_size, replace=0):
  """Applies cutout (https://arxiv.org/abs/1708.04552) to image."""
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # Sample the center location in the image where the zero mask will be applied
  cutout_center_height = tf.random_uniform(
      shape=[], minval=0, maxval=image_height, dtype=tf.int32)

  cutout_center_width = tf.random_uniform(
      shape=[], minval=0, maxval=image_width, dtype=tf.int32)

  lower_pad = tf.maximum(0, cutout_center_height - pad_size)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
  left_pad = tf.maximum(0, cutout_center_width - pad_size)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

  cutout_shape = [
      image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad)
  ]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims,
      constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])
  image = tf.where(
      tf.equal(mask, 0),
      tf.ones_like(image, dtype=image.dtype) * replace, image)
  return image


def imagenet_preprocess_image(image_bytes,
                              is_training=False,
                              use_bfloat16=False,
                              image_size=_IMAGE_SIZE,
                              autoaugment_name=None,
                              use_cutout=False):
  """Preprocesses the given image.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.
    autoaugment_name: `string` that is the name of the autoaugment policy to
      apply to the image. If the value is `None` autoaugment will not be
      applied.
    use_cutout: 'bool' for whether use cutout.

  Returns:
    A preprocessed image `Tensor` with value range of [-1, 1].
  """

  if is_training:
    image = preprocess_for_train(image_bytes, use_bfloat16, image_size,
                                 autoaugment_name)
    if use_cutout:
      image = cutout(image, pad_size=8)
  else:
    image = preprocess_for_eval(image_bytes, use_bfloat16, image_size)

  # Clip the extra values
  image = tf.clip_by_value(image, 0.0, 255.0)
  image = tf.math.divide(image, 255.0)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)

  return image


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy(
    )  # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_str, label, dat_id=None):
  """Creates tf example."""
  feature = {
      'label': _int64_feature(label),
      'image': _bytes_feature(image_str),
  }
  if dat_id is not None:
    feature.update({'id': _int64_feature(dat_id)})
  return tf.train.Example(features=tf.train.Features(feature=feature))


def _parse_image_function(example_proto, include_id=False):
  """Parses tf example."""
  # Parse the input tf.train.Example proto using the dictionary above.
  image_feature_description = {
      'label': tf.io.FixedLenFeature([], tf.int64),
      'image': tf.io.FixedLenFeature([], tf.string),
  }
  if include_id:
    image_feature_description.update(
        {'id': tf.io.FixedLenFeature([], tf.int64)})
  results = tf.io.parse_single_example(example_proto, image_feature_description)
  results['image'] = tf.io.decode_jpeg(
      results['image'], channels=3, name='parse_image_function_decode_jpeg')
  results['label'] = tf.cast(results['label'], tf.int32)
  return results


def write_tfrecords(record_file,
                    ds,
                    ds_size,
                    nshard=50,
                    include_ids=True,
                    filter_fn=None):
  """Rewrites ds as tfrecords that contains data ids."""
  ds = ds.shuffle(ds_size)
  next_item = tf.data.make_one_shot_iterator(ds).get_next()
  dat_id = 0
  part_num = 0
  per_shard_sample = ds_size // nshard + (0 if ds_size % nshard == 0 else 1)
  count = 0
  write_last_round = False
  with tf.Session() as sess:
    img_pl = tf.placeholder(tf.uint8)
    img_str_tf = tf.io.encode_jpeg(img_pl)
    while True:
      try:
        image_str_batch = []
        label_batch = []
        for _ in range(per_shard_sample):
          image, label = sess.run(next_item)
          image_str = sess.run(img_str_tf, feed_dict={img_pl: image})
          if filter_fn is None or filter_fn(label):
            image_str_batch.append(image_str)
            label_batch.append(label)
            count += 1
      except tf.errors.OutOfRangeError:
        if write_last_round:
          tf.logging.info(
              'Generate {} tfrecords ({} samples) with data ids.'.format(
                  nshard, count))
          break
        write_last_round = True
      part_path = record_file + '-{:05d}-of-{:05d}'.format(part_num, nshard)
      part_num += 1
      with tf.io.TFRecordWriter(part_path) as writer:
        for image_str, label in tqdm(
            zip(image_str_batch, label_batch),
            desc='Write tfrecord #%d' % part_num):
          tf_example = image_example(image_str, label,
                                     dat_id if include_ids else None)
          dat_id += 1
          writer.write(tf_example.SerializeToString())


def read_tf_records(record_file, train, include_ids=False):
  """Reads tfrecords and convert to tf.data with data ids."""

  def fetch_dataset_fn(filename):
    buffer_size = 8 * 1024 * 1024  # 8 MiB per file
    dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
    return dataset

  dataset = tf.data.Dataset.list_files(record_file, shuffle=train)
  dataset = dataset.interleave(
      fetch_dataset_fn,
      block_length=16,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  filenames = tf.io.gfile.glob(record_file)
  nums = sum(1 for filename in filenames  # pylint: disable=g-complex-comprehension
             for _ in tf.python_io.tf_record_iterator(filename))
  return dataset.map(
      lambda x: _parse_image_function(x, include_id=include_ids),
      num_parallel_calls=tf.data.experimental.AUTOTUNE), nums

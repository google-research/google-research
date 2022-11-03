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

"""Helper functions to pre-process data inputs for training.

This script modifies raw input images according to the feature importance
estimate. A fraction of the inputs estimated to be most important are replaced
with the global mean.
"""

import numpy as np
import tensorflow.compat.v1 as tf
from interpretability_benchmark.utils.preprocessing_helper import preprocess_image
from interpretability_benchmark.utils.preprocessing_helper import rescale_input

# image size config by model.
IMAGE_DIMS = [224, 224, 3]


def random_ranking(input_image, global_mean, threshold, keep_information):
  """Returns an image where the pixels are randomly modified.

  Dropout is applied to determine which pixels to preserve and which to
  replace with the value determined by the replace_information parameter.

  Args:
    input_image: A 3D float tensor containing the model inputs.
    global_mean: The global mean for each color channel.
    threshold: Percentile of information in the saliency method ranking to
      retain.
    keep_information: A boolean variable that determines whether information is
      kept or is removed.

  Returns:
    feature_ranking: feature ranking input
  Raises:
    ValueError: if the replacement type is not passed or does not match.
  """

  total_pixels = IMAGE_DIMS[0] * IMAGE_DIMS[1] * IMAGE_DIMS[2]
  input_image = tf.reshape(input_image, [total_pixels])

  global_mean_constant = tf.constant(global_mean, shape=[1, 1, 3])
  substitute_information = tf.reshape(
      tf.multiply(global_mean_constant, 1.), [total_pixels])

  fraction = (threshold / 100.)
  multiple = tf.constant(1., shape=[total_pixels], dtype=tf.float32)
  drop_out = tf.nn.dropout(multiple, keep_prob=fraction)
  boolean_mask = tf.cast(drop_out, tf.bool)

  if keep_information:
    feature_ranking = tf.where(boolean_mask, input_image,
                               substitute_information)
  else:
    feature_ranking = tf.where(boolean_mask, substitute_information,
                               input_image)

  feature_ranking = tf.reshape(feature_ranking, IMAGE_DIMS)
  return feature_ranking


def percentage_ranking(saliency_map, input_image, substitute_information,
                       keep_information, number_pixels):
  """Keeps or drops information above a certain percentile."""

  number_pixels = tf.constant(number_pixels, dtype=tf.int32)

  values, indices = tf.nn.top_k(saliency_map, k=number_pixels, sorted=False)
  indices = tf.reshape(indices, [-1, 1])

  feature_ranking = tf.scatter_nd(indices, values, tf.shape(saliency_map))

  if keep_information:
    feature_ranking = tf.where(
        tf.not_equal(feature_ranking, 0.), input_image, substitute_information)
  else:
    feature_ranking = tf.where(
        tf.not_equal(feature_ranking, 0), substitute_information, input_image)

  return feature_ranking


def compute_feature_ranking(input_image,
                            saliency_map,
                            threshold,
                            global_mean,
                            rescale_heatmap=True,
                            keep_information=False,
                            use_squared_value=False):
  """Returns the feature importance ranking estimate.

  If the saliency map ranking is higher than the ranking method threshold, the
  feature ranking estimate is the product of the rescaled [0,1] saliency method
  and the raw input image. If not, the feature estimate is set to
  the mean of the raw input image.

  Args:
    input_image: A 3D float tensor containing the model inputs.
    saliency_map: A 3D float tensor containing the saliency feature ranking.
    threshold: Percentile of information in the saliency method ranking to
      retain.
    global_mean: The global mean for each color channel.
    rescale_heatmap: Boolean, true if saliency heatmap is rescaled.
    keep_information: Boolean, whether the important information should be
      removed or kept.
    use_squared_value: Boolean, whether to take the squared value of the
      heatmap.

  Returns:
    feature_ranking: feature ranking input
  Raises:
    ValueError: The ranking method passed is not known.
  """

  if use_squared_value:
    saliency_map = tf.square(saliency_map)
  else:
    tf.logging.info('not using squared value')

  if rescale_heatmap:
    # re-scale the range of saliency map pixels between [0-1]
    saliency_map = rescale_input(saliency_map)
  else:
    tf.logging.info('not rescaling heatmap')

  total_pixels = IMAGE_DIMS[0] * IMAGE_DIMS[1] * IMAGE_DIMS[2]
  saliency_map = tf.reshape(saliency_map, [total_pixels])
  input_image = tf.reshape(input_image, [total_pixels])

  num_pixels = np.int((threshold / 100.) * total_pixels)

  # we add small epsilon to saliency_method for percentage method
  # epsilon allows us to distinguish between 0 value pixel in saliency heatmap
  # and pixels set to 0 by the tf.scatter_nd gather.
  epsilon = tf.fill([total_pixels], 0.00001)
  saliency_map = tf.add(saliency_map, epsilon)

  global_mean_constant = tf.constant(global_mean, shape=[1, 1, 3])
  substitute_information = tf.reshape(
      tf.multiply(global_mean_constant, 1.), [total_pixels])

  feature_ranking = percentage_ranking(
      saliency_map=saliency_map,
      input_image=input_image,
      substitute_information=substitute_information,
      keep_information=keep_information,
      number_pixels=num_pixels)

  feature_ranking = tf.reshape(feature_ranking, IMAGE_DIMS)

  return feature_ranking


class DataIterator(object):
  """Data input pipeline class.

  Attributes:
    mode: boolean for whether training or eval is occuring.
    data_dir: string indicating path to directory where data is stored.
    saliency_method: Saliency method ranking estimate to evaluate.
    transformation: String indicating modification applied to raw image.
    threshold: Percentile of information to remove from input tensor.
    keep_information: Boolean indicating whether pixels are removed or kept.
    dataset: tfrecord input shards.
    batch_size: Number of inputs in each batch.
    use_squared_value: Boolean indicating whether to take the square of pixels.
    global_mean: Tuple with mean stats for batch of images.
    global_std: Tuple with std stats for batch of images.
    stochastic: Boolean indicating whether ordering of images is stochastic.
    image_size: integer indicating width and length of image.
    num_cores: Int indicating number of cores.
    test_small_sample: Boolean to test workflow.

  Returns:
    feature_ranking: feature_ranking estimate based upon saliency_method.
  """

  def __init__(self,
               mode,
               data_directory,
               saliency_method,
               transformation,
               threshold,
               keep_information,
               use_squared_value,
               mean_stats,
               std_stats,
               stochastic=True,
               image_size=224,
               test_small_sample=False,
               num_cores=8):
    self.mode = mode
    self.data_dir = data_directory
    self.saliency_method = saliency_method
    self.transformation = transformation
    self.threshold = threshold
    self.num_cores = num_cores
    self.keep_information = keep_information
    self.use_squared_value = use_squared_value
    self.global_mean = mean_stats
    self.global_std = std_stats
    self.stochastic = stochastic
    self.image_size = image_size
    self.test_small_sample = test_small_sample

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    if self.test_small_sample:
      image = serialized_example
      label = tf.constant(0, tf.int32)
    else:
      features = tf.parse_single_example(
          serialized_example,
          features={
              'raw_image':
                  tf.FixedLenFeature((), tf.string, default_value=''),
              'height':
                  tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
              'width':
                  tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
              self.saliency_method:
                  tf.VarLenFeature(tf.float32),
              'label':
                  tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
              'prediction_class':
                  tf.FixedLenFeature([], dtype=tf.int64, default_value=-1)
          })
      image = tf.image.decode_image(features['raw_image'], 3)
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

      saliency_heatmap = tf.expand_dims(features[self.saliency_method].values,
                                        0)
      saliency_heatmap = tf.reshape(saliency_heatmap, IMAGE_DIMS)

      if self.transformation in ['modified_image', 'random_baseline']:
        # we apply test_time pre-processing to the raw image before modifying
        # according to the estimator ranking.
        image_preprocess = preprocess_image(
            image, image_size=IMAGE_DIMS[0], is_training=False)

        if self.transformation == 'modified_image':
          tf.logging.info('Computing feature importance estimate now...')
          image = compute_feature_ranking(
              input_image=image_preprocess,
              saliency_map=saliency_heatmap,
              threshold=self.threshold,
              global_mean=self.global_mean,
              rescale_heatmap=True,
              keep_information=self.keep_information,
              use_squared_value=self.use_squared_value)

        if self.transformation == 'random_baseline':
          tf.logging.info('generating a random baseline')
          image = random_ranking(
              input_image=image_preprocess,
              global_mean=self.global_mean,
              threshold=self.threshold,
              keep_information=self.keep_information)

      if self.mode == 'train':
        is_training = True
      else:
        is_training = False

      if self.transformation in ['random_baseline', 'modified_image']:
        tf.logging.info('starting pre-processing for training/eval')
        image = preprocess_image(
            image, image_size=IMAGE_DIMS[0], is_training=is_training)

      if self.transformation == 'raw_image':
        tf.logging.info('starting pre-processing for training/eval')
        image = preprocess_image(
            image, image_size=IMAGE_DIMS[0], is_training=is_training)

      label = tf.cast(tf.reshape(features['label'], shape=[]), dtype=tf.int32)

    return image, label

  def _get_null_input(self, data):
    """Returns a null image (all black pixels)."""
    del data
    return tf.zeros([self.image_size, self.image_size, 3], tf.float32)

  def input_fn(self, params):
    """Input function, iterator for images and labels."""

    data_directory = self.data_dir
    batch_size = params['batch_size']

    if self.test_small_sample:
      dataset = tf.data.Dataset.range(1).repeat().map(self._get_null_input)
    else:
      dataset = tf.data.Dataset.list_files(data_directory, shuffle=False)

      if self.mode == 'train':
        dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.repeat()

      def fetch_dataset(filename):
        dataset = tf.data.TFRecordDataset(filename)
        return dataset

      dataset = dataset.apply(
          tf.data.experimental.parallel_interleave(
              fetch_dataset,
              cycle_length=self.num_cores,
              sloppy=self.stochastic))

      if self.mode == 'train':
        dataset = dataset.shuffle(1024)

    dataset = dataset.map(self.parser, num_parallel_calls=64)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=self.stochastic)

    dataset = dataset.make_one_shot_iterator().get_next()
    return dataset

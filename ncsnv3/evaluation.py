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

"""Utility functions for computing FID/Inception scores."""

import jax
import numpy as np
import six
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub

INCEPTION_TFHUB = '@tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
    INCEPTION_OUTPUT: tf.float32,
    INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def get_inception_model(inceptionv3=False):
  if inceptionv3:
    return tfhub.load(
        'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
  else:
    return tfhub.load(INCEPTION_TFHUB)


def load_dataset_stats(config):
  """Load the pre-computed dataset statistics."""
  filename = None
  if config.data.dataset == 'CIFAR10':
    filename = 'assets/stats/cifar10_stats.npz'
  elif config.data.dataset == 'CELEBA':
    filename = 'assets/stats/celeba_stats.npz'
  elif config.data.dataset == 'LSUN':
    filename = f'assets/stats/lsun_{config.data.category}_{config.data.image_size}_stats.npz'
  else:
    raise ValueError(f'Dataset {config.data.dataset} stats not found.')

  with tf.io.gfile.GFile(filename, 'rb') as fin:
    stats = np.load(fin)
    return stats


def classifier_fn_from_tfhub(output_fields, inception_model,
                             return_tensor=False):
  """Returns a function that can be as a classifier function.

  Copied from tfgan but avoid loading the model each time calling _classifier_fn

  Args:
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    inception_model: A model loaded from TFHub.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]

  def _classifier_fn(images):
    output = inception_model(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

  return _classifier_fn


@tf.function
def run_inception_jit(inputs,
                      inception_model,
                      num_batches=1,
                      inceptionv3=False):
  """Running the inception network. Assuming input is within [0, 255]."""
  if not inceptionv3:
    inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
  else:
    inputs = tf.cast(inputs, tf.float32) / 255.

  return tfgan.eval.run_classifier_fn(
      inputs,
      num_batches=num_batches,
      classifier_fn=classifier_fn_from_tfhub(None, inception_model),
      dtypes=_DEFAULT_DTYPES)


@tf.function
def run_inception_distributed(input_tensor,
                              inception_model,
                              num_batches=1,
                              inceptionv3=False):
  """Distribute the inception network computation to all available TPUs.

  Args:
    input_tensor: the input images. Assumed to be within [0, 255].
    inception_model: the inception network obtained by tfhub.
    num_batches: the number of batches used for dividing the input.
    inceptionv3: use InceptionV3 is True, otherwise use InceptionV1.

  Returns:
    A dictionary with key 'pool_3' and 'logits', representing the pool_3 and
    logits of the inception network respectively.
  """
  num_tpus = jax.local_device_count()
  input_tensors = tf.split(input_tensor, num_tpus, axis=0)
  pool3 = []
  logits = [] if not inceptionv3 else None
  device_format = '/TPU:{}' if 'TPU' in str(jax.devices()[0]) else '/GPU:{}'
  for i, tensor in enumerate(input_tensors):
    with tf.device(device_format.format(i)):
      tensor_on_device = tf.identity(tensor)
      res = run_inception_jit(
          tensor_on_device, inception_model, num_batches=num_batches,
          inceptionv3=inceptionv3)

      if not inceptionv3:
        pool3.append(res['pool_3'])
        logits.append(res['logits'])  # pytype: disable=attribute-error
      else:
        pool3.append(res)

  with tf.device('/CPU'):
    return {
        'pool_3': tf.concat(pool3, axis=0),
        'logits': tf.concat(logits, axis=0) if not inceptionv3 else None
    }

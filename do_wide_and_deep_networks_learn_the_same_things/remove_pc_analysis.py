# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Computes internal CKA with top PC of layer activations removed."""

from itertools import combinations
import os
import pickle

from absl import app
from absl import flags
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import entropy
from sklearn.decomposition import TruncatedSVD, randomized_svd
import tensorflow.compat.v2 as tf
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
from do_wide_and_deep_networks_learn_the_same_things.cifar_train import load_test_data
from do_wide_and_deep_networks_learn_the_same_things.efficient_CKA import *

tf.enable_v2_behavior()


FLAGS = flags.FLAGS
flags.DEFINE_integer('cka_batch', 256, 'Batch size used to approximate CKA')
flags.DEFINE_integer('cka_iter', 10,
                     'Number of iterations to run minibatch CKA approximation')
flags.DEFINE_string('model_dir', '', 'Path to where the trained model is saved')
flags.DEFINE_integer('model_depth', 14,
                     'Only run analysis for models of this depth')
flags.DEFINE_integer('model_width', 1,
                     'Only run analysis for models of this width multiplier')
flags.DEFINE_string('experiment_dir', None,
                    'Path to where the trained model is saved')
flags.DEFINE_integer('model_depth2', 20,
                     'Only run analysis for models of this depth')
flags.DEFINE_integer('model_width2', 2,
                     'Only run analysis for models of this width multiplier')
flags.DEFINE_integer('epoch', 300, 'Use ckpt from this epoch')


def get_activations(images, model):
  """Return a list of activations obtained from a model on a set of images."""
  input_layer = model.input
  layer_outputs = [layer.output for layer in model.layers]
  get_layer_outputs = K.function(input_layer, layer_outputs)
  activations = get_layer_outputs(images)
  return activations


@tf.function(experimental_compile=True)
def power_iterate(x, intermediate_dtype=None, tol=1e-5, verbose=False):
  """Perform power iteration.

  Args:
    x: A matrix.
    intermediate_dtype: Data type for intermediate computation. If `None`, use
      the same dtype as `x`.
    tol: Convergence tolerance. How small the L2 norm between successive
      singular veectors should be before termination.
    verbose: If True (and not compiled), print number of iterations.

  Returns:
  """
  orig_dtype = x.dtype
  if intermediate_dtype is None:
    intermediate_dtype = orig_dtype

  # We want XX^T to be as small as possible.
  transposed = False
  if x.shape[0] > x.shape[1]:
    transposed = True
    x = tf.transpose(x)

  x = tf.cast(x, intermediate_dtype)
  xxt = tf.matmul(x, x, transpose_b=True)

  # Set up loop variables.
  u = tf.random.stateless_normal((x.shape[0], 1), (1337, 1337),
                                 dtype=intermediate_dtype)
  old_u = tf.fill((x.shape[0], 1),
                  tf.constant(float('inf'), dtype=intermediate_dtype))
  u_norm = tf.constant(0.)
  i = 0

  # Perform power iteration until the eigenvector converges.
  while tf.reduce_max(tf.linalg.norm(u - old_u)) > tol:
    old_u = u
    u = tf.matmul(xxt, u)
    u_norm = tf.linalg.norm(u)
    u /= u_norm
    i += 1

  if verbose:
    print('{} iterations'.format(i))
  s = tf.sqrt(u_norm)
  v = tf.nn.l2_normalize(tf.matmul(x, u, transpose_a=True))

  # Swap u and v if we did a transpose in the beginning.
  if transposed:
    u, v = v, u

  return tf.cast(s, orig_dtype), tf.cast(u, orig_dtype), tf.cast(v, orig_dtype)


def remove_first_pc(x, **kwargs):
  """Remove first principal component from x."""
  s, u, v = power_iterate(x, **kwargs)
  return x - tf.matmul(u, v, transpose_b=True) * s


def compute_cka_internal_no_top_component(model_dir,
                                          data_path=None,
                                          dataset_name='cifar10',
                                          use_batch=True):
  """Compute CKA score of each layer in a model to every other layer in the same model, after removing the effect of top component."""
  if dataset_name == 'cifar10':
    filename = 'cka_within_model_remove_first_pc_%d.pkl' % FLAGS.cka_batch
  else:
    suffix = dataset_name.split('_')[-1]
    filename = 'cka_within_model_remove_first_pc_%d_%s.pkl' % (FLAGS.cka_batch,
                                                               suffix)
  out_dir = os.path.join(model_dir, filename)
  if tf.io.gfile.exists(out_dir):
    return

  model = tf.keras.models.load_model(model_dir)
  n_layers = len(model.layers)
  cka = MinibatchCKA(n_layers)
  if use_batch:
    for _ in range(FLAGS.cka_iter):
      dataset = load_test_data(
          FLAGS.cka_batch,
          shuffle=True,
          data_path=data_path,
          dataset_name=dataset_name)
      for images, _ in dataset:
        acts = get_activations(images, model)
        for i, act in enumerate(acts):
          act = act.reshape(act.shape[0], -1).T
          act -= np.mean(act, axis=0)
          acts[i] = remove_first_pc(act)
        cka.update_state(acts)
  else:
    dataset = load_test_data(
        FLAGS.cka_batch, data_path=data_path, dataset_name=dataset_name)
    all_images = tf.concat([x[0] for x in dataset], 0)
    acts = get_activations(all_images, model)
    for i, act in enumerate(acts):
      act = act.reshape(act.shape[0], -1).T
      act -= np.mean(act, axis=0)
      acts[i] = remove_first_pc(act)
    cka.update_state(acts)

  heatmap = cka.result().numpy()
  with tf.io.gfile.GFile(out_dir, 'wb') as f:
    pickle.dump(heatmap, f)


def main(argv):
  try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
      assert device.device_type != 'GPU'
  except:
    pass

  compute_cka_internal_no_top_component(FLAGS.experiment_dir)


if __name__ == '__main__':
  app.run(main)

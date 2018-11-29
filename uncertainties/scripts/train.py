# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Script to launch the algorithms on the last layer uncertainties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf


import uncertainties.sources.cifar.cifar_input_python as cifar_input
import uncertainties.sources.mnist.mnist_input as mnist_input
import uncertainties.sources.models.bootstrap as bootstrap
import uncertainties.sources.models.dropout as dropout
import uncertainties.sources.models.precond as precond
import uncertainties.sources.models.simple as simple
import uncertainties.sources.postprocessing.metrics as metrics
import uncertainties.sources.postprocessing.postprocess as postprocess
import uncertainties.sources.utils.util as util
import gin.tf


FLAGS = flags.FLAGS


flags.DEFINE_string('baseroute', None, 'Directory for inputs.')
flags.DEFINE_string('workdir', None, 'Directory for outputs.')
flags.DEFINE_string('dataset', None,
                    'Name of the dataset {cifar10, cifar100, mnist}')
flags.DEFINE_string('algorithm', None,
                    'Name of the algorithm'
                    '{simple, precond, dropout, bootstrap}')
flags.DEFINE_multi_string('gin_config', [],
                          'List of paths to the config files.')

flags.DEFINE_multi_string('gin_bindings', [],
                          'Newline separated list of Gin parameter bindings.')


def main(unused_argv):

  # Enable passing of configurable references from xmanager.
  # Xmanager passes them as strings while gin wants them unquoted.
  FLAGS.gin_bindings = [
      x if "@" not in x else x.replace("\"", "") for x in FLAGS.gin_bindings  # pylint: disable=g-inconsistent-quotes
  ]
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  if FLAGS.dataset == 'mnist':
    train_mnist()
  elif FLAGS.dataset == 'cifar10':
    train_cifar10()
  elif FLAGS.dataset == 'cifar100':
    train_cifar100()
  else:
    raise NotImplementedError('Dataset not implemented')


@gin.configurable('train_mnist')
def train_mnist(features_mnist_train=gin.REQUIRED,
                features_mnist_test=gin.REQUIRED,
                data_path_mnist=gin.REQUIRED,
                features_notmnist_test=gin.REQUIRED,
                labels_notmnist_test=gin.REQUIRED,
                model_dir=gin.REQUIRED,
                dim_input=gin.REQUIRED,
                num_classes=gin.REQUIRED
               ):
  """Training function."""
  # Define the paths
  data_path_mnist = os.path.join(FLAGS.baseroute, data_path_mnist)
  features_mnist_train = os.path.join(FLAGS.baseroute, features_mnist_train)
  features_mnist_test = os.path.join(FLAGS.baseroute, features_mnist_test)
  features_notmnist_test = os.path.join(FLAGS.baseroute, features_notmnist_test)
  labels_notmnist_test = os.path.join(FLAGS.baseroute, labels_notmnist_test)
  model_dir = os.path.join(FLAGS.baseroute, model_dir)
  # Load the features for mnist
  (_, y_train), (_, y_test) = mnist_input.load_data(data_path_mnist)
  with tf.gfile.Open(features_mnist_train, 'r') as f:
    x_train = np.load(f)
  with tf.gfile.Open(features_mnist_test, 'r') as f:
    x_test = np.load(f)
  dataset = (x_train, y_train), (x_test, y_test)

  model = build_model(dataset, model_dir, dim_input, num_classes)

  # Load notmnist features and labels test dataset
  with tf.gfile.Open(features_notmnist_test, 'r') as f:
    x_notmnist = np.load(f)
  with tf.gfile.Open(labels_notmnist_test, 'r') as f:
    y_notmnist = np.load(f)
  y_notmnist = tf.keras.utils.to_categorical(y_notmnist, num_classes=10)

  # Compute output probabilities on x_test and x_test_notmnist
  # Warning: take time, pass over all the saved weights.
  x = np.vstack((x_test, x_notmnist))
  model.predict(x)

  # Dictionary for y - Metrics
  y_dic = {'mnist': y_test, 'notmnist': y_notmnist}

  # Postprocessing and metrics
  if FLAGS.algorithm in ['simple', 'dropout', 'precond']:
    postprocess.postprocess_mnist(FLAGS.workdir)
    for dataset_str in ['mnist', 'notmnist']:
      path_postprocess = os.path.join(FLAGS.workdir, dataset_str)
      metrics.Metrics(y_dic[dataset_str], path_postprocess)

  # Write the gin config in the working directory
  util.save_gin(os.path.join(FLAGS.workdir, 'gin_configuration.txt'))
  util.write_gin(FLAGS.workdir)


@gin.configurable('train_cifar10')
def train_cifar10(features_cifar10_train=gin.REQUIRED,
                  features_cifar10_test=gin.REQUIRED,
                  data_path_cifar10=gin.REQUIRED,
                  features_cifar10_train_distorted=gin.REQUIRED,
                  distorted=gin.REQUIRED,
                  model_dir=gin.REQUIRED,
                  dim_input=gin.REQUIRED,
                  num_classes=gin.REQUIRED
                 ):
  """Training function."""
  # Define the paths
  features_cifar10_train = os.path.join(FLAGS.baseroute, features_cifar10_train)
  features_cifar10_test = os.path.join(FLAGS.baseroute, features_cifar10_test)
  data_path_cifar10 = os.path.join(FLAGS.baseroute, data_path_cifar10)
  features_cifar10_train_distorted = os.path.join(
      FLAGS.baseroute, features_cifar10_train_distorted)
  model_dir = os.path.join(FLAGS.baseroute, model_dir)
  # Load the features for cifar10
  (_, y_train), (_, y_test) = cifar_input.load_data(distorted,
                                                    data_path_cifar10,
                                                    'cifar10')
  if distorted:
    with tf.gfile.Open(features_cifar10_train_distorted, 'r') as f:
      x_train = np.load(f)
  else:
    with tf.gfile.Open(features_cifar10_train, 'r') as f:
      x_train = np.load(f)
  with tf.gfile.Open(features_cifar10_test, 'r') as f:
    x_test = np.load(f)
  dataset = (x_train, y_train), (x_test, y_test)

  model = build_model(dataset, model_dir, dim_input, num_classes)

  # Compute output probabilities on x_test
  # Warning: take time, pass over all the saved weights.
  model.predict(x_test)

  # Postprocessing and metrics
  if FLAGS.algorithm in ['simple', 'dropout', 'precond']:
    postprocess.postprocess_cifar(FLAGS.workdir, 'cifar10')
    path_postprocess = os.path.join(FLAGS.workdir, 'cifar10')
    metrics.Metrics(y_test, path_postprocess)

  # Write the gin config in the working directory
  util.save_gin(os.path.join(FLAGS.workdir, 'gin_configuration.txt'))
  util.write_gin(FLAGS.workdir)


@gin.configurable('train_cifar100')
def train_cifar100(features_cifar100_train=gin.REQUIRED,
                   features_cifar100_test=gin.REQUIRED,
                   data_path_cifar100=gin.REQUIRED,
                   features_cifar100_train_distorted=gin.REQUIRED,
                   distorted=gin.REQUIRED,
                   model_dir=gin.REQUIRED,
                   dim_input=gin.REQUIRED,
                   num_classes=gin.REQUIRED
                  ):
  """Training function."""
  # Define the paths
  features_cifar100_train = os.path.join(FLAGS.baseroute,
                                         features_cifar100_train)
  features_cifar100_test = os.path.join(FLAGS.baseroute, features_cifar100_test)
  data_path_cifar100 = os.path.join(FLAGS.baseroute, data_path_cifar100)
  features_cifar100_train_distorted = os.path.join(
      FLAGS.baseroute, features_cifar100_train_distorted)
  model_dir = os.path.join(FLAGS.baseroute, model_dir)
  # Load the features for cifar100
  (_, y_train), (_, y_test) = cifar_input.load_data(distorted,
                                                    data_path_cifar100,
                                                    'cifar100')
  if distorted:
    with tf.gfile.Open(features_cifar100_train_distorted, 'r') as f:
      x_train = np.load(f)
  else:
    with tf.gfile.Open(features_cifar100_train, 'r') as f:
      x_train = np.load(f)
  with tf.gfile.Open(features_cifar100_test, 'r') as f:
    x_test = np.load(f)
  dataset = (x_train, y_train), (x_test, y_test)

  model = build_model(dataset, model_dir, dim_input, num_classes)

  # Compute output probabilities on x_test
  # Warning: take time, pass over all the saved weights.
  model.predict(x_test)

  # Postprocessing and metrics
  if FLAGS.algorithm in ['simple', 'dropout', 'precond']:
    postprocess.postprocess_cifar(FLAGS.workdir, 'cifar100')
    path_postprocess = os.path.join(FLAGS.workdir, 'cifar100')
    metrics.Metrics(y_test, path_postprocess)

  # Write the gin config in the working directory
  util.save_gin(os.path.join(FLAGS.workdir, 'gin_configuration.txt'))
  util.write_gin(FLAGS.workdir)


def build_model(dataset, model_dir, dim_input, num_classes):
  """Create the Bayesian Neural Network and sample from it.

  Args:
    dataset: dataset
    model_dir: directory of the model to load the weights of the pretrained
               neural network
    dim_input: dimension of the input vector
    num_classes: number of classes
  Returns:
    model: model
  """
  if FLAGS.algorithm == 'simple':
    model = simple.LastLayerBayesian(dataset, FLAGS.workdir, model_dir,
                                     dim_input, num_classes)
  elif FLAGS.algorithm == 'precond':
    model = precond.LastLayerBayesianPrecond(dataset, FLAGS.workdir, model_dir,
                                             dim_input, num_classes)
  elif FLAGS.algorithm == 'dropout':
    model = dropout.LastLayerDropout(dataset, FLAGS.workdir, model_dir,
                                     dim_input, num_classes)
  elif FLAGS.algorithm == 'bootstrap':
    model = bootstrap.LastLayerBootstrap(dataset, FLAGS.workdir, model_dir,
                                         dim_input, num_classes)
  else:
    raise NotImplementedError('Algorithm not implemented')
  _, _, sampled_weights = model.sample()

  # Saving the weights
  if FLAGS.algorithm in ['simple', 'precond']:
    str_file = 'sampled_weights_' + model.sampler + '.npy'
  elif FLAGS.algorithm == 'bootstrap':
    str_file = 'sampled_weights_w{}.npy'.format(model.worker_id)
  else:
    str_file = 'sampled_weights.npy'
  data_path = os.path.join(FLAGS.workdir, str_file)
  with tf.gfile.Open(data_path, 'wb') as f:
    np.save(f, sampled_weights)

  return model


if __name__ == '__main__':
  app.run(main)

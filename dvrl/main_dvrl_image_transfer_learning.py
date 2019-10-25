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

"""Main experiment of DVRL with transfer learning for image data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import keras
from keras import applications
from keras import layers
from keras import models
import numpy as np
from sklearn import linear_model
import tensorflow as tf

from dvrl import data_loading
from dvrl import dvrl
from dvrl import dvrl_metrics


def main(args):
  """Main function of DVRL with transfer learning for image data.

  Main function of DVRL for corrupted sample discovery and robust learning
  with transfer learning for image data.

  Args:
    args: data_name, train_no, valid_no, noise_rate,
          normalization, network parameters
  """
  # Data name (either cifar10 or cifar100)
  data_name = args.data_name

  # The number of training and validation samples
  dict_no = dict()
  dict_no['train'] = args.train_no
  dict_no['valid'] = args.valid_no
  dict_no['test'] = args.test_no

  # Additional noise ratio
  noise_rate = args.noise_rate

  # Checkpoint file name
  checkpoint_file_name = args.checkpoint_file_name

  # Data loading and label corruption
  noise_idx = data_loading.load_image_data(data_name, dict_no, noise_rate)
  # noise_idx: ground truth noisy label indices

  print('Finished data loading.')

  # Extracts features and labels.
  x_train, y_train, x_valid, y_valid, x_test, y_test = \
  data_loading.load_image_data_from_file('train.npz', 'valid.npz', 'test.npz')
  print('Finished data preprocess.')

  # Encodes samples
  # The preprocessing function used on the pre-training dataset is also
  # applied while encoding the inputs.
  preprocess_function = applications.inception_v3.preprocess_input
  input_shape = (299, 299)

  def encoder_model(architecture='inception_v3', pre_trained_dataset='imagenet',
                    downsample_factor=8):
    """Returns encoder model.

    Defines the encoder model to learn the representations for image dataset.
    In this example, we are considering the InceptionV3 model trained on
    ImageNet dataset, followed by simple average pooling-based downsampling.

    Args:
      architecture: Base architecture of encoder model (e.g. 'inception_v3')
      pre_trained_dataset: The dataset used to pre-train the encoder model
      downsample_factor: Downsample factor for the outputs

    Raises:
      NameError: Returns name errors if architecture is not 'inception_v3'
    """
    tf_input = layers.Input(shape=(input_shape[0], input_shape[1], 3))
    if architecture == 'inception_v3':
      model = applications.inception_v3.InceptionV3(
          input_tensor=tf_input, weights=pre_trained_dataset, include_top=False)
      output_pooled = \
          layers.AveragePooling2D((downsample_factor, downsample_factor),
                                  strides=(downsample_factor,
                                           downsample_factor))(model.output)
    else:
      raise NameError('Invalid architecture')
    return models.Model(model.input, output_pooled)

  # Encodes training samples
  enc_x_train = \
      data_loading.encode_image(x_train,
                                encoder_model,
                                input_shape,
                                preprocess_function)
  # Encodes validation samples
  enc_x_valid = \
      data_loading.encode_image(x_valid,
                                encoder_model,
                                input_shape,
                                preprocess_function)
  # Encodes testing samples
  enc_x_test = \
      data_loading.encode_image(x_test,
                                encoder_model,
                                input_shape,
                                preprocess_function)

  print('Finished data encoding')

  # Run DVRL
  # Resets the graph
  tf.reset_default_graph()
  keras.backend.clear_session()

  # Network parameters
  parameters = dict()
  parameters['hidden_dim'] = args.hidden_dim
  parameters['comb_dim'] = args.comb_dim
  parameters['activation'] = tf.nn.relu
  parameters['iterations'] = args.iterations
  parameters['layer_number'] = args.layer_number
  parameters['batch_size'] = args.batch_size
  parameters['learning_rate'] = args.learning_rate
  parameters['inner_iterations'] = args.inner_iterations
  parameters['batch_size_predictor'] = args.batch_size_predictor

  # Defines problem
  problem = 'classification'

  # Defines predictive model
  pred_model = keras.models.Sequential()
  pred_model.add(keras.layers.Dense(len(set(y_train)), activation='softmax'))
  pred_model.compile(optimizer='adam', loss='categorical_crossentropy',
                     metrics=['accuracy'])

  # Flags for using stochastic gradient descent / pre-trained model
  flags = {'sgd': True, 'pretrain': False}

  # Initalizes DVRL
  dvrl_class = dvrl.Dvrl(enc_x_train, y_train, enc_x_valid, y_valid,
                         problem, pred_model, parameters,
                         checkpoint_file_name, flags)

  # Trains DVRL
  dvrl_class.train_dvrl('accuracy')

  print('Finished DVRL training.')

  # Outputs
  # Data valuation
  dve_out = dvrl_class.data_valuator(enc_x_train, y_train)

  print('Finished data valuation.')

  # Evaluations
  # Evaluation model
  eval_model = linear_model.LogisticRegression(solver='lbfgs',
                                               multi_class='auto',
                                               max_iter=2000)

  # 1. Robust learning (DVRL-weighted learning)
  robust_perf = dvrl_metrics.learn_with_dvrl(dve_out, eval_model,
                                             enc_x_train, y_train,
                                             enc_x_valid, y_valid,
                                             enc_x_test, y_test, 'accuracy')

  print('DVRL-weighted learning performance: ' + str(np.round(robust_perf, 4)))

  # 2. Performance after removing high/low values
  _ = dvrl_metrics.remove_high_low(dve_out, eval_model, enc_x_train, y_train,
                                   enc_x_valid, y_valid, enc_x_test, y_test,
                                   'accuracy', plot=True)

  # 3. Corrupted sample discovery
  # If noise_idx variable exist (explicit indices for noisy sample)
  # and noise_rate is positive value.

  if noise_rate > 0:

    # Evaluates corrupted_sample_discovery
    # and plot corrupted sample discovery results
    _ = dvrl_metrics.discover_corrupted_sample(dve_out,
                                               noise_idx, noise_rate,
                                               plot=True)


if __name__ == '__main__':

  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['cifar10', 'cifar100'],
      help='data name (cifar10 or cifar100)',
      default='cifar10',
      type=str)
  parser.add_argument(
      '--train_no',
      help='number of training samples',
      default=4000,
      type=int)
  parser.add_argument(
      '--valid_no',
      help='number of validation samples',
      default=1000,
      type=int)
  parser.add_argument(
      '--test_no',
      help='number of testing samples',
      default=2000,
      type=int)
  parser.add_argument(
      '--noise_rate',
      help='label corruption ratio',
      default=0.2,
      type=float)
  parser.add_argument(
      '--hidden_dim',
      help='dimensions of hidden states',
      default=100,
      type=int)
  parser.add_argument(
      '--comb_dim',
      help='dimensions of hidden states after combinding with prediction diff',
      default=10,
      type=int)
  parser.add_argument(
      '--layer_number',
      help='number of network layers',
      default=5,
      type=int)
  parser.add_argument(
      '--iterations',
      help='number of iterations',
      default=2000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='number of batch size for RL',
      default=2000,
      type=int)
  parser.add_argument(
      '--inner_iterations',
      help='number of iterations for predictor',
      default=100,
      type=int)
  parser.add_argument(
      '--batch_size_predictor',
      help='number of batch size for predictor',
      default=256,
      type=int)
  parser.add_argument(
      '--learning_rate',
      help='learning rates for RL',
      default=0.01,
      type=float)
  parser.add_argument(
      '--checkpoint_file_name',
      help='file name for saving and loading the trained model',
      default='./tmp/model.ckpt',
      type=str)

  args_in = parser.parse_args()

  # Calls main function
  main(args_in)

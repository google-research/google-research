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

"""Tests for google_research.google_research.cold_posterior_bnn.core.priorfactory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from cold_posterior_bnn.core import priorfactory


class PriorfactoryTest(tf.test.TestCase):

  def test_weight(self):
    """Test that we require the user to provide a weight value."""
    with self.assertRaises(ValueError):
      _ = priorfactory.DefaultPriorFactory()

  def test_fixed_prior(self):
    """Test that we can override prior choice using prior_dict."""
    prior_dict = {
        'dense_A': {
            'bias_regularizer': {
                'class_name': 'NormalRegularizer',
                'config': {'stddev': 2.0, 'weight': 0.5},
            },
        }
    }
    pfac = priorfactory.DefaultPriorFactory(weight=1.0, prior_dict=prior_dict)
    layer = pfac(tf.keras.layers.Dense(24, name='dense_A'))  # use prior_dict
    config = layer.get_config()
    self.assertIsNotNone(config['bias_regularizer'], msg='No prior applied.')
    self.assertEqual(config['bias_regularizer']['class_name'],
                     'NormalRegularizer',
                     msg='Wrong prior applied when prior_dict is used.')
    self.assertAlmostEqual(config['bias_regularizer']['config']['stddev'], 2.0,
                           delta=1.0e-6, msg='Wrong prior parameter value.')
    self.assertAlmostEqual(config['bias_regularizer']['config']['weight'], 0.5,
                           delta=1.0e-6, msg='Wrong prior parameter weight.')
    self.assertIsNone(config['kernel_regularizer'],
                      msg='Prior applied when it should not be applied.')

    layer2 = pfac(tf.keras.layers.Dense(24, name='dense_B'))  # no prior_dict
    config2 = layer2.get_config()
    self.assertIsNotNone(config2['kernel_regularizer'], msg='No prior applied.')
    self.assertIsNotNone(config2['bias_regularizer'], msg='No prior applied.')

  def test_flat_factory(self):
    """Test that the flat prior factory does not modify the layer."""
    pfac = priorfactory.FlatPriorFactory(weight=1.0)
    layer = tf.keras.layers.Dense(24, name='dense')
    config = layer.get_config()
    layer_out = pfac(layer)
    config_out = layer_out.get_config()

    self.assertLen(config, len(config_out), msg='Layer descriptions differ.')
    for key in config:
      self.assertEqual(config[key], config_out[key],
                       msg='Element mismatch in layer after going through '
                       'FlatPriorFactory.')

  def test_gaussian_factory(self):
    pfac = priorfactory.GaussianPriorFactory(prior_stddev=2.0, weight=1.0)
    layer = tf.keras.layers.Dense(24, name='dense')
    layer_out = pfac(layer)
    config_out = layer_out.get_config()
    self.assertAlmostEqual(config_out['bias_regularizer']['config']['stddev'],
                           2.0, delta=1.0e-6, msg='Wrong prior parameter.')

  def test_shifted_gaussian_factory(self):
    pfac = priorfactory.ShiftedGaussianPriorFactory(
        prior_mean=-0.3, prior_stddev=2.0, weight=1.0)
    layer = tf.keras.layers.Dense(24, name='dense')
    layer_out = pfac(layer)
    config_out = layer_out.get_config()
    self.assertAlmostEqual(config_out['bias_regularizer']['config']['mean'],
                           -0.3, delta=1.0e-6, msg='Wrong prior parameter.')

    self.assertAlmostEqual(config_out['bias_regularizer']['config']['stddev'],
                           2.0, delta=1.0e-6, msg='Wrong prior parameter.')

  def test_disabled_bias(self):
    pfac = priorfactory.GaussianPriorFactory(prior_stddev=2.0, weight=1.0)
    layer = tf.keras.layers.Dense(24, name='dense', use_bias=False)
    config = layer.get_config()
    self.assertFalse(config['use_bias'])
    layer_out = pfac(layer)
    config_out = layer_out.get_config()
    self.assertFalse(config_out['use_bias'])


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()

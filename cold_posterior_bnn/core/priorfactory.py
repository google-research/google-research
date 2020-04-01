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

# Lint as: python3
"""Default priors for Bayesian neural networks.

Prior factories can create suitable priors given Keras layers as input.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import logging
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from cold_posterior_bnn.core import frn
from cold_posterior_bnn.core import model as bnnmodel

layers = tf.keras.layers


class PriorFactory(object):
  """Prior factory base class.

  The prior factory is a helper class that makes the task of adding proper
  prior distributions to Keras models easy.

  Examples:
    The following code instantiates a prior factory object and shows how to
    wrap a newly created layer in order to add proper default priors to all
    parameters of the layer.

    >>> pfac = DefaultPriorFactory(weight=1.0/total_train_size)
    >>> dense = pfac(tf.keras.layers.Dense(32))

  """

  def __init__(self, weight=0.0, prior_dict=None):
    """Construct a new PriorFactory object.

    Args:
      weight: prior weight, typically 1.0/total_train_sample_size for Bayesian
        neural networks.  Must be >0.0.
      prior_dict: dict, containing as keys layer.name values and as value a dict
        describing the regularizers to add to the respective layer.
        The prior_dict can be used to override choices made by other
        PriorFactory classes, i.e. it always takes precedence in determining
        priors.

    Raises:
      ValueError: invalid value for weight keyword argument.
    """
    # The weight parameter is critical so we force the user to set a value
    if weight <= 0.0:
      raise ValueError('You must provide a "weight" argument to the prior '
                       'factory.  Typically weight=1.0/total_train_size, '
                       'where total_train_size is the number of iid training '
                       'instances.')

    self.weight = weight
    self.prior_dict = prior_dict

  def _replace(self, config, rdict):
    """Replace a key in a tf.keras.layers.Layer config dictionary.

    This method replaces a regularizer key in a layer.get_config() dictionary
    with specified elements from the rdict regularization dictionary.

    Examples:
      >>> embedding = tf.keras.layers.Embedding(5000, 512)
      >>> config = embedding.get_config()
      >>> pfac = DefaultPriorFactory(weight=1.0/50000.0)
      >>> rdict = {'embeddings_regularizer': {
              'class_name': 'NormalRegularizer',
              'config': {'stddev': 0.1, 'weight': 1.0/50000.0} } }
      >>> pfac._replace(config, rdict)

    Args:
      config: dict, containing the layer.get_config() dictionary to modify.
      rdict: dict, regularizer keys/values to put into config dictionary.
    """
    # If fixed prior is used, replace rdict using prior dictionary
    layer_name = config['name']
    if (self.prior_dict is not None) and (layer_name in self.prior_dict):
      logging.info('Using regularizer for layer "%s" from prior_dict',
                   layer_name)
      rdict = self.prior_dict[layer_name]

    if rdict is None:
      return

    for name in rdict:
      if config[name] is not None:
        logging.warn('Warning: Overriding regularizer from layer "%s"s %s',
                     layer_name, name)

      config[name] = rdict[name]

  def _update_prior(self, layer, config):
    """Update the config dictionary for the given layer.

    This abstract method must be overridden by concrete implementations in
    derived classes.

    The method's job is to select for a given 'layer' the corresponding priors
    that are suitable.

    Args:
      layer: tf.keras.layers.Layer class.
      config: the layer.get_config() dictionary.  This argument must be
        modified.  The modified dictionary will then be used to reconstruct the
        layer.
    """
    raise NotImplementedError('Users must override the _update_prior method '
                              'of PriorFactory')

  def __call__(self, layer):
    """Add a prior to the newly constructed input layer.

    Args:
      layer: tf.keras.layers.Layer that has just been constructed (not built, no
        graph).

    Returns:
      layer_out: the layer with a suitable prior added.
    """
    if not layer.trainable:
      return layer

    # Obtain serialized layer representation and replace priors
    config = layer.get_config()
    self._update_prior(layer, config)

    # Reconstruct prior from updated serialized representation
    with bnnmodel.bnn_scope():
      layer_out = type(layer).from_config(config)

    return layer_out


class FlatPriorFactory(PriorFactory):
  """Flat prior factory.

  This does not add any explicit prior to any layer except those specified
  by the 'prior_dict' argument.  Therefore, the prior corresponds to the
  improper flat prior.
  """

  def __init__(self, **kwargs):
    super(FlatPriorFactory, self).__init__(**kwargs)

  def _update_prior(self, layer, config):
    """Use None, resulting in a flat prior."""
    self._replace(config, None)

DEFAULT_NORMAL_STDDEV = 0.5
DEFAULT_CAUCHY_SCALE = 1.0
DEFAULT_LAPLACE_STDDEV = 1.0
DEFAULT_HE_NORMAL_SCALE = 1.0
DEFAULT_GLOROT_NORMAL_SCALE = 1.0


class DefaultPriorFactory(PriorFactory):
  """Default prior factory for Bayesian neural networks.

  This class contains a selection of suitable default priors suitable for
  Bayesian neural networks.
  """

  def __init__(self, **kwargs):
    super(DefaultPriorFactory, self).__init__(**kwargs)

  def normal(self, _):
    normal_dict = {
        'class_name': 'NormalRegularizer',
        'config': {'stddev': DEFAULT_NORMAL_STDDEV, 'weight': self.weight},
    }
    return normal_dict

  def he_normal(self, _):
    he_normal_dict = {
        'class_name': 'HeNormalRegularizer',
        'config': {'scale': DEFAULT_HE_NORMAL_SCALE, 'weight': self.weight},
    }
    return he_normal_dict

  def glorot_normal(self, _):
    glorot_normal_dict = {
        'class_name': 'GlorotNormalRegularizer',
        'config': {'scale': DEFAULT_GLOROT_NORMAL_SCALE, 'weight': self.weight},
    }
    return glorot_normal_dict

  def cauchy(self, _):
    cauchy_dict = {
        'class_name': 'CauchyRegularizer',
        'config': {'scale': DEFAULT_CAUCHY_SCALE, 'weight': self.weight}
    }
    return cauchy_dict

  def laplace(self, _):
    laplace_dict = {
        'class_name': 'LaplaceRegularizer',
        'config': {'stddev': DEFAULT_LAPLACE_STDDEV, 'weight': self.weight},
    }
    return laplace_dict

  def _update_prior(self, layer, config):
    # Bias terms: use heavy-tailed Laplace prior; here the prior choice really
    #   matters.
    # Dense matrices: fixed Normal prior.  Good choice?
    if isinstance(layer, layers.Dense):
      self._replace(config, {
          'kernel_regularizer': self.he_normal(layer),
          'bias_regularizer': self.cauchy(layer),
      })
    elif isinstance(layer, layers.Embedding):
      self._replace(config, {
          'embeddings_regularizer': self.normal(layer),
      })
    elif isinstance(layer, layers.Conv1D):
      self._replace(config, {
          'kernel_regularizer': self.he_normal(layer),
          'bias_regularizer': self.cauchy(layer),
      })
    elif isinstance(layer, layers.Conv2D):
      self._replace(config, {
          'kernel_regularizer': self.he_normal(layer),
          'bias_regularizer': self.cauchy(layer),
      })
    elif isinstance(layer, layers.LSTM):
      self._replace(config, {
          'kernel_regularizer': self.he_normal(layer),
          'recurrent_regularizer': self.he_normal(layer),
          'bias_regularizer': self.cauchy(layer),
      })
    elif isinstance(layer, frn.FRN):
      self._replace(config, {
          'tau_regularizer': self.cauchy(layer),
          'beta_regularizer': self.cauchy(layer),
          'gamma_regularizer': self.cauchy(layer),
      })
    elif isinstance(layer, frn.TLU):
      self._replace(config, {
          'tau_regularizer': self.cauchy(layer),
      })
    else:
      logging.warning('Layer type "%s" not found', type(layer))


DEFAULT_GAUSSIAN_PFAC_STDDEV = 1.0


class GaussianPriorFactory(PriorFactory):
  """Gaussian prior factory for Bayesian neural networks.

  This prior was used in [Zhang et al., 2019].
  """

  def __init__(self, prior_stddev=DEFAULT_GAUSSIAN_PFAC_STDDEV, **kwargs):
    super(GaussianPriorFactory, self).__init__(**kwargs)
    self.prior_stddev = prior_stddev

  def normal(self, _):
    normal_dict = {
        'class_name': 'NormalRegularizer',
        'config': {'stddev': self.prior_stddev, 'weight': self.weight},
    }
    return normal_dict

  def _update_prior(self, layer, config):

    if isinstance(layer, layers.Dense) or isinstance(layer, layers.Conv1D) or \
        isinstance(layer, layers.Conv2D):
      self._replace(config, {
          'kernel_regularizer': self.normal(layer),
          'bias_regularizer': self.normal(layer),
      })
    elif isinstance(layer, layers.Embedding):
      self._replace(config, {
          'embeddings_regularizer': self.normal(layer),
      })
    elif isinstance(layer, layers.LSTM):
      self._replace(config, {
          'kernel_regularizer': self.normal(layer),
          'recurrent_regularizer': self.normal(layer),
          'bias_regularizer': self.normal(layer),
      })
    else:
      logging.warning('Layer type "%s" not found', type(layer))

DEFAULT_SHIFTED_GAUSSIAN_PFAC_STDDEV = 1.0


class ShiftedGaussianPriorFactory(PriorFactory):
  """Shifted Gaussian (non-zero mean) prior factory for Bayesian neural networks.

  This prior can be used to center a Gaussian prior around a point estimate for
  the neural network. See prior.ShiftedNormalPrior for more information.
  """

  def __init__(self,
               prior_mean=0,
               prior_stddev=DEFAULT_SHIFTED_GAUSSIAN_PFAC_STDDEV,
               **kwargs):
    super(ShiftedGaussianPriorFactory, self).__init__(**kwargs)
    self.prior_mean = prior_mean
    self.prior_stddev = prior_stddev

  def normal(self, _):
    normal_dict = {
        'class_name': 'ShiftedNormalRegularizer',
        'config': {'mean': self.prior_mean,
                   'stddev': self.prior_stddev,
                   'weight': self.weight},
    }
    return normal_dict

  def _update_prior(self, layer, config):

    if isinstance(layer, layers.Dense) or isinstance(layer, layers.Conv1D) or \
        isinstance(layer, layers.Conv2D):
      self._replace(config, {
          'kernel_regularizer': self.normal(layer),
          'bias_regularizer': self.normal(layer),
      })
    elif isinstance(layer, layers.Embedding):
      self._replace(config, {
          'embeddings_regularizer': self.normal(layer),
      })
    elif isinstance(layer, layers.LSTM):
      self._replace(config, {
          'kernel_regularizer': self.normal(layer),
          'recurrent_regularizer': self.normal(layer),
          'bias_regularizer': self.normal(layer),
      })
    else:
      logging.warning('Layer type "%s" not found', type(layer))

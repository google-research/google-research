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

# Lint as: python2, python3
"""Training to forget nuisance variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl import app
from absl import flags
from absl import logging

import numpy as np
import pandas as pd
from scipy import sparse
from six.moves import range
from six.moves import zip
import six.moves.cPickle as pickle
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile

from correct_batch_effects_wdn import io_utils
from correct_batch_effects_wdn import transform
from tensorflow.python.ops import gen_linalg_ops  # pylint: disable=g-direct-tensorflow-import

INPUT_NAME = "inputs"
OUTPUT_NAME = "outputs"
DISCRIMINATOR_NAME = "discriminator"
CLASSIFIER_NAME = "classifier"
UNIT_GAUSS_NAME = "unit_gauss"
CRITIC_LAYER_NAME = "critic_layer"
POSSIBLE_LOSSES = (DISCRIMINATOR_NAME, CLASSIFIER_NAME, UNIT_GAUSS_NAME)
INPUT_KEY_INDEX = 0
REGISTRY_NAME = "params_registry_python_3"
WASSERSTEIN_NETWORK = "WassersteinNetwork"
WASSERSTEIN_2_NETWORK = "Wasserstein2Network"
MEAN_ONLY_NETWORK = "MeanOnlyNetwork"
WASSERSTEIN_SQRD_NETWORK = "WassersteinSqrdNetwork"
WASSERSTEIN_CUBED_NETWORK = "WassersteinCubedNetwork"
POSSIBLE_NETWORKS = [
    WASSERSTEIN_NETWORK, WASSERSTEIN_2_NETWORK, WASSERSTEIN_SQRD_NETWORK,
    WASSERSTEIN_CUBED_NETWORK, MEAN_ONLY_NETWORK
]

FLAGS = flags.FLAGS

flags.DEFINE_string("input_df", None, "Path to the embedding dataframe.")
flags.DEFINE_string("save_dir", None, "location of file to save.")
flags.DEFINE_integer("num_steps_pretrain", None, "Number of steps to pretrain.")
flags.DEFINE_integer("num_steps", None, "Number of steps (after pretrain).")
flags.DEFINE_integer("disc_steps_per_training_step", None, "Number critic steps"
                     "to use per main training step.")
flags.DEFINE_enum("network_type", "WassersteinNetwork", POSSIBLE_NETWORKS,
                  "Network to use. Can be WassersteinNetwork.")
flags.DEFINE_integer("batch_n", 10, "Number of points to use per minibatch"
                     "for each loss.")
flags.DEFINE_float("learning_rate", 1e-4, "Initial learning rate to use.")
flags.DEFINE_float("epsilon", 0.01, "Regularization for covariance.")
flags.DEFINE_integer("feature_dim", 192, "Number of feature dimensions.")
flags.DEFINE_integer("checkpoint_interval", 4000, "Frequency to save to file.")
flags.DEFINE_spaceseplist("target_levels", "compound",
                          "dataframe target levels.")
flags.DEFINE_spaceseplist("nuisance_levels", "batch",
                          "dataframe nuisance levels.")
flags.DEFINE_integer(
    "layer_width", 2, "Width of network to use for"
    "approximating the Wasserstein distance.")
flags.DEFINE_integer(
    "num_layers", 2, "Number of layers to use for"
    "approximating the Wasserstein distance.")
flags.DEFINE_string(
    "reg_dir", None, "Directory to registry file, or None to"
    "save in save_dir.")
flags.DEFINE_float("lambda_mean", 0., "Penalty for the mean term of the affine"
                   "transformation.")
flags.DEFINE_float("lambda_cov", 0., "Penalty for the cov term of the affine"
                   "transformation.")
flags.DEFINE_integer("seed", 42, "Seed to use for numpy.")
flags.DEFINE_integer("tf_seed", 42, "Seed to use for tensorflow.")
flags.DEFINE_float(
    "cov_fix", 0.001, "Multiple of identity to add if using"
    "Wasserstein-2 distance.")

################################################################################
##### Functions and classes for storing and retrieving data
################################################################################


def get_dense_arr(matrix):
  """Convert a sparse matrix to numpy array.

  Args:
    matrix (matrix, sparse matrix, or ndarray): input

  Returns:
    dense numpy array.
  """
  if sparse.issparse(matrix):
    return matrix.toarray()
  else:
    return np.array(matrix)


class DataShuffler(object):
  """Object to hold and shuffle data.

  Adapted from
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

  Attributes:
      inputs (ndarray): The inputs specified in __init__.
      outputs (ndarray): The outputs specified in __init__.
  """

  def __init__(self, inputs, outputs, random_state):
    """Inits DataShuffler given inputs, outputs, and a random state.

    Args:
      inputs (ndarray): 2-dimensional array containing the inputs, where each
        row is an individual input. The columns represent the different
        dimensions of an individual entry.
      outputs (ndarray): 2-dimensional array containing the outputs, where each
        row is an individual output.
      random_state (None or int): seed to feed to numpy.random.
    """
    assert inputs.shape[0] == outputs.shape[0]
    self.inputs = inputs
    self.outputs = outputs
    self._epochs_completed = -1
    self._num_examples = inputs.shape[0]
    self._random_state = random_state
    self._next_indices = []

  def next_batch(self, batch_size, shuffle=True):
    """Helper method for next_batch.

    Args:
      batch_size (int): Number of items to pick.
      shuffle (bool): whether to shuffle the data or not.

    Returns:
      A tuple of 2-dimensional ndarrays whose shape along the first axis is
        equal to batch_size. The rows in the first element correspond to inputs,
        and in the second element to the outputs.
    """
    indices = []
    while len(indices) < batch_size:
      if not self._next_indices:
        self._epochs_completed += 1
        self._next_indices = list(reversed(list(range(self._num_examples))))
        if shuffle:
          self._random_state.shuffle(self._next_indices)
      indices.append(self._next_indices.pop())
    return (get_dense_arr(self.inputs[indices]),
            get_dense_arr(self.outputs[indices]))


def _make_canonical_key(x):
  """Try to convert to a hashable type.

  First, if the input is a list of length 1, take its first component instead.
  Next, try to convert to a tuple if not hashable.

  Args:
    x (list, tuple, string, or None): input to convert to a key

  Returns:
    Hashable object
  """
  if isinstance(x, list) and len(x) == 1:
    x = x[0]
  if not isinstance(x, collections.Hashable):
    return tuple(x)
  else:
    return x


def split_df(df, columns_split):
  """Split a dataframe into two by column.

  Args:
    df (pandas dataframe): input dataframe to split.
    columns_split (int): Column at which to split the dataframes.

  Returns:
    df1, df2 (pandas dataframes): Split df into two dataframe based on column.
    The first has the first column_split-1 columns, and the second has columns
    columns_split onward.

  """
  return df.iloc[:, :columns_split], df.iloc[:, columns_split:]


def tuple_in_group_with_wildcards(needle, haystack):
  """Checks if needle is in haystack, allowing for wildcard components.

  Returns True if either needle or haystack is None, or needle is in haystack.
  Components in haystack are tuples. These tuples can have entries equal to None
  which serve as wildcard components.
  For example, if haystack = [(None, ...), ...], The first tuple in haystack
  has a wildcard for its first component.

  Args:
    needle (tuple): tuple to check if in group
    haystack (list): list of tuples of the same dimensions as tup.

  Returns:
    True if the tuple is in the group, False otherwise

  Raises:
    ValueError: Length of tup must match length of each tuple in group."
  """
  if needle is None or haystack is None:
    return True
  if any(len(needle) != len(it_needle) for it_needle in haystack):
    raise ValueError("Length of tup must match length of each tuple in group.")
  for it_needle in haystack:
    if all(needle_comp == it_needle_comp or it_needle_comp is None
           for needle_comp, it_needle_comp in zip(needle, it_needle)):
      return True
  return False


class DatasetHolder(object):
  """Object to hold datasets from which minibatches are sampled.

  Attributes:
    df_with_input_one_hot (pandas dataframe): Dataframe formatted as inputs
      including one-hot encoding for relevant categories.
    data_shufflers (instances of data_shuffler): The stored data shufflers.
    input_dim (int): Shape of original dataframe df used as an input.
    encoding_to_num (dict): Maps each encoding used to a unique integer.
  """

  def __init__(self,
               df,
               input_category_level=None,
               batch_input_info=None,
               input_output_type=np.array,
               random_state=None):
    """Inits DatasetHolder given a pandas dataframe and specifications.

    Args:
      df (pandas dataframe): all the input points, labeled by index.
      input_category_level (int, level name, or sequence of such, or None): Same
        as level used by pandas groupby method. Granularity level for labels
        added to inputs, unless no_input_labels=True. In that case the original
        inputs are provided
      batch_input_info (string or None): Formatting for the inputs
      input_output_type (class or module to store input and output information):
        Currently supports sparse.lil_matrix and np.array. np.array is the
        default setting.
      random_state (np.random.RandomState): Instance of RandomState used for
        shuffling data.

    Raises:
      ValueError: input_output_type not implemented. Use np.array or
        sparse.lil_matrix.
    """

    if input_output_type not in [np.array, sparse.lil_matrix]:
      raise ValueError("input_output_type not implemented. Use np.array or"
                       "sparse.lil_matrix.")
    self._input_output_type = input_output_type
    if input_output_type == np.array:
      self._zeros_maker = np.zeros
    else:
      self._zeros_maker = sparse.lil_matrix
    self._input_category_level = input_category_level
    self._batch_input_info = batch_input_info
    self.data_shufflers = collections.OrderedDict()
    self.input_dim = df.shape[1]
    self._num_input_categories = len(df.columns)
    self.df_with_input_one_hot = self.categorize_one_hot(
        df, category_level=self._input_category_level, drop_original=False)
    input_one_hot = self.df_with_input_one_hot.drop(
        self.df_with_input_one_hot.columns[list(range(self.input_dim))], axis=1)
    keys = [tuple(row) for row in input_one_hot.values]
    unique_keys = list(collections.OrderedDict.fromkeys(keys))
    self.encoding_to_num = {tuple(row): i for i, row in enumerate(unique_keys)}
    if random_state:
      self._random_state = random_state
    else:
      self._random_state = np.random.RandomState(seed=42)

  def get_random_state(self):
    """Get the random state.

    This is useful for testing.
    Returns:
      random_state (numpy random state).
    """
    return self._random_state

  def add_shufflers(self,
                    group_level=None,
                    label_level=None,
                    which_group_labels=None):
    """Add data_shuffler instances to the DatasetHolder.

    Args:
      group_level (int, level name, or sequence of such, or None): Same as level
        used by pandas groupby method, except when None. Indicates groups of
        inputs over which shufflers are created.
      label_level (int, level name, or sequence of such, or None): Same as level
        used by pandas groupby method, except when None. Indicates level to make
        distinct in outputs.
      which_group_labels (list or None): List to indicate which group_level
        values to use, or None to use all the possible ones.

    Returns:
      new_data_shufflers (dict): The newly added DataShuffler instances.
    """
    if group_level is None:
      groups = [[None, self.df_with_input_one_hot]]
    else:
      groups = self.df_with_input_one_hot.groupby(level=group_level)
    new_data_shufflers = {}
    for group_label, df_piece in groups:
      if not tuple_in_group_with_wildcards(group_label, which_group_labels):
        continue
      label_level_key = _make_canonical_key(label_level)
      data_shuffler_key = (group_label, label_level_key)
      data_shuffler = self.make_data_shuffler(
          df_piece, output_category_level=label_level)
      new_data_shufflers[data_shuffler_key] = data_shuffler
      self.data_shufflers.update(new_data_shufflers)
    return new_data_shufflers

  def make_data_shuffler(self,
                         df_with_input_one_hot,
                         output_category_level=None):
    """Generate a data shuffler from a given dataframe.

    Args:
      df_with_input_one_hot(pandas dataframe): Rows are known points.
      output_category_level (int, level name, or sequence of such, or None):
        Same as level used by pandas groupby method, except when None.

    Returns:
      a data_shuffler instance. The shuffler maps entries in the given dataframe
      to a one hot encoding of the labels at the provided granularity level.

    Raises:
      ValueError: Unknown value for batch_input_info.
    """
    input_df, input_one_hot = split_df(df_with_input_one_hot,
                                       self._num_input_categories)
    output_one_hot = self.categorize_one_hot(
        input_df, category_level=output_category_level)
    outputs = self._input_output_type(output_one_hot.values)
    if self._batch_input_info is not None:
      if self._batch_input_info == "multiplexed":
        inputs = self.encoding_to_multiplexed_encoding(input_df.values,
                                                       input_one_hot.values)
      elif self._batch_input_info == "one_hot":
        inputs = self._input_output_type(df_with_input_one_hot.values)
      else:
        raise ValueError("unknown value for batch_input_info.")
    else:
      inputs = self._input_output_type(input_df.values)
    return DataShuffler(inputs, outputs, self._random_state)

  def categorize_one_hot(self, df, category_level=None, drop_original=True):
    """Generate one-hot encoding from a given dataframe and selected categories.

    Args:
      df (pandas dataframe): input dataframe.
      category_level (int, level name, or sequence of such, or None): Same as
        level used by pandas groupby method, except when None. Used to indicate
        which indices to use for one-hot encoding.
      drop_original (bool): whether or not to drop the original table, leaving
        only the one-hot encoding vector.

    Returns:
      A tuple (df_one_hot, num_columns) where df_one_hot is the generated
        dataframe and num_columns is the number of columns of the original
        input dataframe df.
    """

    ## Ensure dataframe indices are strings
    index_names = df.index.names
    index = df.index.map(mapper=lambda x: tuple(str(v) for v in x))
    index.names = index_names
    df.index = index

    num_columns = len(df.columns)

    ## convert category_level from index to row values
    df_reset_index = df.reset_index(category_level)

    ## convert non-embedding values to one-hot encoding
    df_with_one_hot = pd.get_dummies(df_reset_index)

    ## restore indices
    df_with_one_hot = pd.DataFrame(data=df_with_one_hot.values, index=df.index)

    if drop_original:
      _, one_hot_df = split_df(df_with_one_hot, num_columns)
      return one_hot_df
    else:
      return df_with_one_hot

  def encoding_to_multiplexed_encoding(self, arr, encoding):
    """Generate a multiplexed encoding.

    For each entry, we have a row from arr and a row from encoding, specifying
    the original inputs the encoding value. In the multiplexed encoding
    scheme, the new dimension is num_bins * (num_dim + 1), where num_dim is the
    dimension of the original arr, and num_bins is the number of unique encoding
    values. Almost all values are set to zero, except for the values in bin i,
    where i is an index corresponding to the encoding value. Bin i is populated
    by the original entry from arr, followed by a 1.

    Args:
      arr (ndarray): 2 dimensional numpy array representing the original inputs.
        The rows are the individual entries, and the columns are the various
        coorindates.
      encoding (ndarray): 2 dimensional numpy array whose rows represent
        encodings.

    Returns:
      multiplexed_encoding (ndarray): 2 dimensional array as described above.

    Raises:
      ValueError: First dimension of arr and encoding must match.

    """
    if arr.shape[0] != encoding.shape[0]:
      raise ValueError("First dimension of arr and encoding must match.")
    num_points = arr.shape[0]
    num_dim = arr.shape[1]
    num_categories = len(self.encoding_to_num)
    ones = np.ones((num_points, 1))
    values_with_ones = np.hstack((arr, ones))
    ## TODO(tabakg): make this faster
    multiplexed_values = self._zeros_maker(
        (num_points, (num_dim + 1) * num_categories))
    for row_idx, (value, enc) in enumerate(zip(values_with_ones, encoding)):
      bin_num = self.encoding_to_num[tuple(enc)]
      multiplexed_values[row_idx, bin_num * (num_dim + 1):(bin_num + 1) *
                         (num_dim + 1)] = value
    return multiplexed_values


################################################################################
##### Code for Neural Network
################################################################################


def reverse_gradient(tensor):
  return -tensor + tf.stop_gradient(2 * tensor)


def make_tensor_dict(variable):
  """Returns a function that acts like a dictionary of tensors.

  Notice that the values provided to tf.case must be functions, which may have
  a particular scope. Since our input is a dictionary, we have to include its
  values in the scope of each of the functions provided to tf.case.

  Args:
    variable (dict): Input dictionary.

  Returns:
    f (function): A function mapping the keys from the given dictionary to its
    values in tensorflow.
  """

  ## TODO(tabakg): consider using tf.HashTable as a possible alternative.
  def variable_func(x):
    return lambda: variable[x]

  def dict_func(x):
    return tf.case(
        {tf.reduce_all(tf.equal(x, k)): variable_func(k) for k in variable})

  return dict_func


def make_wb(input_dim, output_dim, name):
  w = tf.Variable(
      np.eye(max(input_dim, output_dim))[:input_dim, :output_dim],
      name="w_" + name,
      dtype=tf.float32)
  b = tf.Variable(np.zeros(output_dim), name="b_" + name, dtype=tf.float32)
  return w, b


def make_discriminator_model(input_dim, activation, layer_width, num_layers):
  """Generates multi-layer model from the feature space to a scalar.

  Args:
    input_dim (int): Number of dimensions of the inputs.
    activation (function): activation layers to use.
    layer_width (int): Number of neurons per layer.
    num_layers (int): Number of layers not including the inputs.

  Returns:
    discriminator_model (function): Maps an input tensor to the output of the
       generated network.
  """
  w = {}
  b = {}
  w[0], b[0] = make_wb(input_dim, layer_width, name=CRITIC_LAYER_NAME + "0")
  for i in range(1, num_layers):
    w[i], b[i] = make_wb(
        layer_width, layer_width, name=CRITIC_LAYER_NAME + str(i))
  w[num_layers], b[num_layers] = make_wb(
      layer_width, 1, name=CRITIC_LAYER_NAME + str(num_layers))

  def discriminator_model(x0):
    x = {0: x0}
    for i in range(num_layers):
      x[i + 1] = activation(tf.matmul(x[i], w[i]) + b[i])
    return tf.matmul(x[num_layers], w[num_layers]) + b[num_layers]

  return discriminator_model


def wasserstein_distance(x_,
                         y_,
                         layer_width,
                         num_layers,
                         batch_n,
                         penalty_lambda=10.,
                         activation=tf.nn.softplus,
                         seed=None):
  """Estimator of the Wasserstein-1 distance between two distributions.

  This is based on the loss used in the following paper:

  Gulrajani, Ishaan, et al. "Improved training of wasserstein gans."
  arXiv preprint arXiv:1704.00028 (2017).

  One important difference between the following implementation and the paper
  above is that we only use the gradient constraint from above. This seems to
  work better for our problem in practice.

  Args:
    x_ (tf.Tensor): Empirical sample from first distribution. Its dimensions
      should be [batch_n, input_dim]
    y_ (tf.Tensor): Empirical sample from second distribution. Its dimensions
      should be [batch_n, input_dim].
    layer_width (int): Number of neurons to use for the discriminator model.
    num_layers (int): Number of layers to use for the discriminator model.
    batch_n (int): Number of elements per batch of x_ and y_.
    penalty_lambda (float): Penalty used to enforce the gradient condition.
      Specifically, the norm of discriminator_model should be no more than 1.
    activation (function): activation layers to use.
    seed (int): Used to randomly pick points where the gradient is evaluated.
      Using seed=None will seed differently each time the function is called.
      However, if using a global seed (tf.set_random_seed(seed)), this will
      reproduce the same results for each run.

  Returns:
    disc_loss (scalar tf.Tensor): The estimated Wasserstein-1 loss.
    gradient_penalty (scalalr tf.Tensor):: Value on the gradient penalty.
    discriminator_model (function): The function used to estimate the
      Wasserstein distance.
  """

  ## Get the number of elements and input size
  _, input_dim = x_.get_shape().as_list()

  ## make discriminator
  discriminator_model = make_discriminator_model(input_dim, activation,
                                                 layer_width, num_layers)

  ## random point for gradient penalty
  epsilon_rand = tf.random_uniform([batch_n, 1],
                                   minval=0,
                                   maxval=1,
                                   dtype=tf.float32,
                                   seed=seed)
  ## make this constant to test source of non-determinism.
  ## At least on desktop machine, this did not cause non-determinism.
  # epsilon_rand = tf.constant(0.5, shape=[batch_n, 1], dtype=tf.float32)
  x_hat = epsilon_rand * x_ + (1.0 - epsilon_rand) * y_

  ## gradient penalty
  (gradient,) = tf.gradients(discriminator_model(x_hat), [x_hat])
  gradient_penalty = penalty_lambda * tf.square(
      tf.maximum(0.0,
                 tf.norm(gradient, ord=2) - 1.0))

  ## calculate discriminator's loss
  disc_loss = (
      tf.reduce_mean(discriminator_model(y_) - discriminator_model(x_)) +
      gradient_penalty)

  return disc_loss, gradient_penalty, discriminator_model


def cov_tf(mean_x, x):
  """Compute the sample covariance matrix.

  Args:
    mean_x (tf.Tensor): Dimensions should be [1, input_dim]
    x (tf.Tensor): Dimensions should be [batch_n, input_dim]

  Returns:
    cov_xx (tf.Tensor): covariance of x.
  """
  mean_centered_x = x - mean_x
  n = tf.cast(tf.shape(x)[0], tf.float32)  ## number of points
  return tf.matmul(tf.transpose(mean_centered_x), mean_centered_x) / (n - 1)


def mat_sqrt_tf(mat):
  """Compute the square root of a non-negative definite matrix."""
  return gen_linalg_ops.matrix_square_root(mat)

  ### old version -- not sure if this would support gradients...
  # [e, v] = tf.self_adjoint_eig(mat)
  # return tf.matmul(tf.matmul(v, tf.diag(tf.sqrt(e))), tf.transpose(v))


def wasserstein_2_distance(x_, y_, mean_only=False):
  """Compute the wasserstein-2 distance squared between two distributions.

  This uses the closed form and assumes x_ and y_ are sampled from Gaussian
  distributions.

  Based on two_wasserstein_tf in
  research/biology/diagnose_a_well/analysis/distance.py

  Args:
    x_ (tf.Tensor): Empirical sample from first distribution. Its dimensions
      should be [batch_n, input_dim]
    y_ (tf.Tensor): Empirical sample from second distribution. Its dimensions
      should be [batch_n, input_dim].
    mean_only (bool): Restrict to use only the mean part.

  Returns:
    wasserstein-2 distance between x_ and y_.
  """
  mean_x_ = tf.reduce_mean(x_, axis=0, keep_dims=True)
  mean_y_ = tf.reduce_mean(y_, axis=0, keep_dims=True)

  if mean_only:
    return transform.sum_of_square(mean_x_ - mean_y_)

  ## If not using mean_only, compute the full W-2 distance metric:

  ## TESTING: adding identity to avoid error
  cov_x_ = cov_tf(mean_x_, x_) + FLAGS.cov_fix * tf.eye(FLAGS.feature_dim)
  cov_y_ = cov_tf(mean_y_, y_) + FLAGS.cov_fix * tf.eye(FLAGS.feature_dim)
  sqrt_cov_x_ = mat_sqrt_tf(cov_x_)

  prod = tf.matmul(tf.matmul(sqrt_cov_x_, cov_y_), sqrt_cov_x_)
  return transform.sum_of_square(mean_x_ -
                                 mean_y_) + tf.trace(cov_x_ + cov_y_ -
                                                     2.0 * mat_sqrt_tf(prod))


class Network(object):
  """Learns features that forget desired information.

  Since we may want different losses for different parts of the data, we use
  an instance of DatasetHolder containing instances of DataShuffler. They keys
  for the dictionary attributes are the same as the ones used for the
  DataShuffler instance, unless otherwise noted.

  There are several implementations, using ideas from the following paper to
  train a discriminator (critic), and worsen its performance during training:

  Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by
  backpropagation." International Conference on Machine Learning. 2015.

  This class has only the attributes common to all networks we will use for
  experiments. Inheriting networks will have additional components completing
  the network and implementaing train, predict, and evaluate.

  Atributes:
    holder (DatasetHolder): Holds the various DataShuffler instances to use.
    loss_specifiers (dict): Maps keys to loss types.
    losses (dict): Maps each loss type to a dictionary, which maps keys to
      the relevant tensorflow objects representing losses.
    input_dim_with_encoder (int): Total input size, including encoding.
    input_dim (int): Input size, not including encoding.
    encoding_dim (int): Input size of the encoding.
  """

  def __init__(self, holder, feature_dim, batch_n):
    """Inits Network with a given instance of DatasetHolder and parameters.

    Args:
      holder (instance of DatasetHolder): Provides data to train model
      feature_dim (int): Dimension of feature space. Currently supports the same
        dimension as the input space only. The reason is that for now we wish to
        initialize the transformation to the identity. However, this could be
        extended in the future.
      batch_n (int): Number of points to use from each DataShuffler during
        training. Currently this is the same for each DataShuffler.

    Raises:
      ValueError: Currently only supporting feature_dim == input_dim.
    """

    self.holder = holder
    self.losses = {loss: {} for loss in POSSIBLE_LOSSES}

    ## initialize useful dimensions and constants
    _, self._input_dim_with_encoder = holder.df_with_input_one_hot.shape
    self._input_dim = holder.input_dim
    self._encoding_dim = self._input_dim_with_encoder - self._input_dim
    self._feature_dim = feature_dim
    if self._input_dim != self._feature_dim:
      raise ValueError("Currently only supporting feature_dim == input_dim. "
                       "But we have feature_dim = %s and input_dim = %s" %
                       (feature_dim, holder.input_dim))
    self._batch_n = batch_n


def ensure_is_lst(z):
  """If a variable is not a list, return a list with only the variable.

  Args:
    z (object): input variable

  Returns:
    z if it is a list, otherwise [z]
  """
  if isinstance(z, list):
    return z
  else:
    return [z]


def get_filtered_key(input_key, indices):
  """Filter input keys to indices.

  Args:
    input_key (tuple): Contains input key information.
    indices: Which indices to select from the input key.

  Returns:
    filtered_key (tuple): The components of the input key designated by indices.
  """
  return tuple([input_key[i] for i in indices])


def sum_of_square(a):
  """Sum of squared elements of Tensor a."""
  return tf.reduce_sum(tf.square(a))


class WassersteinNetwork(Network):
  """Network with Wasserstein loss to forget batch effects.

  This network imposes only Wasserstein losses given a DatasetHolder
  containing instances of Datashuffler.

  The DataShuffler instances in the DatasetHolder are indexed by keys
  corresponding to input index levels for each DatasetHolder, as well as some
  specified output values. Here we only use the inputs (first component of each
  key, which we will call the input key).

  For example, this element might have form: (drug1, batch3).

  The arguments target_levels and nuisance_levels are used to specify the
  indices of the input key that will correspond to variables across which
  forgetting occurs, and variables that should be forgotten, respectively.
  That is, for each unique combination of target_levels, we consider the
  Wasserstein loss of each possible pairwise combination of nuisance_levels.

  For example, in the case when the input key has two elements (drug and batch
  in that order), specifying target_levels=[0] and nuisance_levels=[1] would
  construct the pairwise Wasserstein loss between all possible batches for each
  fixed individual drug.

  For this network, impose the condition that all DataShuffler instances have
  keys with matching input dimensionality. This ensures they all contain the
  necessary information to match distributions across some label categories
  but not others.

  The transformation from the inputs to the features is currently stored using
  the dictionaries w and b. This should be modified if a nonlinear function is
  used instead.

  Attributes:
    wass_loss_target (dict): Maps target keys to the sum of pairwise Wasserstine
      losses for the possible nuisance variables with the same target key.
      Note: The losses are the negative Wasserstein distances.
    grad_pen_target (dict): Maps target keys to the sum of pairwise gradient
      penalties for the possible nuisance variables with the same target key.
    w (dict): Mapping from keys to w tensorflow tensors going from inputs to
      features.
    b (dict): Mapping from keys to b tensorflow tensors going from inputs to
      features.
    ignore_disc (bool): If this is true, do not train a discriminator. This
      should be set to True when using a distance that does not need to be
      learned, e.g. the Wasserstein-2 distance.
  """

  def __init__(self,
               holder,
               feature_dim,
               batch_n,
               target_levels,
               nuisance_levels,
               layer_width=2,
               num_layers=2,
               lambda_mean=0.,
               lambda_cov=0.,
               power=1.,
               ignore_disc=False,
               mean_only=False):
    """Inits WassersteinNetwork.

    Args:
      holder (instance of DatasetHolder): Provides data to train model
      feature_dim (int): Dimension of feature space. Currently supports the same
        dimension as the input space only.
      batch_n (int): Number of points to use from each DataShuffler during
        training. Currently this is the same for each DataShuffler.
      target_levels (int or list of ints): Index or indices indicating which
        components of the input keys should have the property that for points
        sampled with the same values for these components the distributions
        should be indistinguishable.
      nuisance_levels (int or list of ints): Index or indices indicating which
        components of the input keys
      layer_width (int): number of neurons to use per layer for each function
        estimating a Wasserstein distance.
      num_layers (int): number of layers to use for estimating the Wasserstein
        loss (not including input).
      lambda_mean (float): penalty term for the mean term of the transformation.
      lambda_cov (float): penalty term for the cov term of the transformation.
      power (float): power of each pair-wise wasserstein distance to use.
      ignore_disc (bool): If this is true, do not train a discriminator. This
        should be set to True when using a distance that does not need to be
        learned, e.g. the Wasserstein-2 distance.
      mean_only (bool): Using the Wasserstein-2 distance, but only the mean
        component (i.e. no covariance dependence). This is for experimental
        purposes.

    Raises:
      ValueError: Keys must have the same input dimensionality.
    """

    super(WassersteinNetwork, self).__init__(holder, feature_dim, batch_n)
    self._layer_width = layer_width
    self._num_layers = num_layers
    self._target_levels = ensure_is_lst(target_levels)
    self._nuisance_levels = ensure_is_lst(nuisance_levels)
    self._lambda_mean = lambda_mean
    self._lambda_cov = lambda_cov
    shufflers = holder.data_shufflers

    ## The first component of each key indicates which categories have been
    ## used for sampling (i.e. the input keys).
    input_key_lengths = [len(key[INPUT_KEY_INDEX]) for key in shufflers]
    if not all(length == input_key_lengths[0] for length in input_key_lengths):
      raise ValueError("Keys must have the same input dimensionality.")

    ## Generate all possible keys using only the components of the target
    ## or nuisance variables.
    self._unique_targets = sorted(
        list(
            set([
                get_filtered_key(s[INPUT_KEY_INDEX], self._target_levels)
                for s in shufflers
            ])))
    self._unique_nuisances = sorted(
        list(
            set([
                get_filtered_key(s[INPUT_KEY_INDEX], self._nuisance_levels)
                for s in shufflers
            ])))

    ## Map from each possible target key to all input keys that generated it.
    self._keys_for_targets = collections.defaultdict(list)
    for target in self._unique_targets:
      for s in shufflers:
        if target == get_filtered_key(s[INPUT_KEY_INDEX], self._target_levels):
          self._keys_for_targets[target].append(s)

    ## Generate input placeholders.
    self._x_inputs, self._x_vals = self.get_input_vals()

    ## Make features.
    self.w, self.b, self._features = self.add_input_to_feature_layer()

    ## Generate pairwise Wasserstein losses.
    if ignore_disc:
      self.wass_loss_target = self.pairwise_wasserstein_2(mean_only=mean_only)
      self.grad_pen_target = None
      self.ignore_disc = True
    else:
      self.wass_loss_target, self.grad_pen_target = self.pairwise_wasserstein(
          power=power)
      self.ignore_disc = False

  def get_input_vals(self):
    """Obtain the input values from a given dataframe.

    There might be additional encodings which we wish to strip away.
    The additional encodings have dimension _encoding_dim and appear first.
    The input values have dimension _input_dim and appear second.

    Returns:
      x_vals (dict): Maps input keys to placeholders for the input values.
    """
    x_vals = {}
    x_inputs = {}
    for key in self.holder.data_shufflers:
      x_inputs[key] = tf.placeholder(
          tf.float32, [None, self._input_dim_with_encoder],
          name="x_" + INPUT_NAME)
      x_vals[key], _ = tf.split(x_inputs[key],
                                [self._input_dim, self._encoding_dim], 1)
    return x_inputs, x_vals

  def add_input_to_feature_layer(self):
    """Add a layer from the inputs to features.

    The transformation for inputs with the same nuisance variables is fixed.

    Returns:
      features (dict): Maps each data_shuffler key to the feature tensor.
    """
    w = {}
    b = {}
    features = {}
    for nuisance_key in self._unique_nuisances:
      w[nuisance_key], b[nuisance_key] = make_wb(
          self._input_dim, self._feature_dim, name=INPUT_NAME)
    for key in self.holder.data_shufflers:
      nuisance_key = get_filtered_key(key[INPUT_KEY_INDEX],
                                      self._nuisance_levels)
      features[key] = tf.matmul(self._x_vals[key],
                                w[nuisance_key] + b[nuisance_key])
    return w, b, features

  def pairwise_wasserstein(self, power=1.):
    """Generate pairwise Wasserstein losses.

    The pairs are the various nuisance variables (e.g. batch). This is done
    separately for each target variable (e.g. compound).

    Args:
      power (float): power of each pair-wise wasserstein distance to use.

    Returns:
      wass_loss_target (dict): Maps from target keys to sum of pairwise losses.
      grad_pen_target (dict): Maps from target keys to sum of pairwise gradient
        penalties.

    """
    wass_loss_target = {}
    grad_pen_target = {}

    ## Gradient reversal if using a discriminator.
    grad_rev_features = {
        key: reverse_gradient(self._features[key]) for key in self._features
    }

    for target in self._unique_targets:
      num_per_target = len(self._keys_for_targets[target])
      # When the target appears in multiple domains.
      if num_per_target > 1:
        normalization = num_per_target * (num_per_target - 1) / 2.
        wass_loss_target[target] = 0
        grad_pen_target[target] = 0
        ## Iterate through all pairs of nuisance for a given target
        for i in range(num_per_target):
          for j in range(i + 1, num_per_target):
            key_i = tuple(self._keys_for_targets[target][i])
            key_j = tuple(self._keys_for_targets[target][j])

            ## Generate W1 distance and gradient penalty.
            wass_dists, grad_pens, _ = wasserstein_distance(
                grad_rev_features[key_i],
                grad_rev_features[key_j],
                self._layer_width,
                self._num_layers,
                self._batch_n,
                seed=None)
            wass_loss_target[target] += tf.math.pow(wass_dists,
                                                    power) / normalization
            grad_pen_target[target] += grad_pens / normalization

    return wass_loss_target, grad_pen_target

  def pairwise_wasserstein_2(self, mean_only=False):
    """Generate pairwise Wasserstein-2 squared losses.

    This uses the closed-form solution assuming Guassian distributions.

    The pairs are the various nuisance variables (e.g. batch). This is done
    separately for each target variable (e.g. compound).

    Args:
      mean_only (bool): Restrict to use only the mean part.

    Returns:
      wass_loss_target (dict): Maps from target keys to sum of pairwise losses.
      grad_pen_target (dict): Maps from target keys to sum of pairwise gradient
        penalties.

    """
    wass_loss_target = {}

    for target in self._unique_targets:
      num_per_target = len(self._keys_for_targets[target])
      # When the target appears in multiple domains.
      if num_per_target > 1:
        wass_loss_target[target] = 0
        normalization = num_per_target * (num_per_target - 1) / 2.
        ## Iterate through all pairs of nuisance for a given target
        for i in range(num_per_target):
          for j in range(i + 1, num_per_target):
            key_i = tuple(self._keys_for_targets[target][i])
            key_j = tuple(self._keys_for_targets[target][j])

            ## Generate W2 distance and gradient penalty.
            wass_2_dists = wasserstein_2_distance(
                self._features[key_i],
                self._features[key_j],
                mean_only=mean_only) / normalization
            wass_loss_target[target] += wass_2_dists

    return wass_loss_target

  def penalty_term(self):
    """Penalty term on the affine transformation.

    This can be used to ensure that the affine transformation remains close to
    the identity transformation.

    Returns:
      loss_value (tensorflow float): Value of the penalty term to add.
    """
    loss_value = 0.
    identity = tf.eye(self._input_dim, dtype=tf.float32)
    for nuisance_key in self._unique_nuisances:
      w = self.w[nuisance_key]
      b = self.b[nuisance_key]
      loss_value += (
          self._lambda_mean * sum_of_square(b) +
          self._lambda_cov * sum_of_square(w - identity) / self._input_dim)
    return loss_value / len(self.w)

  def train(self,
            save_dir,
            num_steps_pretrain,
            num_steps,
            disc_steps_per_training_step,
            learning_rate,
            tf_optimizer=tf.train.RMSPropOptimizer,
            save_checkpoints=True,
            save_to_pickle=True):
    """Trains the network, saving checkpoints along the way.

    Args:
      save_dir (str): Directory to save pickle file
      num_steps_pretrain (int): Total steps for pre-training
      num_steps (int): Total step for main training loop.
      disc_steps_per_training_step (int): Number of training steps for the
        discriminator per training step for the features.
      learning_rate (float): Learning rate for the algorithm.
      tf_optimizer (tf Optimizer): Which optimizer to use
      save_checkpoints (bool): Whether or not to use tensorflow checkpoints.
      save_to_pickle (bool): Whether or not to save to a pickle file. This may
        be convenient for some purposes.
    """

    ## Generate paths and names for saving results.
    input_df_name = os.path.basename(FLAGS.input_df)
    params = (
        ("input_df", input_df_name),
        ("network_type", FLAGS.network_type),
        ("num_steps_pretrain", num_steps_pretrain),
        ("num_steps", num_steps),
        ("batch_n", self._batch_n),
        ("learning_rate", learning_rate),
        ("feature_dim", self._feature_dim),
        ("disc_steps_per_training_step", disc_steps_per_training_step),
        ("target_levels", tuple(FLAGS.target_levels)),
        ("nuisance_levels", tuple(FLAGS.nuisance_levels)),
        ("layer_width", self._layer_width),
        ("num_layers", self._num_layers),
        ("lambda_mean", self._lambda_mean),
        ("lambda_cov", self._lambda_cov),
        ("cov_fix", FLAGS.cov_fix),
    )

    folder_name = str(params)
    folder_path = os.path.join(save_dir, folder_name)
    ## Not writing to registry...
    # if FLAGS.reg_dir is None:
    #   reg_path = os.path.join(save_dir, REGISTRY_NAME + ".pkl")
    # else:
    #   reg_path = os.path.join(FLAGS.reg_dir, REGISTRY_NAME + ".pkl")
    pickle_path = os.path.join(folder_path, "data.pkl")
    checkpoint_path = os.path.join(folder_path, "checkpoints")

    for p in [save_dir, folder_path, checkpoint_path]:
      if not gfile.Exists(p):
        gfile.MkDir(p)

    ## Tensorflow items used for training
    global_step = tf.Variable(0, trainable=False, name="global_step")
    increment_global_step_op = tf.assign(global_step, global_step + 1)
    sorted_wass_values = [
        self.wass_loss_target[k] for k in self.wass_loss_target
    ]
    wasserstein_loss = tf.reduce_mean(sorted_wass_values)
    if self.ignore_disc:
      loss = wasserstein_loss
    else:
      loss = wasserstein_loss + self.penalty_term()
      grad_pen = tf.reduce_mean(list(self.grad_pen_target.values()))
    input_vars = [
        var for var in tf.trainable_variables() if INPUT_NAME in var.name
    ]
    critic_vars = [
        var for var in tf.trainable_variables() if CRITIC_LAYER_NAME in var.name
    ]
    optimizer = tf_optimizer(learning_rate)
    input_trainer = optimizer.minimize(
        loss, global_step=global_step, var_list=input_vars)
    if not self.ignore_disc:
      critic_trainer = optimizer.minimize(
          loss, global_step=global_step, var_list=critic_vars)

    tf.summary.scalar("loss", loss)
    if not self.ignore_disc:
      tf.summary.scalar("grad_pen", grad_pen)
    ## TODO(tabakg): Figure out why summary_op is not working.
    ## There currently seems to be an issue using summay_op, so it's set to None
    ## TODO(tabakg): There is also an issue with Supervisor when trying to load
    ## existing checkpoint files.
    _ = tf.summary.merge_all()  ## Should be summary_op
    sv = tf.train.Supervisor(logdir=checkpoint_path, summary_op=None)

    ## Used for saving history to pickle file
    loss_hist = []
    grad_pen_hist = []  ## still initialize when not using critic

    ## random history, used to monitor random seeds
    random_nums = []

    def do_loss_without_step():
      """Get losses and update loss_hist and gran_pen_hist, no training step."""

      feed_dict = {}
      for key, shuffler in self.holder.data_shufflers.items():
        input_mini, _ = shuffler.next_batch(self._batch_n)
        feed_dict[self._x_inputs[key]] = input_mini
      random_nums.append(self.holder.get_random_state().uniform(
          0, 1))  ## Testing random seed
      if self.ignore_disc:
        loss_val = sess.run([loss], feed_dict=feed_dict)[0]
        grad_pen_val = None
      else:
        loss_val, grad_pen_val = sess.run([loss, grad_pen], feed_dict=feed_dict)

      loss_hist.append(loss_val)
      grad_pen_hist.append(grad_pen_val)

    def do_train_step(trainer, increment_global_step_op, train=True):
      """A single training step.

      Side effects: Updates loss_hist, grad_pen_hist

      Args:
        trainer (Tf Operation): Specifies how to update variables in each step.
        increment_global_step_op (tf op): used to increment step.
        train (boolean): Whether or not to train. If train is false, only step
          and record loss values without actually training.

      Returns:
        step (int): Current timestep.
      """
      feed_dict = {}
      for key, shuffler in self.holder.data_shufflers.items():
        input_mini, _ = shuffler.next_batch(self._batch_n)
        feed_dict[self._x_inputs[key]] = input_mini
      if train:
        if self.ignore_disc:
          _, loss_val = sess.run([trainer, loss], feed_dict=feed_dict)
          grad_pen_val = None
        else:
          _, loss_val, grad_pen_val = sess.run([trainer, loss, grad_pen],
                                               feed_dict=feed_dict)
        step = tf.train.global_step(sess, global_step)  ## get updated step.
      else:
        if self.ignore_disc:
          loss_val = sess.run([loss], feed_dict=feed_dict)[0]
          grad_pen_val = None
        else:
          loss_val, grad_pen_val = sess.run([loss, grad_pen],
                                            feed_dict=feed_dict)
        ## if trainer is not ran, increment global step anyway.
        step = sess.run(increment_global_step_op)

      loss_hist.append(loss_val)
      grad_pen_hist.append(grad_pen_val)

      if (step % FLAGS.checkpoint_interval == 0 and
          step >= FLAGS.num_steps_pretrain):
        if save_checkpoints:
          sv.saver.save(sess, checkpoint_path, global_step)
        if save_to_pickle:
          w_val, b_val = sess.run([self.w, self.b], feed_dict=feed_dict)
          contents = {
              step: {
                  "loss_hist": loss_hist,
                  "grad_pen_hist": grad_pen_hist,
                  "w_val": w_val,
                  "b_val": b_val,
                  "random_nums": random_nums,
              },
              "params": dict(params)
          }
          add_contents_to_pickle(pickle_path, contents)
          # add_name_to_registry(reg_path, pickle_path)

      return step

    ## Tested as a potential cause for indeterminism.
    ## This did not cause indeterminism on the desktop.
    # config = tf.ConfigProto()
    # config.inter_op_parallelism_threads = 1
    # config.intra_op_parallelism_threads = 1
    # with sv.managed_session(config=config) as sess:
    with sv.managed_session() as sess:
      step = tf.train.global_step(sess, global_step)

      ## Get initial losses without stepping
      do_loss_without_step()

      ## Pre training to adjust Wasserstein function
      while step < num_steps_pretrain:
        if self.ignore_disc:
          break
        step = do_train_step(critic_trainer, increment_global_step_op)

      ## Main training part
      main_step = step - num_steps_pretrain
      while main_step < num_steps * (disc_steps_per_training_step + 1):
        ## Adjust critic disc_steps_per_training_step times
        if disc_steps_per_training_step != 0:
          while (main_step + 1) % disc_steps_per_training_step > 0:
            if self.ignore_disc:
              break
            step = do_train_step(critic_trainer, increment_global_step_op)
            main_step = step - num_steps_pretrain
        ## Train features if estimated distance is positive
        if self.ignore_disc:
          pos_dist = True  ## if ignoring discriminator, this is not an issue
        else:
          pos_dist = (-loss_hist[-1] > 0)
        step = do_train_step(
            input_trainer, increment_global_step_op, train=pos_dist)
        main_step = step - num_steps_pretrain
      sv.stop()


################################################################################
##### Functions for Saving to Pickle.
################################################################################


def read_pickle_helper(file_path, default=set):
  """Helper function to read data from a pickle file.

  If the file exists, load it. Otherwise, initialize the default type.

  Args:
    file_path (str): Path to pickle file.
    default (iterable): Python object stored in the pickle file.

  Returns:
    contents (iterable): The loaded contents or an empty default type.
  """
  if gfile.Exists(file_path):
    with gfile.GFile(file_path, mode="r") as f:
      contents = pickle.loads(f.read())
  else:
    contents = default()
  return contents


def write_pickle_helper(file_path, contents):
  """Helper function to write contents to a pickle file.

  Args:
    file_path (str): Path to pickle file.
    contents (iterable): Contents to save to pickle file_path.
  """
  with gfile.GFile(file_path, mode="w") as f:
    f.write(pickle.dumps(contents))


def add_name_to_registry(reg_path, pickle_path):
  """Adds the file_path to the set stored in the pickle file reg_path.

  The registry file contains a dictionary whose keys are possible datasets.
  The values for each dataset key is another dictionary, mapping from a
  trasformation name to a list of file paths. Each of these paths corresponds
  to a saved transformation file that was generated using different parameters.

  Args:
    reg_path (str): Registry path.
    pickle_path (str): Path to registered pickle file.
  """
  if FLAGS.network_type == WASSERSTEIN_NETWORK:
    transform_name = "wasserstein_transform"
  elif FLAGS.network_type == WASSERSTEIN_2_NETWORK:
    transform_name = "wasserstein_2_transform"
  elif FLAGS.network_type == WASSERSTEIN_SQRD_NETWORK:
    transform_name = "wasserstein_squared_transform"
  elif FLAGS.network_type == WASSERSTEIN_CUBED_NETWORK:
    transform_name = "wasserstein_cubed_transform"
  elif FLAGS.network_type == MEAN_ONLY_NETWORK:
    transform_name = "mean_only_transform"
  else:
    raise ValueError("Unknown network type, please add...")

  reg = read_pickle_helper(reg_path, dict)
  if FLAGS.input_df not in reg:
    reg[FLAGS.input_df] = {}
  if transform_name not in reg[FLAGS.input_df]:
    reg[FLAGS.input_df][transform_name] = []
  if pickle_path not in reg[FLAGS.input_df][transform_name]:
    reg[FLAGS.input_df][transform_name].append(pickle_path)
  write_pickle_helper(reg_path, reg)


def add_contents_to_pickle(file_path, contents):
  """Adds contents to a pickle file.

  Args:
    file_path (str): Full path to the pickle file.
    contents (dict): Maps a timestep to parameters at that time.
  """
  old_contents = read_pickle_helper(file_path, dict)
  contents.update(old_contents)
  write_pickle_helper(file_path, contents)


def main(argv):
  del argv  # Unused.

  if FLAGS.network_type not in POSSIBLE_NETWORKS:
    raise ValueError("Unknown network type.")
  tf.reset_default_graph()
  with tf.Graph().as_default():
    tf.set_random_seed(FLAGS.tf_seed)

    ## Load embedding dataframe
    df = io_utils.read_dataframe_from_hdf5(FLAGS.input_df)

    ## TODO(tabakg): Add training routine for this network as well.
    ## This network did not seem to work as well, so this is secondary priority.

    using_w = WASSERSTEIN_NETWORK in FLAGS.network_type
    using_w_sqrd = WASSERSTEIN_SQRD_NETWORK in FLAGS.network_type
    using_w_cubed = WASSERSTEIN_CUBED_NETWORK in FLAGS.network_type
    using_w_1 = using_w or using_w_sqrd or using_w_cubed
    using_w_2 = WASSERSTEIN_2_NETWORK in FLAGS.network_type
    using_mean_only = MEAN_ONLY_NETWORK in FLAGS.network_type

    ## Script for initializing WassersteinNetwork
    if using_w_1 or using_w_2 or using_mean_only:
      if using_w:
        logging.info("using WASSERSTEIN_NETWORK.")
        power = 1.
      elif using_w_sqrd:
        logging.info("using WASSERSTEIN_SQRD_NETWORK.")
        power = 2.
      elif using_w_cubed:
        logging.info("using WASSERSTEIN_CUBED_NETWORK.")
        power = 3.

      ## TODO(tabakg): Possible bug when input_category_level=None.
      holder = DatasetHolder(
          df,
          input_category_level=FLAGS.nuisance_levels,
          batch_input_info="one_hot",
          random_state=np.random.RandomState(seed=FLAGS.seed))

      ## Old
      holder.add_shufflers(FLAGS.nuisance_levels + FLAGS.target_levels, None)

      # holder.add_shufflers(FLAGS.nuisance_levels, None)

      nuisance_levels = list(range(len(FLAGS.nuisance_levels)))
      target_levels = list(
          range(
              len(FLAGS.nuisance_levels),
              len(FLAGS.nuisance_levels) + len(FLAGS.target_levels)))

      if using_w_1:
        network = WassersteinNetwork(
            holder,
            FLAGS.feature_dim,
            FLAGS.batch_n,
            target_levels,
            nuisance_levels,
            layer_width=FLAGS.layer_width,
            num_layers=FLAGS.num_layers,
            lambda_mean=FLAGS.lambda_mean,
            lambda_cov=FLAGS.lambda_cov,
            power=power)

      else:  # using_w_2 or using_mean_only

        if using_mean_only:
          logging.info("using MEAN_ONLY_NETWORK.")
          mean_only = True
        else:
          logging.info("using WASSERSTEIN_2_NETWORK.")
          mean_only = False

        network = WassersteinNetwork(
            holder,
            FLAGS.feature_dim,
            FLAGS.batch_n,
            target_levels,
            nuisance_levels,
            layer_width=FLAGS.layer_width,
            num_layers=FLAGS.num_layers,
            lambda_mean=FLAGS.lambda_mean,
            lambda_cov=FLAGS.lambda_cov,
            ignore_disc=True,
            mean_only=mean_only)

      network.train(FLAGS.save_dir, FLAGS.num_steps_pretrain, FLAGS.num_steps,
                    FLAGS.disc_steps_per_training_step, FLAGS.learning_rate)


if __name__ == "__main__":
  app.run(main)

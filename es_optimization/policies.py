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

# Lint as: python3
r"""Library for creating different architectures for policies.

Each policy \Pi: S -> A is a mapping from the set of states to the set of
actions. Each policy provides a method that takes as an input state s and
outputs action a recommended by the policy.
"""

import abc
import copy
import math
import numpy as np
import scipy


class Policy(abc.ABC):
  r"""Abstract class for different policies \Pi: S -> A.

  Class is responsible for creating different policies and provides an interface
  for computing actions recommended by policies in different input states.
  In particular, this class provides an interface that accepts compressed
  vectorized form of the policy and decompresses it.

  Standard procedure for improving the parameters of the policy with an
  interface given by the class:

  policy = policies.ParticularClassThatInheritsFromBaseClass(...)
  vectorized_network = policy.get_initial()
  while(...):
    new_vectorized_network = SomeTransformationOf(vectorized_network)
    policy.update(new_vectorized_network)

  and SomeTransformationOf is a single step of some optimization procedure such
  as gradient descent that sees the policy in the vectorized form.
  """

  @abc.abstractmethod
  def update(self, vectorized_parameters):
    """Updates the policy using new parameters from <vectorized_parameters>.

    Updates the parameters of the policy using new parameters encoded by
    <vectorized_parameters>. The size of the vector <vectorized_parameters>
    should be the number of all biases and weights of the neural network.
    We use the convention where parameters encoding matrices of connections of
    the neural network come in <vectorized_parameters> before parameters
    encoding biases and furthermore the order in <vectorized_parameters> of
    parameters encoding weights for different matrices/biases-vectors is
    inherited from the order of these matrices/biases-vectors in the
    decompressed neural network. Details regarding compression depend on
    different neural network architectures used (such as: structured and
    unstructured) and are given in the implementations of that abstract method
    in specific classes that inherit from Policy.

    Args:
      vectorized_parameters: parameters of the neural network in the vectorized
        form.

    Returns:
    """
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def get_action(self, state):
    """Returns the action proposed by a policy in a given state.

    Returns an action proposed by the policy in <state>.

    Args:
      state: input state

    Returns:
      Action proposed by the policy represented by an object of the class in a
      given state.
    """
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def get_initial(self):
    """Returns the default parameters of the policy in the vectorized form.

    Initial parameters of the policy are output in the vectorized form.

    Args:

    Returns:
      Numpy array encoding in the vectorized form initial parameters of the
      policy.
    """
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def get_total_num_parameters(self):
    """Outputs total number of parameters of the policy.

    Args:

    Returns:
      Total number of parameters used by the policy.
    """
    raise NotImplementedError('Abstract method')

  def reset(self):
    """Resets any relevant parameters in the policy."""
    pass


class UnstructuredNeuralNetworkPolicy(Policy):
  """Derives from Policy and encodes a policy using unstructured neural network.

  This class encodes agent's policy as an unstructured neural network fed with
  the state of an agent and outputting recommended action. "Unstructured" means
  that the matrices of the neural network are not constrained to live in the
  lower-dimensional space, have low-displacement rank, etc. Thus the policy
  is determined by the full set of all the weights and biases.
  """

  def __init__(
      self,
      state_dimensionality,
      action_dimensionality,
      hidden_layers,
      nonlinearities,
      low=None,
      high=None,
  ):
    """Sets up parameters of the unstructured neural network.

    Args:
      state_dimensionality: dimensionality of the first layer
      action_dimensionality: dimensionality of the last layer
      hidden_layers: list of sizes of hidden layers
      nonlinearities: list of nonlinear mapping applied pointwise in hidden
                      layers; each nonlinearity is a mapping f: R^{n} ->R^{n},
                        where n - dimensionality of the input vector as well as
                        its nonlinear transformation
      low: A list of minimum bounds for the action.
      high: A list of maximum bounds for the action array.
    """
    matrices = []

    matrices.append(
        np.zeros(state_dimensionality * hidden_layers[0]).reshape(
            hidden_layers[0], state_dimensionality))
    for i in range(0, len(hidden_layers) - 1):
      matrices.append(
          np.zeros(hidden_layers[i] * hidden_layers[i + 1]).reshape(
              hidden_layers[i + 1], hidden_layers[i]))
    matrices.append(
        np.zeros(hidden_layers[len(hidden_layers) - 1] *
                 action_dimensionality).reshape(
                     action_dimensionality,
                     hidden_layers[len(hidden_layers) - 1]))
    biases = []
    for i in range(len(hidden_layers)):
      biases.append(np.zeros(hidden_layers[i]).reshape(hidden_layers[i], 1))
    self.matrices = matrices
    self.biases = biases
    self.nonlinearities = nonlinearities
    self.state_dimensionality = state_dimensionality
    self.action_dimensionality = action_dimensionality
    self.hidden_layers = hidden_layers
    self.low = low
    self.high = high
    super().__init__()

  def update(self, vectorized_parameters):
    new_matrices = []
    current_index = 0

    new_matrices.append(vectorized_parameters[current_index:current_index +
                                              self.state_dimensionality *
                                              self.hidden_layers[0]].reshape(
                                                  self.hidden_layers[0],
                                                  self.state_dimensionality))
    current_index += self.state_dimensionality * self.hidden_layers[0]
    for i in range(0, len(self.hidden_layers) - 1):
      new_matrices.append(
          vectorized_parameters[current_index:current_index +
                                self.hidden_layers[i] *
                                self.hidden_layers[i + 1]].reshape(
                                    self.hidden_layers[i + 1],
                                    self.hidden_layers[i]))
      current_index += self.hidden_layers[i] * self.hidden_layers[i + 1]

    new_matrices.append(
        vectorized_parameters[current_index:current_index +
                              self.hidden_layers[len(self.hidden_layers) - 1] *
                              self.action_dimensionality].reshape(
                                  self.action_dimensionality,
                                  self.hidden_layers[len(self.hidden_layers) -
                                                     1]))
    current_index += self.hidden_layers[len(self.hidden_layers) -
                                        1] * self.action_dimensionality
    new_biases = []
    for i in range(len(self.hidden_layers)):
      new_biases.append(vectorized_parameters[current_index:current_index +
                                              self.hidden_layers[i]].reshape(
                                                  self.hidden_layers[i], 1))
      current_index += self.hidden_layers[i]
    self.matrices = new_matrices
    self.biases = new_biases

  def get_action(self, state):
    state = np.reshape(state, (len(state), 1))
    for i in range(len(self.matrices) - 1):
      state = np.matmul(self.matrices[i], state)
      state = np.add(state, self.biases[i])
      state = (self.nonlinearities[i])(state)
    action = np.matmul(self.matrices[len(self.matrices) - 1], state)

    if self.low is not None and self.high is not None:
      action = np.tanh(action)
      for i in range(len(action)):
        action[i][0] = (
            action[i][0] * (self.high[i] - self.low[i]) / 2.0 +
            (self.low[i] + self.high[i]) / 2.0)

    return action

  def get_initial(self):
    # The initial policy is given by weights and biases taken independently at
    # random from the Gaussian distribution.
    np.random.seed(100)
    vectorized_list = []
    for i in range(len(self.matrices)):
      next_matrix = 1.0 / math.sqrt(float(len(
          self.matrices[i]))) * np.random.randn(
              len(self.matrices[i]) * len(self.matrices[i][0]))
      vectorized_list.append(next_matrix)
    for i in range(len(self.biases)):
      next_biases_vector = np.random.randn(len(self.biases[i]))
      vectorized_list.append(next_biases_vector)
    vectorized_network = np.concatenate(vectorized_list)
    return vectorized_network

  def get_total_num_parameters(self):
    total = 0
    for i in range(len(self.matrices)):
      total += len(self.matrices[i]) * len(self.matrices[i][0])
    for i in range(len(self.biases)):
      total += len(self.biases[i])
    return total


class TwoLayerTanhToeplitzNNP(Policy):
  """Derives from Policy and encodes a policy using Toeplitz neural network.

  This class encodes agent's policy as a structured neural network fed with
  the state of an agent and outputting recommended action. The neural network
  has two hidden layers, each followed by tanh nonlinearity. All the matrices
  of connections are constrained to be Toeplitz matrices. This policy also
  supports state normalization. If the state_normalization flag is on,
  the policy keeps track of the necessary state normalization parameters.
  First it has a field self.global_num_steps that allows to store the number of
  global steps taken so far and used in the computation of the state mean
  and state covariances. It will also have a fields self.state_mean and
  self.state_covariance that allow to store the state mean and the state
  covariance respectively. When state_normalization = True, there are two
  main changes:
  1) The policy evaluation changes. Specifically, if the state mean is mu,
  the state covariance is cov, and the neural network computes function f, the
  policy takes the form pi: S -> A. Where p(s) = f( diag(cov)^{-1/2} (s-mu)).
  Where diag(cov)^{-1/2} stands for the diagonal of the state covariance
  raised to minus 1/2.
  2) Storing and reading a vectorized policy includes parameters encoding
  the global number of steps, state mean and state covariance. The vectorized
  parameters vector takes the form:
  [global_num_steps, state_mean, vectorized(state_covariance), nn_params]
  where vectorized(state_covariance) is a state_dim**2 vector made of a
  vectorized version of the state covariance matrix. When the
  state_normalization flag is on, all methods including init,
  get_action, get_initial, update work under this underlying assumption.
  """

  def __init__(self,
               state_dimensionality,
               action_dimensionality,
               first_hidden_size,
               second_hidden_size,
               low=None,
               high=None,
               state_normalization=False):
    """Sets up parameters of the unstructured neural network.

    Args:
      state_dimensionality: dimensionality of the first layer
      action_dimensionality: dimensionality of the last layer
      first_hidden_size: size of the first hidden layer
      second_hidden_size: size of the second hidden layer
      low: array of lower bounds for actions' dimensions
      high: array for upper bounds for actions' dimensions
      state_normalization: determines if state normalization is used or not
    """
    first_threshold = state_dimensionality + first_hidden_size - 1
    second_threshold = first_threshold + first_hidden_size + second_hidden_size
    second_threshold -= 1
    third_threshold = second_threshold + second_hidden_size
    third_threshold += action_dimensionality - 1
    fourth_threshold = third_threshold + first_hidden_size
    fifth_threshold = fourth_threshold + second_hidden_size

    nb_parameters = (state_dimensionality + first_hidden_size -
                     1) + (first_hidden_size + second_hidden_size -
                           1) + (second_hidden_size + action_dimensionality -
                                 1) + first_hidden_size + second_hidden_size
    vectorized_parameters = np.zeros(nb_parameters)

    first_column = vectorized_parameters[0:first_hidden_size]
    first_row = vectorized_parameters[first_hidden_size - 1:first_threshold]
    first_matrix = scipy.linalg.toeplitz(first_column, first_row)

    second_column = vectorized_parameters[first_threshold:first_threshold +
                                          second_hidden_size]
    second_row = vectorized_parameters[first_threshold + second_hidden_size -
                                       1:second_threshold]
    second_matrix = scipy.linalg.toeplitz(second_column, second_row)

    third_column = vectorized_parameters[second_threshold:second_threshold +
                                         action_dimensionality]
    third_row = vectorized_parameters[second_threshold + action_dimensionality -
                                      1:third_threshold]
    third_matrix = scipy.linalg.toeplitz(third_column, third_row)

    first_biases = vectorized_parameters[
        third_threshold:fourth_threshold].reshape((first_hidden_size, 1))
    second_biases = vectorized_parameters[
        fourth_threshold:fifth_threshold].reshape((second_hidden_size, 1))

    self.matrices = [first_matrix, second_matrix, third_matrix]
    self.biases = [first_biases, second_biases]

    self.state_dimensionality = state_dimensionality
    self.action_dimensionality = action_dimensionality
    self.first_hidden_size = first_hidden_size
    self.second_hidden_size = second_hidden_size

    self.first_threshold = first_threshold
    self.second_threshold = second_threshold
    self.third_threshold = third_threshold
    self.fourth_threshold = fourth_threshold
    self.fifth_threshold = fifth_threshold

    self.state_normalization = state_normalization
    if state_normalization:
      self.global_num_steps = 0
      self.state_mean = np.zeros(self.state_dimensionality)
      self.state_covariance = np.zeros(
          (self.state_dimensionality, self.state_dimensionality))

    def tanh(x):
      critical_bareer = 20.0
      if x > critical_bareer:
        return 1.0
      if x < -critical_bareer:
        return -1.0
      return 2.0 / (1.0 + math.exp(0.0 - 2.0 * x)) - 1.0

    def nonlinearity(state):
      for i in range(len(state)):
        state[i][0] = tanh(state[i][0])
      return state

    self.nonlinearity = nonlinearity

    self.low = low
    self.high = high
    super().__init__()

  def update(self, vectorized_parameters):

    if self.state_normalization:
      self.global_num_steps = vectorized_parameters[0]
      self.state_mean = vectorized_parameters[1:self.state_dimensionality + 1]
      cov_size = self.state_dimensionality**2
      cov_dims = (self.state_dimensionality, self.state_dimensionality)
      self.state_covariance = vectorized_parameters[self.state_dimensionality +
                                                    1:cov_size +
                                                    self.state_dimensionality +
                                                    1]
      self.state_covariance = np.reshape(self.state_covariance, cov_dims)
      vectorized_parameters = vectorized_parameters[1 + cov_size +
                                                    self.state_dimensionality:]

    first_column = vectorized_parameters[0:self.first_hidden_size]
    first_row = vectorized_parameters[self.first_hidden_size -
                                      1:self.first_threshold]
    first_matrix = scipy.linalg.toeplitz(first_column, first_row)

    second_column = vectorized_parameters[self.
                                          first_threshold:self.first_threshold +
                                          self.second_hidden_size]
    second_row = vectorized_parameters[self.first_threshold +
                                       self.second_hidden_size -
                                       1:self.second_threshold]
    second_matrix = scipy.linalg.toeplitz(second_column, second_row)

    third_column = vectorized_parameters[self.second_threshold:self
                                         .second_threshold +
                                         self.action_dimensionality]
    third_row = vectorized_parameters[self.second_threshold +
                                      self.action_dimensionality -
                                      1:self.third_threshold]
    third_matrix = scipy.linalg.toeplitz(third_column, third_row)

    first_biases = vectorized_parameters[self.third_threshold:self
                                         .fourth_threshold].reshape(
                                             (self.first_hidden_size, 1))
    second_biases = vectorized_parameters[self.fourth_threshold:self
                                          .fifth_threshold].reshape(
                                              (self.second_hidden_size, 1))

    self.matrices = [first_matrix, second_matrix, third_matrix]
    self.biases = [first_biases, second_biases]

  def get_action(self, state):
    if self.state_normalization:
      centered_state = state - self.state_mean
      squareroot_covariance = np.diag(self.state_covariance)
      squareroot_covariance = np.sqrt(squareroot_covariance)
      big_vl = np.power(10.0, 11)
      cov_mask = (squareroot_covariance < np.power(10.0, -8)) * big_vl
      squareroot_covariance = cov_mask + squareroot_covariance
      inverse_squareroot_covariance = 1.0 / squareroot_covariance
      state = inverse_squareroot_covariance * centered_state

    state = np.reshape(state, (len(state), 1))
    state = np.matmul(self.matrices[0], state)
    state = np.add(state, self.biases[0])
    state = (self.nonlinearity)(state)
    state = np.matmul(self.matrices[1], state)
    state = np.add(state, self.biases[1])
    state = (self.nonlinearity)(state)
    action = np.matmul(self.matrices[2], state)

    if self.low is not None and self.high is not None:
      action = np.tanh(action)
      for i in range(len(action)):
        action[i][0] = (
            action[i][0] * (self.high[i] - self.low[i]) / 2.0 +
            (self.low[i] + self.high[i]) / 2.0)

    return action

  def get_initial(self):
    # The initial policy is given by weights and biases taken independently at
    # random from the Gaussian distribution.
    np.random.seed(100)
    vec_first_biases = np.random.randn(self.first_hidden_size)
    vec_second_biases = np.random.randn(self.second_hidden_size)

    vec_first_vector = 1.0 / math.sqrt(float(
        self.first_hidden_size)) * np.random.randn(self.first_hidden_size +
                                                   self.state_dimensionality -
                                                   1)
    vec_second_vector = 1.0 / math.sqrt(float(
        self.second_hidden_size)) * np.random.randn(self.second_hidden_size +
                                                    self.first_hidden_size - 1)
    vec_third_vector = 1.0 / math.sqrt(float(
        self.action_dimensionality)) * np.random.randn(
            self.action_dimensionality + self.second_hidden_size - 1)
    vectorized_network = np.concatenate([
        vec_first_vector, vec_second_vector, vec_third_vector, vec_first_biases,
        vec_second_biases
    ])
    if self.state_normalization:
      num_state_normalization_parameters = 1 + self.state_dimensionality
      num_state_normalization_parameters += self.state_dimensionality**2
      vectorized_network = np.concatenate(
          [np.zeros(num_state_normalization_parameters), vectorized_network])
    return vectorized_network

  def get_total_num_parameters(self):
    total = (self.state_dimensionality + self.first_hidden_size -
             1) + (self.first_hidden_size + self.second_hidden_size -
                   1) + (self.second_hidden_size + self.action_dimensionality -
                         1) + self.first_hidden_size + self.second_hidden_size
    if self.state_normalization:
      total += self.state_dimensionality + self.state_dimensionality**2 + 1
    return total


def core_convolve(long_vector, short_vector, jump):
  index = 0
  final = []
  long_l = len(long_vector)
  short_l = len(short_vector)
  while index + short_l <= long_l:
    final.append(np.sum(long_vector[index:(index + short_l)] * short_vector))
    index += jump
  return np.array(final)


def convolve(list_of_vectors, weights, stride, biases, nonlinearity):
  """Convolves the batch of vectors with weight matrix.

  Applies a convolutional layer by convolving the batch of vectors with
  weight matrix. The convolutional is characterized by stride-scalar, bias
  vector and nonlinear mapping applied at the end.

  Args:
    list_of_vectors:
    weights: weight matrix
    stride:  stride-scalar defining the convolution
    biases:  vector of bias-terms
    nonlinearity:  nonlinear mapping applied at the end of the convolution

  Returns:
    Convolved batch of vectors.
  """
  final = []
  for i in range(len(weights)):
    conv_res_local = None
    for j in range(len(weights[i])):
      c = core_convolve(list_of_vectors[j], weights[i][j], stride)
      if conv_res_local is None:
        conv_res_local = c
      else:
        conv_res_local += c
    conv_res_local += biases[i] * np.ones(len(conv_res_local))
    r = nonlinearity(np.array(conv_res_local))
    final.append(r)
  return np.array(final)


class Conv1DPolicy(Policy):
  """Derives from Policy and encodes a convolutional policy.

  Convolutional policy that applies to the input state a series of 1d
  convolutions followed by the fully connected layer. This policy uses two
  element-wise nonlinearities: the first one is applied at the end of every
  convolutional layer. The second one is applied in the fully connected
  feedforward neural network.
  """

  def __init__(self,
               state_dimensionality,
               action_dimensionality,
               filter_sizes,
               strides,
               feature_detectors_sizes,
               nonlinearity,
               second_nonlinearity,
               nb_input_channels=3):
    self.state_dimensionality = state_dimensionality
    self.action_dimensionality = action_dimensionality
    self.filter_sizes = filter_sizes
    self.strides = strides
    self.feature_detectors_sizes = feature_detectors_sizes
    self.nb_input_channels = nb_input_channels

    self.biases = []
    self.weights = []
    for _ in range(len(filter_sizes)):
      (self.biases).append([])
      (self.weights).append([])
    self.nonlinearity = nonlinearity
    self.column = None
    self.row = None
    self.second_biases = None
    self.second_nonlinearity = second_nonlinearity
    self.final_s = self.state_dimensionality / self.nb_input_channels
    for i in range(len(self.filter_sizes)):
      jump = self.strides[i]
      d_init = self.final_s
      count = 0
      index = 0
      while index + self.filter_sizes[i] <= d_init:
        count += 1
        index += jump
      self.final_s = count
    super().__init__()

  def update(self, vectorized_parameters):
    self.biases = []
    self.weights = []
    for _ in range(len(self.filter_sizes)):
      (self.biases).append([])
      (self.weights).append([])
    index = 0
    for i in range(self.feature_detectors_sizes[0]):
      size = self.filter_sizes[0] * self.nb_input_channels
      (self.weights[0]).append(
          vectorized_parameters[index:index + size].reshape(
              self.nb_input_channels, self.filter_sizes[0]))
      index += size
      size = 1
      (self.biases[0]).append(vectorized_parameters[index:index + size])
      index += size
    for i in range(1, len(self.filter_sizes)):
      for _ in range(self.feature_detectors_sizes[i]):
        size = self.filter_sizes[i] * self.feature_detectors_sizes[i - 1]
        (self.weights[i]).append(
            vectorized_parameters[index:index + size].reshape(
                self.feature_detectors_sizes[i - 1], self.filter_sizes[i]))
        index += size
        size = 1
        (self.biases[i]).append(vectorized_parameters[index:index + size])
        index += size
    size1 = self.final_s * self.feature_detectors_sizes[
        len(self.feature_detectors_sizes) - 1]
    size2 = self.action_dimensionality
    self.row = np.array(vectorized_parameters[index:index + size1])
    self.column = np.array(
        vectorized_parameters[(index + size1 - 1):(index + size1 + size2 - 1)])
    index += size1 + size2 - 1
    self.second_biases = vectorized_parameters[index:]

  def get_action(self, state):
    channels = np.transpose(
        np.reshape(state, (self.final_s, self.nb_input_channels)))
    for i in range(len(self.filter_sizes)):
      channels = convolve(channels, self.weights[i], self.strides[i],
                          self.biases[i], self.nonlinearity)
    action = self.second_nonlinearity(
        np.matmul(
            scipy.linalg.toeplitz(self.column, self.row),
            channels.reshape((len(self.row), 1))) +
        (self.second_biases).reshape((len(self.column), 1)))
    return action

  def get_initial(self):
    init_convo = []
    num_unstructured = self.final_s * self.feature_detectors_sizes[len(
        self.feature_detectors_sizes) - 1] + 2 * self.action_dimensionality - 1
    num_unstructured_weights = self.final_s * self.feature_detectors_sizes[
        len(self.feature_detectors_sizes) - 1] + self.action_dimensionality - 1
    num_unstructured_biases = num_unstructured - num_unstructured_weights
    num_convo = (self.filter_sizes[0] * self.nb_input_channels +
                 1) * self.feature_detectors_sizes[0]
    for i in range(self.feature_detectors_sizes[0]):
      counter = self.filter_sizes[0] * self.nb_input_channels
      init_convo += (1.0 / math.sqrt(float(counter)) *
                     np.random.randn(counter)).tolist()
      init_convo += (np.random.randn(1)).tolist()
    for i in range(1, len(self.filter_sizes)):
      for _ in range(self.feature_detectors_sizes[i]):
        counter = self.filter_sizes[i] * self.feature_detectors_sizes[i - 1]
        init_convo += (1.0 / math.sqrt(float(counter)) *
                       np.random.randn(counter)).tolist()
        init_convo += (np.random.randn(1)).tolist()
      num_convo += self.filter_sizes[i] * self.feature_detectors_sizes[
          i] * self.feature_detectors_sizes[i -
                                            1] + self.feature_detectors_sizes[i]
    random_sequence = 2.0 * (np.random.rand(num_unstructured_weights) - 0.5)
    init_part_2 = (1.0 / math.sqrt(
        float(self.final_s * self.feature_detectors_sizes[
            len(self.feature_detectors_sizes) - 1]))) * random_sequence
    init_part_3 = np.random.randn(num_unstructured_biases)
    return np.array(init_convo + init_part_2.tolist() + init_part_3.tolist())

  def get_total_num_parameters(self):
    num_unstructured = self.final_s * self.feature_detectors_sizes[len(
        self.feature_detectors_sizes) - 1] + 2 * self.action_dimensionality - 1
    num_convo = (self.filter_sizes[0] * self.nb_input_channels +
                 1) * self.feature_detectors_sizes[0]
    for i in range(1, len(self.filter_sizes)):
      num_convo += self.filter_sizes[i] * self.feature_detectors_sizes[
          i] * self.feature_detectors_sizes[i -
                                            1] + self.feature_detectors_sizes[i]
    return num_unstructured + num_convo


class IdentityPolicy(Policy):
  """Derives from Policy and encodes identity policy.

  Trivial identity policy that outputs as action the state vector. That policy
  is not useful on its own but becomes very handy while designing hybrid
  policies that split state vector into chunks processed independently by
  different policies, concatenated together and finally, fed to ultimate policy
  that produces an action.
  """

  def update(self, vectorized_parameters):
    pass

  def get_action(self, state):
    return state

  def get_initial(self):
    return np.array([])

  def get_total_num_parameters(self):
    return 0


class HybridPolicy(Policy):
  """Derives from Policy and encodes hybrid policy.

  Hybrid policy that partitions input state vector into two sub-states. Those
  two sub-states are independently processed by two different policies. They
  outputs are being concatenated and given as an input state to the third
  policy that produces final action.
  """

  def __init__(self,
               first_policy,
               first_state_dim,
               second_policy,
               third_policy,
               flattened=True,
               renorm_nonlinearity=None,
               size_of_first_state_part=None,
               size_of_second_state_part=None):
    self.first_policy = first_policy
    self.first_state_dim = first_state_dim
    self.second_policy = second_policy
    self.third_policy = third_policy
    self.nb_params_1 = first_policy.get_total_num_parameters()
    self.nb_params_2 = second_policy.get_total_num_parameters()
    self.nb_params_3 = third_policy.get_total_num_parameters()
    self.total = self.nb_params_1 + self.nb_params_2 + self.nb_params_3
    self.flattened = flattened
    self.renorm_nonlinearity = renorm_nonlinearity
    self.size_of_first_state_part = size_of_first_state_part
    self.size_of_second_state_part = size_of_second_state_part
    super().__init__()

  def update(self, vectorized_parameters):
    vectorized_parameters_1 = copy.copy(
        vectorized_parameters[0:self.nb_params_1])
    vectorized_parameters_2 = copy.copy(
        vectorized_parameters[self.nb_params_1:(self.nb_params_1 +
                                                self.nb_params_2)])
    vectorized_parameters_3 = copy.copy(
        vectorized_parameters[(self.nb_params_1 + self.nb_params_2):])
    (self.first_policy).update(vectorized_parameters_1)
    (self.second_policy).update(vectorized_parameters_2)
    (self.third_policy).update(vectorized_parameters_3)

  def get_action(self, state):
    if not self.flattened:
      state = np.array(
          (state[0].reshape(self.size_of_first_state_part)).tolist() +
          (state[1].reshape(self.size_of_second_state_part)).tolist())
    state_1 = state[0:self.first_state_dim]
    state_2 = state[self.first_state_dim:]
    a1 = (self.first_policy).get_action(state_1)
    a1 = a1.reshape(len(a1))
    if self.renorm_nonlinearity is not None:
      a1 = self.renorm_nonlinearity(a1)
    a2 = (self.second_policy).get_action(state_2)
    a2 = a2.reshape(len(a2))
    if self.renorm_nonlinearity is not None:
      a2 = self.renorm_nonlinearity(a2)
    new_state = np.array(a1.tolist() + a2.tolist())
    return (self.third_policy).get_action(new_state)

  def get_initial(self):
    init_1 = ((self.first_policy).get_initial()).tolist()
    init_2 = ((self.second_policy).get_initial()).tolist()
    init_3 = ((self.third_policy).get_initial()).tolist()
    return np.array(init_1 + init_2 + init_3)

  def get_total_num_parameters(self):
    return self.total

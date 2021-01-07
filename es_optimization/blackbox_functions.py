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

r"""Library for creating different tensorized versions of blackbox functions.

Responsible for creating different tensorized versions of blackbox functions.
Each blackbox function accepts a list of tensors and outputs a list of tensors.
"""

import abc
from typing import Union, List, Any, Callable
import numpy as np
from es_optimization import policies
from es_optimization import rl_environments


class BlackBoxFunction(abc.ABC):
  r"""Abstract class for different blackbox functions.

  Class is responsible for creating different blackbox functions. In particular,
  it provides a way to define functions that take as arguments the parameters
  of neural networks and output the total reward obtained by applying
  policies defined by these neural networks in a particular RL environment.
  """

  @abc.abstractmethod
  def get_function(self):
    """Outputs the blackbox function.

    Outputs the blackbox function. The blackbox function takes as an input
    the list of numpy arrays (each numpy array is a different argument)
    and outputs a list of numpy arrays.

    Args:

    Returns:
    """
    raise NotImplementedError('Abstract method')


class TesterBlackBox(BlackBoxFunction):
  r"""Class responsible for creating simple tester blackbox functions.

  Class inheriting from BlackBoxFunction and responsible for creating function
  of the form:
        f(x_0, x_1, x_2, x_3, x_4) =
                -[(x_0-1.0)^2+(x_1-2.0)^2+(x_2-3.0)^2+(x_3-4.0)^2+(x_4-5.0)^2].
  """

  def __init__(self):
    pass

  def get_function(self):

    def blackbox_function(parameters, _):
      value_0 = (parameters[0] - 1.0) * (parameters[0] - 1.0)
      value_1 = (parameters[1] - 2.0) * (parameters[1] - 2.0)
      value_2 = (parameters[2] - 3.0) * (parameters[2] - 3.0)
      value_3 = (parameters[3] - 4.0) * (parameters[3] - 4.0)
      value_4 = (parameters[4] - 5.0) * (parameters[4] - 5.0)
      return -(value_0 + value_1 + value_2 + value_3 + value_4)

    return blackbox_function


def rl_rollout(policy,
               environment,
               number_of_steps):
  """Runs <number_of_steps> steps in the <environment> by conducting <policy>.

  Args:
    policy: applied policy
    environment:  environment in which policy is deployed. Object environment
      should provide three methods, namely *restart* - responsible for resetting
      the environment and not taking any arguments, *deterministic_start* -
      responsible for setting up deterministic initial configuration of the
      environment and *step()* - taking as an argument an action and outputting
      a list of at least three elements. These elements are: [new_state, reward,
      done, _] where: <new_state> - new state after applying <action>, <reward>
        - the immediate reward obtained after applying <action> in the current
        state and transitioning to <new_state>, <done> - boolean that indicates
        whether the current episode has been completed. Examples of RL
        environments that match this framework are all OpenAI Gym tasks
    number_of_steps:  upper bound on the number of steps of the single rollout
      (the episode might be potentially completed before <number_of_steps> steps
      are conducted)

  Returns:
    Total cost of the rollout (negated total reward).
  """
  state = environment.deterministic_start()
  sum_reward = 0
  steps = number_of_steps

  for _ in range(steps):
    proposed_action = policy.get_action(state)
    proposed_action = np.reshape(proposed_action, (len(proposed_action)))
    state, reward, done, _ = environment.step(proposed_action)
    sum_reward += reward
    if done:
      break
  return float(0.0 - sum_reward)


def renormalize(state, mean_state_vector,
                std_state_vector):
  """Outputs renormalized state vector using mean and std dev information.

  Outputs renormalized state vector given by the following formula:
    state_renormalized = (state - mean_state_vector) / renormalized_state_vector
  (all operations conducted element-wise).
  Args:
    state: state vector to be renormalized
    mean_state_vector: vector of mean dimension values
    std_state_vector: vector of std devs for different dimensions

  Returns:
    renormalized state vector
  """
  if mean_state_vector is None:
    return state

  if (isinstance(mean_state_vector, list) and not mean_state_vector):
    return state
  elif (isinstance(mean_state_vector, np.ndarray) and
        mean_state_vector.size == 0):
    return state
  else:
    state_shape = state.shape
    centralized_state_vector = [
        a - b for a, b in zip(state.flatten().tolist(), mean_state_vector)
    ]
    for i in range(len(std_state_vector)):
      if std_state_vector[i] == 0.0:
        std_state_vector[i] = 1.0
    renormalized_state = [
        a / b for a, b in zip(centralized_state_vector, std_state_vector)
    ]
    renorm_state = np.array(renormalized_state).reshape(state_shape)
    return renorm_state


def renormalize_with_epsilon(
    state, mean_state_vector,
    std_state_vector):
  """Outputs renormalized state vector using mean and std dev information.

  Outputs renormalized state vector given by the following formula:
    state_renormalized = (state - mean_state_vector) /
                          std_state_vector + epsilon
  (all operations conducted element-wise).

  epsilon prevents divide by zero errors and is set to 1e-8

  Args:
    state: matrix, state to be renormalized
    mean_state_vector: list of mean dimension values
    std_state_vector: list of std devs for different dimensions

  Returns:
    renormalized state vector
  """
  if (isinstance(mean_state_vector, np.ndarray) and
      mean_state_vector.size > 0) or (isinstance(mean_state_vector, list) and
                                      mean_state_vector):
    state_shape = state.shape
    state = state.flatten()
    mean = np.asarray(mean_state_vector)
    std = np.asarray(std_state_vector)
    norm_state = (state - mean) / (std + 1e-8)
    norm_state = norm_state.reshape(state_shape)
    return norm_state
  else:
    return state


def rl_extended_rollout(policy,
                        hyperparameters,
                        environment,
                        number_of_steps):
  """Runs <number_of_steps> steps in the <environment> by conducting <policy>.

  Args:
    policy: applied policy
    hyperparameters: the list of hyperparameters
    environment:  environment in which policy is deployed. Object environment
      should provide three methods, namely *restart* - responsible for resetting
      the environment and not taking any arguments, *deterministic_start* -
      responsible for setting up deterministic initial configuration of the
      environment and *step()* - taking as an argument an action and outputting
      a list of at least three elements. These elements are
                  [new_state, reward, done, _] where: <new_state> - new state
                    after applying <action>, <reward> - the immediate reward
                    obtained after applying <action> in the current state and
                    transitioning to <new_state>, <done> - boolean that
                    indicates whether the current episode has been completed.
                    Examples of RL environments that match this framework are
                    all OpenAI Gym tasks
    number_of_steps:  upper bound on the number of steps of the single rollout
      (the episode might be potentially completed before <number_of_steps> steps
      are conducted)

  Returns:
    Total cost of the rollout (negated total reward).
  """

  state = environment.deterministic_start()

  sum_reward = 0
  steps = number_of_steps
  # Vector such that its i^th entry stores the sum of i^th dimensions of the
  # states visited so far.
  sum_state_vector = []
  # Vector such that its i^th entry stores the sum of squares of i^th dimensions
  # of the states visited so far.
  squares_state_vector = []
  nb_points = 0
  mean_state_vector = []
  std_state_vector = []
  hyperparameters = hyperparameters[1:]
  if hyperparameters:
    state_dim = int(len(hyperparameters) / 2)
    mean_state_vector = hyperparameters[:state_dim]
    std_state_vector = hyperparameters[state_dim:]
    sum_state_vector = [0.0] * state_dim
    squares_state_vector = [0.0] * state_dim

  for _ in range(steps):
    proposed_action = policy.get_action(
        renormalize(state, mean_state_vector, std_state_vector))
    proposed_action = np.reshape(proposed_action, (len(proposed_action)))
    state, reward, done, _ = environment.step(proposed_action)

    if hyperparameters:
      # Updating sum_state_vector and squares_state_vector based on the new
      # visited state (look above for the definition of: sum_state_vector and
      # squares_state_vector).
      nb_points += 1
      squared = [x * x for x in state.tolist()]
      sum_state_vector = [sum(x) for x in zip(sum_state_vector, state.tolist())]
      squares_state_vector = [
          sum(x) for x in zip(squares_state_vector, squared)
      ]
    sum_reward += reward
    if done:
      break

  if hyperparameters:
    return [
        float(sum_reward),
        [float(nb_points)] + sum_state_vector + squares_state_vector
    ]
  else:
    return [float(sum_reward), []]


def get_environment(environment_name):
  """Gets the environment given the environment name.

  Args:
    environment_name: The name of the environment.

  Returns:
    environment: The corresponding instantiated RL environment.

  Raises:
    ValueError: Environment name not from allowed set of environments.
  """

  environment_class_dict = {
      'ContMountainCar': rl_environments.ContMountainCar,
      'Pendulum': rl_environments.Pendulum
  }

  try:
    env_class = environment_class_dict[environment_name]()
  except:
    raise ValueError('Environment Name not in allowed set of environments.')
  return env_class

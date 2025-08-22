# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Gym environment implementation."""

import copy
import json
from typing import Any, Dict, List, Optional, Tuple

import gym
from gym.utils import seeding
import numpy as np
import pandas as pd

from sd_gym import bptk_simulator
from sd_gym import core
from sd_gym import pysd_simulator


class SDEnv(gym.Env):
  """Gym environnment based on system dynamics models.

  SDEnv accepts a .stmx or .xmile file and creates an environment that steps
  through the simulation encoded in the file.
  """

  backends = {
      'BPTK_Py': bptk_simulator.BPTKSDSimulator,
      'PySD': pysd_simulator.PySDSimulator,
  }

  def __init__(self, params):
    # Configure simulator
    if isinstance(params.simulator, str):
      try:
        params.simulator = self.backends.get(params.simulator)
      except KeyError as exc:
        raise AttributeError(
            'params.simulator must be one of ["BPTK_Py", "PySD"]'
        ) from exc

    sd_sim = params.simulator(params)
    self.state = core.State(sd_sim,
                            sd_sim.get_start_time(),
                            sd_sim.get_initial_conditions())
    self.params = params

    self.is_reset = False

    # Initialize spaces

    # Only constants can be actioned.
    # Default to all constants if no actionables are specified.
    if params.actionables:
      self._check_var_names_valid(
          'actionables', params.actionables, self.state.sd_sim.list_constants()
      )
    self.actionables = params.actionables or self.state.sd_sim.list_constants()

    if params.sd_var_limits_override:
      self._check_var_names_valid(
          'sd_var_limits_override',
          params.sd_var_limits_override.keys(),
          self.actionables,
      )

    if params.categorical_sd_vars:
      self._check_var_names_valid(
          'categorical_sd_vars',
          params.categorical_sd_vars.keys(),
          self.actionables
      )

    self.action_space = self._generate_action_space(self.actionables, params)

    # All variables can be observed unless they are actionable.
    # Default to all variables if no observables are specified.
    if params.observables:
      self._check_var_names_valid(
          'observables', params.observables, self.state.sd_sim.list_variables()
      )
    self.observables = list(
        set(params.observables or self.state.sd_sim.list_variables())
        - set(self.actionables)
    )

    self.observation_space = self._generate_obs_space(
        self.observables, (params.observation_len,)
    )

    stocks_constants = set(
        self.state.sd_sim.list_constants() + self.state.sd_sim.list_stocks()
    )

    # Initial conditions can only be set for non-actionable stocks or constants.
    if params.initial_conditions_override:
      self._check_var_names_valid(
          'initial_conditions',
          params.initial_conditions_override.keys(),
          list(stocks_constants - set(self.actionables)),
      )

    # check that env_dt is a non-negative multiple of sd_dt.
    self._env_dt = params.env_dt or self.state.sd_sim.get_timestep()
    self.history = []  # type: List[Tuple[core.State, Any]]
    self.state = None  # type: Optional[core.State]
    self.reward_fn = None  # type: Optional[core.RewardFn]

    # Copy params so if environment mutates params it is contained to this
    # environment instance.
    self.initial_params = copy.deepcopy(params)

  def _generate_obs_space(self, var_names, shape):
    """Generates the observation space for the environment.

    The observation space is a dict of all observable variables.
    Type defaults to a float without limits, to minimize unexpected breakages.
    """
    return gym.spaces.Dict(
        {
            k: gym.spaces.Box(
                -1 * np.inf, np.inf, dtype=np.float64, shape=shape
            )
            for k in var_names
        }
    )

  def _generate_action_space(self, var_names, params):
    """Generates the action space for the environment, using parameters."""
    spaces = {}

    # Continuous variables (including ints) i.e. all actionalble vars that are
    # not explicitly made discrete.

    # Set type to float if unit has '/' or 'per ' in it (inferring division),
    # or if it's 'unitless' or 'fraction' or 'dmnl'. Then override with params.
    units_types = {var: np.int64 for var in var_names}
    overrides_units_types = {
        i.lower(): j for i, j in params.sd_units_types.items()
    }
    for k in units_types:
      units = self.state.sd_sim.list_units().get(k, '').lower()
      if (
          '/' in units
          or 'per ' in units
          or 'unitless' in units
          or 'fraction' in units
          or 'dmnl' in units
      ):
        units_types[k] = np.float64
      if units.lower() in overrides_units_types:
        units_types[k] = overrides_units_types[units.lower()]

    # Override limits with units limits, then model-specified limits,
    # then with params.
    box_vars = [
        var for var in var_names if var not in params.categorical_sd_vars.keys()
    ]
    limits = {}
    for k, v in units_types.items():
      if k not in box_vars:
        continue
      if np.issubdtype(v, np.integer):
        int_max = np.floor(np.iinfo(v).max / 2)
        limits[k] = (-1 * int_max, int_max)
      else:
        float_max = np.finfo(v).max / 2
        limits[k] = (-1 * float_max, 1 * float_max)

    defaults_units_limits = {
        i.lower(): j for i, j in params.default_sd_units_limits.items()
    }
    for k in limits:
      units = self.state.sd_sim.list_units().get(k, '').lower()
      if units.lower() in defaults_units_limits:
        self._override_limits(limits, {k: defaults_units_limits[units.lower()]})

    self._override_limits(limits, self.state.sd_sim.list_limits())
    self._override_limits(limits, params.sd_var_limits_override)

    box_spaces = {
        k: gym.spaces.Box(
            limits[k][0], limits[k][1], dtype=units_types[k], shape=(1,)
        )
        for k in box_vars
    }
    spaces.update(box_spaces)

    # Discrete variables
    discrete_spaces = {
        k: gym.spaces.Box(0, len(v) - 1, dtype=np.int32, shape=())
        for k, v in params.categorical_sd_vars.items()
    }
    spaces.update(discrete_spaces)

    if params.parameterize_action_space:
      return gym.spaces.Dict(
          {
              k: gym.spaces.Tuple([gym.spaces.Discrete(2), v])
              for k, v in spaces.items()
          }
      )
    else:
      return gym.spaces.Dict(spaces)

  def step(
      self, action
  ):
    """Run one timestep of the environment's dynamics.

    Args:
        action: An action provided by the agent. A member of `action_space`.

    Returns:
        observation: Agent's observation of the current environment. A member
          of `observation_space`.
        reward: Scalar reward returned after previous action. This should be the
          output of a `RewardFn` provided by the agent.
        done: Whether the episode has ended, in which case further step() calls
          will return undefined results.
        info: A dictionary with auxiliary diagnostic information.

    Raises:
      AssertionError: If called before first reset().
      gym.error.InvalidAction: If `action` is not in `self.action_space`.
    """
    if self.state is None:
      raise AssertionError(
          'State is None. State must be initialized before taking a step.'
          'If using core.FairnessEnv, subclass and implement necessary methods.'
      )

    if not self.action_space.contains(action):
      raise gym.error.InvalidAction('Invalid action: %s' % action)

    self.history.append((self.state, action))

    current_time = self.state.time
    next_end_time = current_time + self._env_dt

    # Make sure that we don't exceed model end time.
    next_end_time = min(next_end_time, self.state.sd_sim.get_stop_time())

    # Run simulation forward and return pd.DataFrame
    outputs = self.state.sd_sim.run(
        next_end_time,
        self.params.initial_conditions_override,
        self._prep_actions(action, next_end_time),
    )

    self.state.update(self.state.sd_sim.get_current_time(), outputs)
    observation = self._get_observable_state()

    assert self.observation_space.contains(observation), (
        'Observation %s is not contained in self.observation_space'
        % observation
    )

    # Compute a reward_fn if one is given.
    reward = 0
    if self.reward_fn:
      reward = self.reward_fn(self._get_all_state())
    return observation, reward, self._is_done(), {}

  def _prep_actions(self, action, next_end_time):
    """Preprocesses actions for injection into the simulation."""
    var_actions = {}
    for k, v in action.items():
      if self.params.parameterize_action_space:
        take_action, val = v
        if not take_action:
          continue
      else:
        val = v

      if k in self.params.categorical_sd_vars:
        val = self.params.categorical_sd_vars[k][val.item()]
      elif isinstance(val, np.ndarray) and not val.ndim:
        val = val.item()
      elif isinstance(val, (np.ndarray, list)):
        val = pd.Series(
            val, index=np.linspace(self.state.time, next_end_time, len(val))
        )

      var_actions[k] = val

    return var_actions

  def _is_done(self):
    """Checks if model end time has been reached."""
    return np.isclose(self.state.time, self.state.sd_sim.get_stop_time())

  def _get_all_state(self):
    """Returns a dictionary of the current state."""
    all_state = {}
    if not self.state.obs_timeseries.empty:
      for var_name in self.state.sd_sim.list_variables():
        all_state[var_name] = self.state.obs_timeseries[var_name].values[-1]
    return all_state

  def _get_observable_state(self):
    """Returns a dictionary of the entire history of observable states.

    Agents may prefer to restrict observations to the last x rows of this
    data frame because otherwise the data changes in size each step.
    """
    obs_state = {}
    if not self.state.obs_timeseries.empty:
      for var_name in self.observables:
        var_series = self.state.obs_timeseries[var_name].values
        obs_var_series = var_series[-1 * self.params.observation_len :]
        obs_state[var_name] = obs_var_series
    return obs_state

  def reset(self):
    """Resets the simulation.

    Reloads SD model file, resets time to initial time and configures
    some of the parameters in the internal params file if they
    were not assigned.

    Returns:
      observation: The observable features for the first interaction.
    """
    self.is_reset = True
    sd_sim = self.params.simulator(self.params)
    self.state = core.State(sd_sim,
                            sd_sim.get_start_time(),
                            sd_sim.get_initial_conditions())
    if self.params.reward_function:
      self.reward_fn = self.params.reward_function.reset(self._get_all_state())
    self.history = []
    return self._get_observable_state()

  def seed(self, seed = None):  # pytype: disable=signature-mismatch  # overriding-return-type-checks
    """Sets the seed for this env's random number generator."""
    rng, seed = seeding.np_random(seed)
    self.state.rng = rng
    return [seed]

  def serialize_history(self):
    """Serialize history to JSON.

    Returns:
      A string containing a serialized JSON representation of the environment's
      history.
    """
    # Sanitize history by handling non-json-serializable state.
    sanitized_history = [
        (json.loads(state.to_json()), action) for state, action in self.history
    ]
    return json.dumps(
        {'environment': repr(self.__class__), 'history': sanitized_history},
        cls=core.GymEncoder,
        sort_keys=True,
    )

  def _check_var_names_valid(self, param, var_names, var_list):
    invalid_vars = [v for v in var_names if v not in var_list]
    if invalid_vars:
      raise ValueError(
          'Invalid variables specified as %s: %s' % (param, invalid_vars)
      )

  def _override_limits(
      self,
      limits,
      overrides,
  ):
    if overrides:
      for k, v in overrides.items():
        if k in limits:
          limits[k] = (
              v[0] if v[0] is not None else limits[k][0],
              v[1] if v[1] is not None else limits[k][1],
          )

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

"""params, state implementations, and abstract classes for SD Gym environment."""

import abc
import enum
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import attr
import numpy as np
import pandas as pd


@attr.s
class Params(object):
  """System dynamics environment parameters."""
  asdict = attr.asdict

  # Location of a .stmx, .xmile file.
  sd_file = attr.ib()  # type: str

  # Timestep at which the internal numerical ODE solver simulates the SD model.
  # Determined automatically if not set.
  sd_dt = attr.ib(default=None)  # type: Optional[float]

  # Time between environment steps.
  # Defaults to `sd_dt` if not set here.
  env_dt = attr.ib(default=None)  # type: Optional[float]

  # Override start and stop times defined in the model file.
  starttime = attr.ib(default=None)  # type: Optional[float]
  stoptime = attr.ib(default=None)  # type: Optional[float]

  # The names of the variables that are observable.
  # Defaults to all variables.
  observables = attr.ib(factory=list)  # type: Optional[list[str]]

  # Length of observations.
  # Defaults to 1 i.e. only the timestep of the simulation.
  observation_len = attr.ib(default=1)  # type: Optional[int]

  # The names of the variables that are actionable.
  # Defaults to all stocks and constants.
  actionables = attr.ib(factory=list)  # type: Optional[list[str]]

  # Whether to have an additional indicator for taking an action or not.
  parameterize_action_space = attr.ib(default=False) # type: Optional[bool]

  # Map of SD units to numpy types.
  sd_units_types = attr.ib(factory=dict)  # type: Optional[Dict[str, np.dtype]]

  # Max and min values for actions.
  default_sd_units_limits = attr.ib(factory=dict)  # type: Optional[Dict[str, Tuple[float, float]]]
  sd_var_limits_override = attr.ib(factory=dict)  # type: Optional[Dict[str, Tuple[float, float]]]

  # Variables that are categorical
  categorical_sd_vars = attr.ib(factory=dict)  # type: Optional[Dict[str, List[float]]]

  # Set initial conditions for variables.
  initial_conditions_override = attr.ib(factory=dict)  # type: Optional[Dict[str, float]]

  # Set a reward function.
  reward_function = attr.ib(default=None)  # type: Optional[RewardFn]

  # Which package to use to run the SD simulation
  simulator = attr.ib(default='BPTK_Py')  # type: Union[str, SDSimulator]


# Values with associated with this key within dictionaries are given
# special treatment as RandomState internals during JSON serialization /
# deserialization.  This works around an issue where RandomState itself fails
# to serialize.
RANDOM_STATE_KEY = '__random_state__'


class GymEncoder(json.JSONEncoder):
  """Encoder to handle common gym and numpy objects."""

  def default(self, o):
    # First check if the object has a to_jsonable() method which converts it to
    # a representation that can be json encoded.
    try:
      return o.to_jsonable()
    except AttributeError:
      pass

    if callable(o):
      return {'callable': o.__name__}

    if isinstance(o, (bool, np.bool_)):
      return int(o)

    if isinstance(o, enum.Enum):
      return {'__enum__': str(o)}

    if isinstance(o, np.ndarray):
      return o.tolist()
    if isinstance(o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                      np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
      return int(o)
    if isinstance(o, (bool, np.bool_)):
      return str(o)
    if isinstance(o, (np.float16, np.float32, np.float64)):
      return float(o)
    if isinstance(o, np.random.RandomState):
      state = o.get_state()
      return {
          RANDOM_STATE_KEY:
              (state[0], state[1].tolist(), state[2], state[3], state[4])
      }
    if isinstance(o, Params) or isinstance(o, State):
      return o.asdict()
    return json.JSONEncoder.default(self, o)


def to_json(dictionary, sort_keys=True, **kw):
  return json.dumps(dictionary, cls=GymEncoder, sort_keys=sort_keys, **kw)


@attr.s(cmp=False)
class State(object):
  """The state of the SD environment.

  Includes the current state of the systems dynamics model variables as well as
  the model itself. We include the model as part of the state because its
  parameters and equations are mutable.
  """

  asdict = attr.asdict

  # Simulation object
  sd_sim = attr.ib(default=None)  # type: Optional[SDSimulator]

  # Current time
  time = attr.ib(default=None)  # type: Optional[float]

  # Timeseries of all observations
  obs_timeseries = attr.ib(default=None)  # type: Optional[pd.DataFrame]

  def update(self, new_time, next_timeseries):
    # The first row of the dataset is a repeat of the last existing row
    # Drop last row of current time series since initial conditions may change
    # through actions.
    if len(self.obs_timeseries) <= 1:
      self.obs_timeseries = next_timeseries
    else:
      idx = self.obs_timeseries.index[:-1]
      self.obs_timeseries = pd.concat(
          [self.obs_timeseries.loc[idx], next_timeseries]
      )
    self.time = new_time

  def to_json(self):
    return to_json(self)

  def __eq__(self, other):
    return self.to_json() == other.to_json()

  def __ne__(self, other):
    return self.to_json() != other.to_json()


class RewardFn(abc.ABC):
  """Base reward function.

  A reward function describes how to extract a scalar reward from state or
  changes in state.

  Subclasses should override the __call__ function.
  """

  @abc.abstractmethod
  def __call__(self, observation):
    raise NotImplementedError

  @abc.abstractmethod
  def reset(self, obs):
    raise NotImplementedError


class SDSimulator(abc.ABC):
  """Abstract class for wrapping systems dynamics simulation software."""

  @abc.abstractmethod
  def run(
      self,
      stop_time,
      initial_conditions,
      var_actions,
  ):
    """Runs the systems dynamics model forward in time.

    Args:
      stop_time: Time simulation ends
      initial_conditions: Initial conditions to set only at beginning.
      var_actions: Actions provided by the agent anytime during simulationg.

    Returns:
      df: A pandas DataFrame containing the state of each model variable
        at each simulated time step.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def reset(self):
    """Resets the simulation so that it matches the original sd model file."""
    raise NotImplementedError

  @abc.abstractmethod
  def list_variables(self):
    """Lists the names of all variables."""
    raise NotImplementedError

  @abc.abstractmethod
  def list_stocks(self):
    """Lists the names of all stock variables."""
    raise NotImplementedError

  @abc.abstractmethod
  def list_constants(self):
    """Lists the names of all non-stock constant variables."""
    raise NotImplementedError

  @abc.abstractmethod
  def list_limits(self):
    """Lists the names and limits of all variables."""
    raise NotImplementedError

  @abc.abstractmethod
  def list_units(self):
    """Lists the names and units of all variables."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_start_time(self):
    """Gets the start time for the simulation."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_stop_time(self):
    """Gets the stop time for the simulation."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_timestep(self):
    """Gets the time step for the simulation."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_current_time(self):
    """Gets the current time in the simulation."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_initial_conditions(self):
    """Gets the initial conditions for the simulation."""
    raise NotImplementedError

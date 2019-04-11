"""Adds random forces to the base of Minitaur during the simulation steps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = os.path.dirname(os.path.dirname(parentdir))
os.sys.path.insert(0,parentdir)

import math
import numpy as np
from pybullet_envs.minitaur.envs import env_randomizer_base

_PERTURBATION_START_STEP = 100
_PERTURBATION_INTERVAL_STEPS = 200
_PERTURBATION_DURATION_STEPS = 10
_HORIZONTAL_FORCE_UPPER_BOUND = 120
_HORIZONTAL_FORCE_LOWER_BOUND = 240
_VERTICAL_FORCE_UPPER_BOUND = 300
_VERTICAL_FORCE_LOWER_BOUND = 500


class MinitaurPushRandomizer(env_randomizer_base.EnvRandomizerBase):
  """Applies a random impulse to the base of Minitaur."""

  def __init__(
      self,
      perturbation_start_step=_PERTURBATION_START_STEP,
      perturbation_interval_steps=_PERTURBATION_INTERVAL_STEPS,
      perturbation_duration_steps=_PERTURBATION_DURATION_STEPS,
      horizontal_force_bound=None,
      vertical_force_bound=None,
  ):
    """Initializes the randomizer.

    Args:
      perturbation_start_step: No perturbation force before the env has advanced
        this amount of steps.
      perturbation_interval_steps: The step interval between applying
        perturbation forces.
      perturbation_duration_steps: The duration of the perturbation force.
      horizontal_force_bound: The lower and upper bound of the applied force
        magnitude when projected in the horizontal plane.
      vertical_force_bound: The z component (abs value) bound of the applied
        perturbation force.
    """
    self._perturbation_start_step = perturbation_start_step
    self._perturbation_interval_steps = perturbation_interval_steps
    self._perturbation_duration_steps = perturbation_duration_steps
    self._horizontal_force_bound = (
        horizontal_force_bound if horizontal_force_bound else
        [_HORIZONTAL_FORCE_LOWER_BOUND, _HORIZONTAL_FORCE_UPPER_BOUND])
    self._vertical_force_bound = (
        vertical_force_bound if vertical_force_bound else
        [_VERTICAL_FORCE_LOWER_BOUND, _VERTICAL_FORCE_UPPER_BOUND])

  def randomize_env(self, env):
    """Randomizes the simulation environment.

    Args:
      env: The Minitaur gym environment to be randomized.
    """
    pass

  def randomize_step(self, env):
    """Randomizes simulation steps.

    Will be called at every timestep. May add random forces/torques to Minitaur.

    Args:
      env: The Minitaur gym environment to be randomized.
    """
    base_link_ids = env.minitaur.chassis_link_ids
    if env.env_step_counter % self._perturbation_interval_steps == 0:
      self._applied_link_id = base_link_ids[np.random.randint(
          0, len(base_link_ids))]
      horizontal_force_magnitude = np.random.uniform(
          self._horizontal_force_bound[0], self._horizontal_force_bound[1])
      theta = np.random.uniform(0, 2 * math.pi)
      vertical_force_magnitude = np.random.uniform(
          self._vertical_force_bound[0], self._vertical_force_bound[1])
      self._applied_force = horizontal_force_magnitude * np.array(
          [math.cos(theta), math.sin(theta), 0]) + np.array(
              [0, 0, -vertical_force_magnitude])

    if (env.env_step_counter % self._perturbation_interval_steps <
        self._perturbation_duration_steps) and (env.env_step_counter >=
                                                self._perturbation_start_step):
      env.pybullet_client.applyExternalForce(
          objectUniqueId=env.minitaur.quadruped,
          linkIndex=self._applied_link_id,
          forceObj=self._applied_force,
          posObj=[0.0, 0.0, 0.0],
          flags=env.pybullet_client.LINK_FRAME)

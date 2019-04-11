"""Randomize the minitaur_gym_alternating_leg_env when reset() is called.

The randomization include swing_offset, extension_offset of all legs that mimics
bent legs, desired_pitch from user input, battery voltage and motor damping.
"""

import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = os.path.dirname(os.path.dirname(parentdir))
os.sys.path.insert(0,parentdir)

import numpy as np
import tensorflow as tf
from pybullet_envs.minitaur.envs import env_randomizer_base

# Absolute range.
NUM_LEGS = 4
BATTERY_VOLTAGE_RANGE = (14.8, 16.8)
MOTOR_VISCOUS_DAMPING_RANGE = (0, 0.01)


class MinitaurAlternatingLegsEnvRandomizer(
    env_randomizer_base.EnvRandomizerBase):
  """A randomizer that changes the minitaur_gym_alternating_leg_env."""

  def __init__(self,
               perturb_swing_bound=0.1,
               perturb_extension_bound=0.1,
               perturb_desired_pitch_bound=0.01):
    super(MinitaurAlternatingLegsEnvRandomizer, self).__init__()
    self.perturb_swing_bound = perturb_swing_bound
    self.perturb_extension_bound = perturb_extension_bound
    self.perturb_desired_pitch_bound = perturb_desired_pitch_bound

  def randomize_env(self, env):
    perturb_magnitude = np.random.uniform(
        low=-self.perturb_swing_bound,
        high=self.perturb_swing_bound,
        size=NUM_LEGS)
    env.set_swing_offset(perturb_magnitude)
    tf.logging.info("swing_offset: {}".format(perturb_magnitude))

    perturb_magnitude = np.random.uniform(
        low=-self.perturb_extension_bound,
        high=self.perturb_extension_bound,
        size=NUM_LEGS)
    env.set_extension_offset(perturb_magnitude)
    tf.logging.info("extension_offset: {}".format(perturb_magnitude))

    perturb_magnitude = np.random.uniform(
        low=-self.perturb_desired_pitch_bound,
        high=self.perturb_desired_pitch_bound)
    env.set_desired_pitch(perturb_magnitude)
    tf.logging.info("desired_pitch: {}".format(perturb_magnitude))

    randomized_battery_voltage = np.random.uniform(BATTERY_VOLTAGE_RANGE[0],
                                                   BATTERY_VOLTAGE_RANGE[1])
    env.minitaur.SetBatteryVoltage(randomized_battery_voltage)
    tf.logging.info("battery_voltage: {}".format(randomized_battery_voltage))

    randomized_motor_damping = np.random.uniform(MOTOR_VISCOUS_DAMPING_RANGE[0],
                                                 MOTOR_VISCOUS_DAMPING_RANGE[1])
    env.minitaur.SetMotorViscousDamping(randomized_motor_damping)
    tf.logging.info("motor_damping: {}".format(randomized_motor_damping))

"""Randomize the minitaur_gym_env when reset() is called."""
import random

import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = os.path.dirname(os.path.dirname(parentdir))
os.sys.path.insert(0,parentdir)

import numpy as np
from pybullet_envs.minitaur.envs import env_randomizer_base

# Relative range.
MINITAUR_BASE_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
MINITAUR_LEG_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
# Absolute range.
BATTERY_VOLTAGE_RANGE = (14.8, 16.8)  # Unit: Volt
MOTOR_VISCOUS_DAMPING_RANGE = (0, 0.01)  # Unit: N*m*s/rad (torque/angular vel)
MINITAUR_LEG_FRICTION = (0.8, 1.5)  # Unit: dimensionless


class MinitaurEnvRandomizer(env_randomizer_base.EnvRandomizerBase):
  """A randomizer that change the minitaur_gym_env during every reset."""

  def __init__(self,
               minitaur_base_mass_err_range=MINITAUR_BASE_MASS_ERROR_RANGE,
               minitaur_leg_mass_err_range=MINITAUR_LEG_MASS_ERROR_RANGE,
               battery_voltage_range=BATTERY_VOLTAGE_RANGE,
               motor_viscous_damping_range=MOTOR_VISCOUS_DAMPING_RANGE):
    self._minitaur_base_mass_err_range = minitaur_base_mass_err_range
    self._minitaur_leg_mass_err_range = minitaur_leg_mass_err_range
    self._battery_voltage_range = battery_voltage_range
    self._motor_viscous_damping_range = motor_viscous_damping_range

  def randomize_env(self, env):
    self._randomize_minitaur(env.minitaur)

  def _randomize_minitaur(self, minitaur):
    """Randomize various physical properties of minitaur.

    It randomizes the mass/inertia of the base, mass/inertia of the legs,
    friction coefficient of the feet, the battery voltage and the motor damping
    at each reset() of the environment.

    Args:
      minitaur: the Minitaur instance in minitaur_gym_env environment.
    """
    base_mass = minitaur.GetBaseMassesFromURDF()
    randomized_base_mass = random.uniform(
        np.array(base_mass) * (1.0 + self._minitaur_base_mass_err_range[0]),
        np.array(base_mass) * (1.0 + self._minitaur_base_mass_err_range[1]))
    minitaur.SetBaseMasses(randomized_base_mass)

    leg_masses = minitaur.GetLegMassesFromURDF()
    leg_masses_lower_bound = np.array(leg_masses) * (
        1.0 + self._minitaur_leg_mass_err_range[0])
    leg_masses_upper_bound = np.array(leg_masses) * (
        1.0 + self._minitaur_leg_mass_err_range[1])
    randomized_leg_masses = [
        np.random.uniform(leg_masses_lower_bound[i], leg_masses_upper_bound[i])
        for i in xrange(len(leg_masses))
    ]
    minitaur.SetLegMasses(randomized_leg_masses)

    randomized_battery_voltage = random.uniform(BATTERY_VOLTAGE_RANGE[0],
                                                BATTERY_VOLTAGE_RANGE[1])
    minitaur.SetBatteryVoltage(randomized_battery_voltage)

    randomized_motor_damping = random.uniform(MOTOR_VISCOUS_DAMPING_RANGE[0],
                                              MOTOR_VISCOUS_DAMPING_RANGE[1])
    minitaur.SetMotorViscousDamping(randomized_motor_damping)

    randomized_foot_friction = random.uniform(MINITAUR_LEG_FRICTION[0],
                                              MINITAUR_LEG_FRICTION[1])
    minitaur.SetFootFriction(randomized_foot_friction)

"""A config file for parameters and their ranges in dynamics randomization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


PARAM_RANGE = {
    # The following ranges are in percentage. e.g. 0.8 means 80%.
    "mass": [0.8, 1.2],
    "inertia": [0.5, 1.5],
    "motor strength": [0.8, 1.2],
    # The following ranges are the physical values, in SI unit.
    "motor friction": [0, 0.05],  # Viscous damping (Nm s/rad).
    "control step": [0.003, 0.02],  # Time inteval (s).
    "latency": [0.0, 0.04],  # Time inteval (s).
    "lateral friction": [0.5, 1.25],  # Friction coefficient (dimensionless).
    "battery": [14.0, 16.8],  # Voltage (V).
    "joint friction": [0, 0.05],  # Coulomb friction torque (Nm).
}


def all_params():
  """Randomize all the physical parameters."""
  return PARAM_RANGE

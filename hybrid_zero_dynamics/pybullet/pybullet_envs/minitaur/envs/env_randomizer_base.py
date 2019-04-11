"""Abstract base class for environment randomizer."""

import abc


class EnvRandomizerBase(object):
  """Abstract base class for environment randomizer.

  Randomizes physical parameters of the objects in the simulation and adds
  perturbations to the stepping of the simulation.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def randomize_env(self, env):
    """Randomize the simulated_objects in the environment.

    Will be called at when env is reset. The physical parameters will be fixed
    for that episode and be randomized again in the next environment.reset().

    Args:
      env: The Minitaur gym environment to be randomized.
    """
    pass

  def randomize_step(self, env):
    """Randomize simulation steps.

    Will be called at every timestep. May add random forces/torques to Minitaur.

    Args:
      env: The Minitaur gym environment to be randomized.
    """
    pass

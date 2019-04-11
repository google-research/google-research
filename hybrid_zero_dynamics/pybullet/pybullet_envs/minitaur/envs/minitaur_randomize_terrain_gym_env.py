"""Gym environment of minitaur which randomize the terrain at each reset."""
import math
import os

import numpy as np
#from google3.pyglib import flags
#from google3.pyglib import gfile
from pybullet_envs.minitaur.envs import minitaur
from pybullet_envs.minitaur.envs import minitaur_gym_env

#flags.DEFINE_string(
#    'terrain_dir', '/cns/od-d/home/brain-minitaur/terrain_obj',
#    'The directory which contains terrain obj files to be used.')
#flags.DEFINE_string('storage_dir', '/tmp',
#                    'The full path to the temporary directory to be used.')
#FLAGS = flags.FLAGS


class MinitaurRandomizeTerrainGymEnv(minitaur_gym_env.MinitaurGymEnv):
  """The gym environment for the minitaur with randomized terrain.

  It simulates a minitaur (a quadruped robot) on a randomized terrain. The state
  space include the angles, velocities and torques for all the motors and the
  action space is the desired motor angle for each motor. The reward function is
  based on how far the minitaur walks in 1000 steps and penalizes the energy
  expenditure.

  """

  def reset(self):
    self._pybullet_client.resetSimulation()
    self._pybullet_client.setPhysicsEngineParameter(
        numSolverIterations=self._num_bullet_solver_iterations)
    self._pybullet_client.setTimeStep(self._time_step)
    terrain_visual_shape_id = -1
    terrain_mass = 0
    terrain_position = [0, 0, 0]
    terrain_orientation = [0, 0, 0, 1]
    terrain_file_name = self.load_random_terrain(FLAGS.terrain_dir)
    terrain_collision_shape_id = self._pybullet_client.createCollisionShape(
        shapeType=self._pybullet_client.GEOM_MESH,
        fileName=terrain_file_name,
        flags=1,
        meshScale=[0.5, 0.5, 0.5])
    self._pybullet_client.createMultiBody(terrain_mass,
                                          terrain_collision_shape_id,
                                          terrain_visual_shape_id,
                                          terrain_position,
                                          terrain_orientation)
    self._pybullet_client.setGravity(0, 0, -10)
    self.minitaur = (minitaur.Minitaur(
        pybullet_client=self._pybullet_client,
        urdf_root=self._urdf_root,
        time_step=self._time_step,
        self_collision_enabled=self._self_collision_enabled,
        motor_velocity_limit=self._motor_velocity_limit,
        pd_control_enabled=self._pd_control_enabled,
        on_rack=self._on_rack))
    self._last_base_position = [0, 0, 0]
    for _ in xrange(100):
      if self._pd_control_enabled:
        self.minitaur.ApplyAction([math.pi / 2] * 8)
      self._pybullet_client.stepSimulation()
    return self._get_observation()

  def load_random_terrain(self, terrain_dir):
    """Load a random terrain obj file.

    Args:
      terrain_dir: The directory which contains terrain obj files to be used.

    Returns:
      terrain_file_name_complete: The complete terrain obj file name in the
      local directory.
    """
    terrain_file_names_all = gfile.ListDir(terrain_dir)
    terrain_file_name = np.random.choice(terrain_file_names_all)
    asset_source = os.path.join(terrain_dir, terrain_file_name)
    asset_destination = os.path.join(FLAGS.storage_dir, terrain_file_name)
    gfile.Copy(asset_source, asset_destination, overwrite=True)
    terrain_file_name_complete = os.path.join(FLAGS.storage_dir,
                                              terrain_file_name)
    return terrain_file_name_complete

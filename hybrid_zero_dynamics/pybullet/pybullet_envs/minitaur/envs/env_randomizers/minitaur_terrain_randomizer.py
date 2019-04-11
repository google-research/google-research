"""Generates a random terrain at Minitaur gym environment reset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = os.path.dirname(os.path.dirname(parentdir))
os.sys.path.insert(0,parentdir)

import itertools
import math
import enum
import numpy as np

from pybullet_envs.minitaur.envs import env_randomizer_base

_GRID_LENGTH = 15
_GRID_WIDTH = 10
_MAX_SAMPLE_SIZE = 30
_MIN_BLOCK_DISTANCE = 0.7
_MAX_BLOCK_LENGTH = _MIN_BLOCK_DISTANCE
_MIN_BLOCK_LENGTH = _MAX_BLOCK_LENGTH / 2
_MAX_BLOCK_HEIGHT = 0.05
_MIN_BLOCK_HEIGHT = _MAX_BLOCK_HEIGHT / 2


class PoissonDisc2D(object):
  """Generates 2D points using Poisson disk sampling method.

  Implements the algorithm described in:
    http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
  Unlike the uniform sampling method that creates small clusters of points,
  Poisson disk method enforces the minimum distance between points and is more
  suitable for generating a spatial distribution of non-overlapping objects.
  """

  def __init__(self, grid_length, grid_width, min_radius, max_sample_size):
    """Initializes the algorithm.

    Args:
      grid_length: The length of the bounding square in which points are
        sampled.
      grid_width: The width of the bounding square in which points are
        sampled.
      min_radius: The minimum distance between any pair of points.
      max_sample_size: The maximum number of sample points around a active site.
        See details in the algorithm description.
    """
    self._cell_length = min_radius / math.sqrt(2)
    self._grid_length = grid_length
    self._grid_width = grid_width
    self._grid_size_x = int(grid_length / self._cell_length) + 1
    self._grid_size_y = int(grid_width / self._cell_length) + 1
    self._min_radius = min_radius
    self._max_sample_size = max_sample_size

    # Flattern the 2D grid as an 1D array. The grid is used for fast nearest
    # point searching.
    self._grid = [None] * self._grid_size_x * self._grid_size_y

    # Generate the first sample point and set it as an active site.
    first_sample = np.array(
        np.random.random_sample(2)) * [grid_length, grid_width]
    self._active_list = [first_sample]

    # Also store the sample point in the grid.
    self._grid[self._point_to_index_1d(first_sample)] = first_sample

  def _point_to_index_1d(self, point):
    """Computes the index of a point in the grid array.

    Args:
      point: A 2D point described by its coordinates (x, y).

    Returns:
      The index of the point within the self._grid array.
    """
    return self._index_2d_to_1d(self._point_to_index_2d(point))

  def _point_to_index_2d(self, point):
    """Computes the 2D index (aka cell ID) of a point in the grid.

    Args:
      point: A 2D point (list) described by its coordinates (x, y).

    Returns:
      x_index: The x index of the cell the point belongs to.
      y_index: The y index of the cell the point belongs to.
    """
    x_index = int(point[0] / self._cell_length)
    y_index = int(point[1] / self._cell_length)
    return x_index, y_index

  def _index_2d_to_1d(self, index2d):
    """Converts the 2D index to the 1D position in the grid array.

    Args:
      index2d: The 2D index of a point (aka the cell ID) in the grid.

    Returns:
      The 1D position of the cell within the self._grid array.
    """
    return index2d[0] + index2d[1] * self._grid_size_x

  def _is_in_grid(self, point):
    """Checks if the point is inside the grid boundary.

    Args:
      point: A 2D point (list) described by its coordinates (x, y).

    Returns:
      Whether the point is inside the grid.
    """
    return (0 <= point[0] < self._grid_length) and (0 <= point[1] <
                                                    self._grid_width)

  def _is_in_range(self, index2d):
    """Checks if the cell ID is within the grid.

    Args:
      index2d: The 2D index of a point (aka the cell ID) in the grid.

    Returns:
      Whether the cell (2D index) is inside the grid.
    """

    return (0 <= index2d[0] < self._grid_size_x) and (0 <= index2d[1] <
                                                      self._grid_size_y)

  def _is_close_to_existing_points(self, point):
    """Checks if the point is close to any already sampled (and stored) points.

    Args:
      point: A 2D point (list) described by its coordinates (x, y).

    Returns:
      True iff the distance of the point to any existing points is smaller than
      the min_radius
    """
    px, py = self._point_to_index_2d(point)
    # Now we can check nearby cells for existing points
    for neighbor_cell in itertools.product(
        xrange(px - 1, px + 2), xrange(py - 1, py + 2)):

      if not self._is_in_range(neighbor_cell):
        continue

      maybe_a_point = self._grid[self._index_2d_to_1d(neighbor_cell)]
      if maybe_a_point is not None and np.linalg.norm(
          maybe_a_point - point) < self._min_radius:
        return True

    return False

  def sample(self):
    """Samples new points around some existing point.

    Removes the sampling base point and also stores the new jksampled points if
    they are far enough from all existing points.
    """
    active_point = self._active_list.pop()
    for _ in xrange(self._max_sample_size):
      # Generate random points near the current active_point between the radius
      random_radius = np.random.uniform(self._min_radius, 2 * self._min_radius)
      random_angle = np.random.uniform(0, 2 * math.pi)

      # The sampled 2D points near the active point
      sample = random_radius * np.array(
          [np.cos(random_angle), np.sin(random_angle)]) + active_point

      if not self._is_in_grid(sample):
        continue

      if self._is_close_to_existing_points(sample):
        continue

      self._active_list.append(sample)
      self._grid[self._point_to_index_1d(sample)] = sample

  def generate(self):
    """Generates the Poisson disc distribution of 2D points.

    Although the while loop looks scary, the algorithm is in fact O(N), where N
    is the number of cells within the grid. When we sample around a base point
    (in some base cell), new points will not be pushed into the base cell
    because of the minimum distance constraint. Once the current base point is
    removed, all future searches cannot start from within the same base cell.

    Returns:
      All sampled points. The points are inside the quare [0, grid_length] x [0,
      grid_width]
    """

    while self._active_list:
      self.sample()

    all_sites = []
    for p in self._grid:
      if p is not None:
        all_sites.append(p)

    return all_sites


class TerrainType(enum.Enum):
  """The randomzied terrain types we can use in the gym env."""
  RANDOM_BLOCKS = 1
  TRIANGLE_MESH = 2


class MinitaurTerrainRandomizer(env_randomizer_base.EnvRandomizerBase):
  """Generates an uneven terrain in the gym env."""

  def __init__(
      self,
      terrain_type=TerrainType.TRIANGLE_MESH,
      mesh_filename="robotics/reinforcement_learning/minitaur/envs/testdata/"
      "triangle_mesh_terrain/terrain9735.obj",
      mesh_scale=None):
    """Initializes the randomizer.

    Args:
      terrain_type: Whether to generate random blocks or load a triangle mesh.
      mesh_filename: The mesh file to be used. The mesh will only be loaded if
        terrain_type is set to TerrainType.TRIANGLE_MESH.
      mesh_scale: the scaling factor for the triangles in the mesh file.
    """
    self._terrain_type = terrain_type
    self._mesh_filename = mesh_filename
    self._mesh_scale = mesh_scale if mesh_scale else [1.0, 1.0, 0.3]

  def randomize_env(self, env):
    """Generate a random terrain for the current env.

    Args:
      env: A minitaur gym environment.
    """

    if self._terrain_type is TerrainType.TRIANGLE_MESH:
      self._load_triangle_mesh(env)
    if self._terrain_type is TerrainType.RANDOM_BLOCKS:
      self._generate_convex_blocks(env)

  def _load_triangle_mesh(self, env):
    """Represents the random terrain using a triangle mesh.

    It is possible for Minitaur leg to stuck at the common edge of two triangle
    pieces. To prevent this from happening, we recommend using hard contacts
    (or high stiffness values) for Minitaur foot in sim.

    Args:
      env: A minitaur gym environment.
    """
    env.pybullet_client.removeBody(env.ground_id)
    terrain_collision_shape_id = env.pybullet_client.createCollisionShape(
        shapeType=env.pybullet_client.GEOM_MESH,
        fileName=self._mesh_filename,
        flags=1,
        meshScale=self._mesh_scale)
    env.ground_id = env.pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=terrain_collision_shape_id,
        basePosition=[0, 0, 0])

  def _generate_convex_blocks(self, env):
    """Adds random convex blocks to the flat ground.

    We use the Possion disk algorithm to add some random blocks on the ground.
    Possion disk algorithm sets the minimum distance between two sampling
    points, thus voiding the clustering effect in uniform N-D distribution.

    Args:
      env: A minitaur gym environment.

    """

    poisson_disc = PoissonDisc2D(_GRID_LENGTH, _GRID_WIDTH, _MIN_BLOCK_DISTANCE,
                                 _MAX_SAMPLE_SIZE)

    block_centers = poisson_disc.generate()

    for center in block_centers:
      # We want the blocks to be in front of the robot.
      shifted_center = np.array(center) - [2, _GRID_WIDTH / 2]

      # Do not place blocks near the point [0, 0], where the robot will start.
      if abs(shifted_center[0]) < 1.0 and abs(shifted_center[1]) < 1.0:
        continue
      half_length = np.random.uniform(_MIN_BLOCK_LENGTH, _MAX_BLOCK_LENGTH) / (
          2 * math.sqrt(2))
      half_height = np.random.uniform(_MIN_BLOCK_HEIGHT, _MAX_BLOCK_HEIGHT) / 2
      box_id = env.pybullet_client.createCollisionShape(
          env.pybullet_client.GEOM_BOX,
          halfExtents=[half_length, half_length, half_height])
      env.pybullet_client.createMultiBody(
          baseMass=0,
          baseCollisionShapeIndex=box_id,
          basePosition=[shifted_center[0], shifted_center[1], half_height])

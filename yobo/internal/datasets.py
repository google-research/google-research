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

# pylint: skip-file
"""Different datasets implementation plus a general port for all the datasets."""

import abc
import copy
import functools
import json
import os
from os import path
import queue
import threading
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import chex
import cv2
from etils import epath
import gin
import jax
import mediapy as media
import numpy as np
from PIL import Image
import pycolmap
import tensorflow as tf
import trimesh

gfile = tf.io.gfile

from google_research.yobo.internal import camera_utils
from google_research.yobo.internal import configs
from google_research.yobo.internal import image_io
from google_research.yobo.internal import image_utils
from google_research.yobo.internal import raw_utils
from google_research.yobo.internal import utils
from google_research.yobo.internal.dataset import mask_utils

from multiprocessing import Pool


def ParallelMap(f, l):
  with Pool() as p:
    return p.map(f, l)


gin.config.external_configurable(
    camera_utils.transform_poses_pca, module='camera_utils'
)
gin.config.external_configurable(
    camera_utils.transform_poses_focus, module='camera_utils'
)

# Local directory to copy pose data into so it can be read more quickly.
LOCAL_COLMAP_DIR = '/tmp/colmap/'
LOCAL_MESHDIR = '/tmp/'


def load_dataset(split, train_dir, config):
  """Loads a split of a dataset using the data_loader specified by `config`."""
  dataset_dict = {
      'blender': Blender,
      'llff': LLFF,
      'tat_nerfpp': TanksAndTemplesNerfPP,
      'tat_fvs': TanksAndTemplesFVS,
      'dtu': DTU,
      'pixelrig': PixelRig,
      'rtmv': RTMV,
      'aerial': Aerial,
      'maya': Maya,
  }
  return dataset_dict[config.dataset_loader](split, train_dir, config)


def convert_colmap_cam(cam):
  """Converts COLMAP camera parameters into our format."""

  fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
  pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))

  type_ = cam.camera_type

  if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
    params = None
    camtype = camera_utils.ProjectionType.PERSPECTIVE

  elif type_ == 1 or type_ == 'PINHOLE':
    params = None
    camtype = camera_utils.ProjectionType.PERSPECTIVE

  if type_ == 2 or type_ == 'SIMPLE_RADIAL':
    params = {k: 0.0 for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
    params['k1'] = cam.k1
    camtype = camera_utils.ProjectionType.PERSPECTIVE

  elif type_ == 3 or type_ == 'RADIAL':
    params = {k: 0.0 for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
    params['k1'] = cam.k1
    params['k2'] = cam.k2
    camtype = camera_utils.ProjectionType.PERSPECTIVE

  elif type_ == 4 or type_ == 'OPENCV':
    params = {k: 0.0 for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
    params['k1'] = cam.k1
    params['k2'] = cam.k2
    params['p1'] = cam.p1
    params['p2'] = cam.p2
    camtype = camera_utils.ProjectionType.PERSPECTIVE

  elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
    params = {k: 0.0 for k in ['k1', 'k2', 'k3', 'k4']}
    params['k1'] = cam.k1
    params['k2'] = cam.k2
    params['k3'] = cam.k3
    params['k4'] = cam.k4
    camtype = camera_utils.ProjectionType.FISHEYE

  return pixtocam, params, camtype


class NeRFSceneManager(pycolmap.SceneManager):
  """COLMAP pose loader."""

  def process(
      self,
      load_points: bool = False,
  ) -> Tuple[
      Sequence[str],
      np.ndarray,
      np.ndarray,
      Optional[Mapping[str, float]],
      camera_utils.ProjectionType,
  ]:
    """Applies NeRF-specific postprocessing to the loaded pose data.

    Args:
      load_points: If True, load the colmap points.

    Returns:
      a tuple [image_names, poses, pixtocam, distortion_params].
      image_names:  contains the only the basename of the images.
      poses: [N, 4, 4] array containing the camera to world matrices.
      pixtocam: [N, 3, 3] array containing the camera to pixel space matrices.
      distortion_params: mapping of distortion param name to distortion
        parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
    """

    self.load_cameras()
    self.load_images()
    if load_points:
      self.load_points3D()

    camdata = self.cameras
    imdata = self.images
    w2c_mats = []
    p2c_mats = []
    distortion_params = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
      im = imdata[k]
      rot = im.R()
      trans = im.tvec.reshape(3, 1)
      w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
      w2c_mats.append(w2c)
      pixtocam, params, camtype = convert_colmap_cam(camdata[im.camera_id])
      p2c_mats.append(pixtocam)
      distortion_params.append(params)
    w2c_mats = np.stack(w2c_mats, axis=0)
    pixtocams = np.stack(p2c_mats, axis=0)
    distortion_params = jax.tree.map(
        lambda *args: np.array(args), *distortion_params
    )

    # Convert extrinsics to camera-to-world.
    c2w_mats = np.linalg.inv(w2c_mats)
    poses = c2w_mats[:, :3, :4]

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    names = [imdata[k].name for k in imdata]

    # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
    poses = poses @ np.diag([1, -1, -1, 1])

    return names, poses, pixtocams, distortion_params, camtype


def find_colmap_data(data_dir, colmap_subdir=None):
  """Locate COLMAP pose data."""
  if colmap_subdir is None:
    search_paths = ['sparse/0/', 'sparse/', 'colmap/sparse/0/']
  else:
    search_paths = [colmap_subdir]
  for search_path in search_paths:
    d = os.path.join(data_dir, search_path)
    if utils.file_exists(d):
      return d
  raise ValueError(f'{data_dir} has no COLMAP data folder.')


def flatten_data(images):
  """Flattens list of variable-resolution images into an array of pixels."""

  def flatten_and_concat(values, n):
    return np.concatenate([np.array(z).reshape(-1, n) for z in values])

  def index_array(i, w, h):
    x, y = camera_utils.pixel_coordinates(w, h)
    i = np.full((h, w), i)
    return np.stack([i, x, y], axis=-1)

  height = np.array([z.shape[0] for z in images])
  width = np.array([z.shape[1] for z in images])
  indices = [
      index_array(i, w, h) for i, (w, h) in enumerate(zip(width, height))
  ]
  indices = flatten_and_concat(indices, 3)
  pixels = flatten_and_concat(images, 3)
  return pixels, indices


def _compute_near_far_planes_from_config(
    config: configs.Config, scene_metadata: Optional[dict[str, Any]]
) -> tuple[float, float]:
  """Computes near and far planes based on the config settings."""
  near = config.near
  far = config.far
  if (
      config.near_plane_meters is not None
      or config.far_plane_meters is not None
  ):
    assert (
        scene_metadata is not None and 'meters_per_colmap' in scene_metadata
    ), (
        'When using near_plane_meters or far_plane_meters, calibration images'
        ' are required to be present in the dataset.'
    )
    colmap_units_per_meter = 1.0 / scene_metadata['meters_per_colmap']
    if config.near_plane_meters is not None:
      near = config.near_plane_meters * colmap_units_per_meter
      logging.info(
          'Setting near plane from meters: %f (colmap units/m: %f)',
          near,
          colmap_units_per_meter,
      )
    if config.far_plane_meters is not None:
      far = config.far_plane_meters * colmap_units_per_meter
      logging.info(
          'Setting far plane from meters: %f (colmap units/m: %f)',
          far,
          colmap_units_per_meter,
      )
  return near, far


def load_llff_posedata(data_dir):
  """Load poses from a `poses_bounds.npy` file as specified by LLFF."""
  # Load pre-computed poses_bounds.npy in the format described in
  # https://github.com/Fyusion/LLFF. For example, this can be generated with
  # vision::sfm based pose estimation from the Insitu pipeline.
  posefile = os.path.join(data_dir, 'poses_bounds.npy')
  if not utils.file_exists(posefile):
    raise ValueError(f'poses_bounds.npy does not exist in {data_dir}.')

  with utils.open_file(posefile, 'rb') as fp:
    poses_arr = np.load(fp)
  bounds = poses_arr[:, -2:]

  # "hwf" stands for (height, width, focal).
  poses_hwf = poses_arr[:, :-2].reshape([-1, 3, 5])
  poses_llff = poses_hwf[:, :, :4]
  # Convert from [down, right, backwards] to [right, up, backwards] coordinates.
  nerf_to_llff = np.array([
      [0, -1, 0, 0],
      [1, 0, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
  ])
  poses = poses_llff @ nerf_to_llff
  h, w, f = poses_hwf[0, :, 4]
  pixtocams = camera_utils.get_pixtocam(f, w, h)
  distortion_params = None
  camtype = camera_utils.ProjectionType.PERSPECTIVE
  return poses, pixtocams, distortion_params, camtype, bounds


def load_ngp_posedata(config, data_dir, pose_file_name='transforms.json'):
  """Load poses from a `transforms.json` file as used in Blender/Instant NGP."""
  pose_file = os.path.join(data_dir, pose_file_name)
  with utils.open_file(pose_file, 'r') as fp:
    meta = json.load(fp)

  if config.num_dataset_images > 0:
    meta['frames'] = meta['frames'][: config.num_dataset_images]

  w = meta['w'] if 'w' in meta else None
  h = meta['h'] if 'h' in meta else None

  def extract_intrinsics(frame, w, h):
    focal_keys = [
        'fl_x',
        'fl_y',
        'camera_angle_x',
        'camera_angle_y',
        'focal_in_mm',
    ]
    if not any([k in frame for k in focal_keys]):
      return None
    # Extract principal point.
    cx = frame['cx'] if 'cx' in frame else w / 2.0
    cy = frame['cy'] if 'cy' in frame else h / 2.0
    # Extract focal lengths, use field of view if focal not directly saved.
    if 'focal_in_mm' in frame:
      fx = w * frame['focal_in_mm'] / frame['sensor_size_horizontal_in_mm']
    elif 'fl_x' in frame:
      fx = frame['fl_x']
    else:
      fx = 0.5 * w / np.tan(0.5 * float(frame['camera_angle_x']))

    if 'fl_y' in frame:
      fy = frame['fl_y']
    elif 'camera_angle_y' in frame:
      fy = 0.5 * h / np.tan(0.5 * float(frame['camera_angle_y']))
    else:
      fy = fx
    # Create inverse intrinsics matrix.
    return np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))

  def extract_distortion(frame):
    # Extract the distortion coefficients if they are available.
    coeffs = ['k1', 'k2', 'p1', 'p2']
    if not any([c in frame for c in coeffs]):
      return None
    else:
      return {c: frame[c] if c in frame else 0.0 for c in coeffs}

  data_dir = gpath.GPath(data_dir)
  base_dir = (data_dir / meta['frames'][0]['file_path']).parent

  def find_file(frame):
    filepath = data_dir / frame['file_path']
    files = utils.listdir(filepath.parent)
    # Some NGP exporters do not include the image type extension, so search for
    # a few common ones.
    exts = ['.png', '.jpg', '.exr']
    # Try no extension, all lowercase, all uppercase.
    ext_list = [''] + [s.lower() for s in exts] + [s.upper() for s in exts]
    for ext in ext_list:
      filepath_try = filepath.stem + ext
      if filepath_try in files:
        return ext
    return None

  exts = ParallelMap(find_file, meta['frames'])

  names = []
  nameprefixes = []
  camtoworlds = []
  pixtocams = []
  distortion_params = []
  for ext, frame in zip(exts, meta['frames']):
    if ext is None:
      continue
    filepath = data_dir / frame['file_path']
    filename = (filepath.parent / (filepath.stem + ext)).name
    nameprefixes.append(frame['file_path'])
    names.append(filename)
    camtoworlds.append(np.array(frame['transform_matrix']))
    if w is None or h is None:
      # Blender JSON files may not have `w` and `h`, need to take from image.
      f = os.path.join(base_dir, filename)
      is_exr = f.lower().endswith('.exr')
      load_fn = image_io.load_exr if is_exr else image_io.load_img
      h, w = load_fn(f).shape[:2]
    pixtocams.append(extract_intrinsics(frame, w, h))
    distortion_params.append(extract_distortion(frame))
  camtoworlds = np.stack(camtoworlds, axis=0).astype(np.float32)

  # If intrinsics or distortion not stored per-image, use global parameters.
  if pixtocams[0] is None:
    pixtocams = extract_intrinsics(meta, w, h)
  else:
    pixtocams = np.stack(pixtocams, axis=0)

  if distortion_params[0] is None:
    distortion_params = extract_distortion(meta)
  else:
    distortion_params = jax.tree.map(
        lambda *args: np.array(args), *distortion_params
    )

  if 'camera_type' in meta:
    if 'fisheye' in meta['camera_type']:
      camtype = camera_utils.ProjectionType.FISHEYE_EQUISOLID
    else:
      camtype = camera_utils.ProjectionType(meta['camera_type'])
  else:
    camtype = camera_utils.ProjectionType.PERSPECTIVE

  return names, camtoworlds, pixtocams, distortion_params, camtype, nameprefixes


def load_arcore_posedata(data_dir, arcore_metadata_file_name):
  """Load poses from a Lens Spatial ARCore data JSON file."""
  # Filename usually either 'metadata.json' or 'original_metadata.json'.

  arcore_metadata_file_path = os.path.join(data_dir, arcore_metadata_file_name)
  with utils.open_file(arcore_metadata_file_path) as i:
    arcore_metadata = json.load(i)

  fx, fy = arcore_metadata['intrinsics']['focal_length']
  cx, cy = arcore_metadata['intrinsics']['principal_point']
  # Swap these due to ARCore landscape/portrait eccentricities!
  # All data is saved in portrait but ARCore stores these params in landscape.
  cx, cy = cy, cx
  fx, fy = fy, fx
  camtopix = camera_utils.intrinsic_matrix(fx, fy, cx, cy)
  pixtocam = np.linalg.inv(camtopix)
  distortion_params = None

  image_names = []
  c2w_poses = []
  for image_data in arcore_metadata['images']:
    image_name = image_data['path']
    # Conversion from column-major order.
    pose = np.asarray(image_data['matrix']).reshape((4, 4)).T
    pose = pose[:3, :4]
    c2w_poses.append(pose)
    image_names.append(image_name)

  c2w_poses = np.array(c2w_poses)
  camtype = camera_utils.ProjectionType.PERSPECTIVE
  return image_names, c2w_poses, pixtocam, distortion_params, camtype


class Dataset(metaclass=abc.ABCMeta):
  """Dataset Base Class.

  Base class for a NeRF dataset. Can create batches of ray and color data used
  for training or rendering a NeRF model.

  Each subclass is responsible for loading images and camera poses from disk by
  implementing the _load_renderings() method. This data is used to generate
  train and test batches of ray + color data for feeding through the NeRF model.
  The ray parameters are calculated in _generate_rays().

  An asynchronous batch queue iterator can be created for a Dataset using the
  RayBatcher class found below.

  Attributes:
    alphas: np.ndarray, optional array of alpha channel data.
    cameras: tuple summarizing all camera extrinsic/intrinsic/distortion params.
    jax_cameras: cameras in the JAX camera class format.
    camtoworlds: np.ndarray, a list of extrinsic camera pose matrices.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.
    data_dir: str, location of the dataset on disk.
    disp_images: np.ndarray, optional array of disparity (inverse depth) data.
    distortion_params: dict, the camera distortion model parameters.
    exposures: optional per-image exposure value (shutter * ISO / 1000).
    max_exposure: Maximum of exposures in all images (test and train)
    far: float, far plane value for rays.
    focal: float, focal length from camera intrinsics.
    height: int, height of images.
    images: np.ndarray, array of RGB image data.
    mask_images: np.ndarray, array of image masks (masks are 3 channels).
    metadata: dict, optional metadata for raw datasets.
    lossmult: np.ndarray, per-image weights to apply in loss calculation.
    near: float, near plane value for rays.
    normal_images: np.ndarray, optional array of surface normal vector data.
    pixtocams: np.ndarray, one or a list of inverse intrinsic camera matrices.
    pixtocam_ndc: np.ndarray, the inverse intrinsic matrix used for NDC space.
    poses: np.ndarray, optional array of auxiliary camera pose data.
    rays: utils.Rays, ray data for every pixel in the dataset.
    render_exposures: optional list of exposure values for the render path.
    render_path: bool, indicates if a smooth camera path should be generated.
    semantic_images: np.ndarray, optional array of semantic data.
    size: int, number of images in the dataset.
    split: str, indicates if this is a "train" or "test" dataset.
    width: int, width of images.
    scene_metadata: dict, optional metadata computed for scene.
    jax_camera_from_tuple_fn: A function that converts camera tuples to JAX
      cameras.
    scene_bbox: optional scene bounding box.
  """

  def __init__(
      self, split: str, data_dir: str, config: configs.Config, **kwargs
  ):
    super().__init__()

    # Initialize attributes
    self._patch_size = np.maximum(config.patch_size, 1)
    self._batch_size = config.batch_size // jax.process_count()
    if self._patch_size**2 > self._batch_size:
      raise ValueError(
          f'Patch size {self._patch_size}^2 too large for '
          + f'per-process batch size {self._batch_size}'
      )
    self._batching = utils.BatchingMethod(config.batching)
    self._use_tiffs = config.use_tiffs
    self._use_exrs = config.use_exrs
    self._load_disps = config.compute_disp_metrics
    self._load_normals = config.compute_normal_metrics
    self._semantic_dir = config.semantic_dir
    self._num_border_pixels_to_mask = config.num_border_pixels_to_mask
    self._apply_bayer_mask = config.apply_bayer_mask
    self._cast_rays_in_train_step = config.cast_rays_in_train_step
    self._cast_rays_in_eval_step = config.cast_rays_in_eval_step
    self._flattened = False

    self.split = utils.DataSplit(split)
    self.data_dir = data_dir
    self.near = config.near
    self.far = config.far
    self.scene_bbox = config.scene_bbox
    self.render_path = config.render_path
    self.distortion_params = None
    self.disp_images = None
    self.normal_images = None
    self.semantic_images = None
    self.alphas = None
    self.mask_images = None
    self.poses = None
    self.pixtocam_ndc = None
    self.metadata = None
    self.camtype = camera_utils.ProjectionType.PERSPECTIVE
    self.exposures = None
    self.max_exposure = None
    self.render_exposures = None
    self.lossmult = None
    self.scene_metadata = None
    self.mesh = None

    if isinstance(config.scene_bbox, float):
      b = config.scene_bbox
      self.scene_bbox = np.array(((-b,) * 3, (b,) * 3))
    elif config.scene_bbox is not None:
      self.scene_bbox = np.array(config.scene_bbox)
    else:
      self.scene_bbox = None

    # Providing type comments for these attributes, they must be correctly
    # initialized by _load_renderings() (see docstring) in any subclass.
    self.images: Union[np.ndarray, List[np.ndarray]] = None
    self.camtoworlds: np.ndarray = None
    self.pixtocams: np.ndarray = None
    self.height: int = None
    self.width: int = None
    self.focal: float = None

    # Load data from disk using provided config parameters.
    self._load_renderings(config, **kwargs)

    self.near, self.far = _compute_near_far_planes_from_config(
        config, self.scene_metadata
    )

    if self.poses is None:
      self.poses = self.camtoworlds

    if self.focal is None:
      # Take focal length (fx) from first camera as default for visualization.
      self.focal = 1.0 / float(self.pixtocams.ravel()[0])

    if self.render_path:
      if config.render_path_file is not None:
        render_path_file = config.render_path_file
        if not os.path.isabs(render_path_file):
          render_path_file = os.path.join(self.data_dir, render_path_file)
        with utils.open_file(render_path_file, 'rb') as fp:
          render_poses = np.load(fp)
        self.camtoworlds = render_poses
      if config.render_resolution is not None:
        if config.render_focal is None:
          # If no focal specified, preserve vertical field of view.
          new_height = config.render_resolution[1]
          config.render_focal = new_height / self.height * self.focal
        self.width, self.height = config.render_resolution
      if config.render_focal is not None:
        self.focal = config.render_focal
      if config.render_camtype is not None:
        self.camtype = camera_utils.ProjectionType(config.render_camtype)

      self.distortion_params = None
      if self.camtype == camera_utils.ProjectionType.PANORAMIC:
        self.pixtocams = np.diag(
            [2.0 * np.pi / self.width, np.pi / self.height, 1.0]
        )
      else:
        self.pixtocams = camera_utils.get_pixtocam(  # pytype: disable=annotation-type-mismatch  # jax-ndarray
            self.focal, self.width, self.height
        )

    self._n_examples = self.camtoworlds.shape[0]

    z_range = None
    if config.z_min is not None and config.z_max is not None:
      z_range = (config.z_min, config.z_max)

    # Broadcast pixtocams if there is only one provided.
    if self.pixtocams.ndim < self.camtoworlds.ndim:
      self.pixtocams = np.broadcast_to(
          self.pixtocams[None], (self.camtoworlds.shape[0], 3, 3)
      )

    self.cameras = (
        self.pixtocams,
        self.camtoworlds,
        self.distortion_params,
        self.pixtocam_ndc,
        z_range,
    )

    # Cache the partial conversion function.
    self.jax_camera_from_tuple_fn = functools.partial(
        camera_utils.jax_camera_from_tuple,
        projection_type=self.camtype,
    )

    # Don't generate jax_cameras when the render path is set, since we don't
    # need them anyway and the hijacking logic makes it difficult.
    if not self.render_path:
      image_sizes = np.array([(x.shape[1], x.shape[0]) for x in self.images])
      self.jax_cameras = jax.vmap(self.jax_camera_from_tuple_fn)(
          self.cameras, image_sizes
      )

  @property
  def size(self):
    return self._n_examples

  # Would be nice to use `@functools.cached_property` is it was supported by
  # CiderV language service
  @property
  def data_path(self) -> epath.Path:
    """pathlib-like version of `data_dir`."""
    return epath.Path(self.data_dir)

  @abc.abstractmethod
  def _load_renderings(self, config, **kwargs):
    # pyformat: disable
    """Load images and poses from disk.

    Args:
      config: utils.Config, user-specified config parameters. In inherited
        classes, this method must set the following public attributes:
        - images: [N, height, width, 3] array for RGB images.
        - disp_images: [N, height, width] array for depth data (optional).
        - normal_images: [N, height, width, 3] array for normals (optional).
        - camtoworlds: [N, 3, 4] array of extrinsic pose matrices.
        - poses: [..., 3, 4] array of auxiliary pose data (optional).
        - pixtocams: [N, 3, 4] array of inverse intrinsic matrices.
        - distortion_params: dict, camera lens distortion model parameters.
        - height: int, height of images.
        - width: int, width of images.
        - focal: float, focal length to use for ideal pinhole rendering.
      **kwargs: forwarded kwargs from Dataset constructor.
    """
    # pyformat: enable

  def _make_ray_batch(
      self,
      pix_x_int: np.ndarray,
      pix_y_int: np.ndarray,
      cam_idx: Union[np.ndarray, np.int32],
      lossmult: Optional[np.ndarray] = None,
      rgb: Optional[np.ndarray] = None,
  ) -> utils.Batch:
    """Creates ray data batch from pixel coordinates and camera indices.

    All arguments must have broadcastable shapes. If the arguments together
    broadcast to a shape [a, b, c, ..., z] then the returned utils.Rays object
    will have array attributes with shape [a, b, c, ..., z, N], where N=3 for
    3D vectors and N=1 for per-ray scalar attributes.

    Args:
      pix_x_int: int array, x coordinates of image pixels.
      pix_y_int: int array, y coordinates of image pixels.
      cam_idx: int or int array, camera indices.
      lossmult: float array, weight to apply to each ray when computing loss fn.
      rgb: float array, optional RGB values to use in batch.

    Returns:
      A dict mapping from strings utils.Rays or arrays of image data.
      This is the batch provided for one NeRF train or test iteration.
    """

    broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
    ray_kwargs = {
        'lossmult': broadcast_scalar(1.0) if lossmult is None else lossmult,
        'near': broadcast_scalar(self.near),
        'far': broadcast_scalar(self.far),
        'cam_idx': broadcast_scalar(cam_idx),
    }
    # Collect per-camera information needed for each ray.
    if self.metadata is not None:
      # Exposure index and relative shutter speed, needed for RawNeRF.
      for key in ['exposure_idx', 'exposure_values']:
        idx = 0 if self.render_path else cam_idx
        ray_kwargs[key] = broadcast_scalar(self.metadata[key][idx])
    if self.exposures is not None:
      idx = 0 if self.render_path else cam_idx
      ray_kwargs['exposure_values'] = broadcast_scalar(self.exposures[idx])
    if self.render_path and self.render_exposures is not None:
      ray_kwargs['exposure_values'] = broadcast_scalar(
          self.render_exposures[cam_idx]
      )

    pixels = utils.Pixels(pix_x_int, pix_y_int, **ray_kwargs)
    if (
        self._cast_rays_in_train_step and self.split == utils.DataSplit.TRAIN
    ) or (self._cast_rays_in_eval_step and self.split == utils.DataSplit.TEST):
      # Fast path, defer ray computation to the training loop (on device).
      rays = pixels
    else:
      # Slow path, do ray computation using numpy (on CPU).
      rays = camera_utils.cast_ray_batch(  # pytype: disable=wrong-arg-types  # jax-ndarray
          self.cameras, pixels, self.camtype, self.scene_bbox, xnp=np
      )

    # Create data batch.
    batch = {}
    batch['rays'] = rays
    if not self.render_path:
      if rgb is not None:
        batch['rgb'] = rgb
      else:
        batch['rgb'] = self.images[cam_idx, pix_y_int, pix_x_int]
      if self._semantic_dir and self.semantic_images is not None:
        _, extra_h, extra_w, _ = self.semantic_images.shape
        if self._flattened:
          np_heights = np.asarray(self.heights)[cam_idx]
          np_widths = np.asarray(self.widths)[cam_idx]
          extra_y_int = (
              np.round((pix_y_int + 1) / np_heights * extra_h).astype(int) - 1
          )
          extra_x_int = (
              np.round((pix_x_int + 1) / np_widths * extra_w).astype(int) - 1
          )
        else:
          img_h, img_w = self.images[0].shape[:2]
          extra_y_int = (
              np.round((pix_y_int + 1) / img_h * extra_h).astype(int) - 1
          )
          extra_x_int = (
              np.round((pix_x_int + 1) / img_w * extra_w).astype(int) - 1
          )
        batch['semantic'] = self.semantic_images[
            cam_idx, extra_y_int, extra_x_int
        ]
    if self._load_disps and self.disp_images is not None:
      batch['disps'] = self.disp_images[cam_idx, pix_y_int, pix_x_int]
    if (
        self._load_normals
        and self.normal_images is not None
        and self.alphas is not None
    ):
      batch['normals'] = self.normal_images[cam_idx, pix_y_int, pix_x_int]
      batch['alphas'] = self.alphas[cam_idx, pix_y_int, pix_x_int]
    if self.mask_images is not None:
      batch['masks'] = self.mask_images[cam_idx, pix_y_int, pix_x_int]
    return utils.Batch(**batch)

  def _next_train(self) -> utils.Batch:
    """Sample next training batch (random rays)."""
    if self._flattened:
      # In the case where all images have been flattened into an array of pixels
      # take a random sample from this entire array.
      n_pixels = self.indices_flattened.shape[0]
      metaindices = np.random.randint(0, n_pixels, (self._batch_size,))
      indices_flattened = self.indices_flattened[metaindices]
      cam_idx = indices_flattened[..., 0]
      pix_x_int = indices_flattened[..., 1]
      pix_y_int = indices_flattened[..., 2]
      rgb = self.images_flattened[metaindices]

    else:
      # We assume all images in the dataset are the same resolution, so we use
      # the same width/height for sampling all pixels coordinates in the batch.
      # Batch/patch sampling parameters.
      num_patches = self._batch_size // self._patch_size**2
      lower_border = self._num_border_pixels_to_mask
      upper_border = self._num_border_pixels_to_mask + self._patch_size - 1
      # Random pixel patch x-coordinates.
      pix_x_int = np.random.randint(
          lower_border, self.width - upper_border, (num_patches, 1, 1)
      )
      # Random pixel patch y-coordinates.
      pix_y_int = np.random.randint(
          lower_border, self.height - upper_border, (num_patches, 1, 1)
      )
      # Add patch coordinate offsets.
      # Shape will broadcast to (num_patches, _patch_size, _patch_size).
      patch_dx_int, patch_dy_int = camera_utils.pixel_coordinates(
          self._patch_size, self._patch_size
      )
      pix_x_int = pix_x_int + patch_dx_int
      pix_y_int = pix_y_int + patch_dy_int
      # Random camera indices.
      if self._batching == utils.BatchingMethod.ALL_IMAGES:
        cam_idx = np.random.randint(0, self._n_examples, (num_patches, 1, 1))
      else:
        cam_idx = np.random.randint(0, self._n_examples, (1,))
      rgb = None

    if self._apply_bayer_mask:
      # Compute the Bayer mosaic mask for each pixel in the batch.
      lossmult = raw_utils.pixels_to_bayer_mask(pix_x_int, pix_y_int)
    elif self.lossmult is not None:
      lossmult = self.lossmult[cam_idx].reshape(-1, 1)
    else:
      lossmult = None

    return self._make_ray_batch(
        pix_x_int, pix_y_int, cam_idx, lossmult=lossmult, rgb=rgb
    )

  def generate_flattened_ray_batch(
      self, cam_idx, n_samples=10000
  ) -> utils.Batch:
    """Generate flattened ray batch for a specified camera in the dataset."""
    images_flattened, indices_flattened = flatten_data(
        self.images[cam_idx][None]
    )
    n_pixels = images_flattened.shape[0]
    mask_indices = np.random.randint(0, n_pixels, (n_samples,))
    cam_idx = indices_flattened[..., 0][mask_indices]
    pix_x_int = indices_flattened[..., 1][mask_indices]
    pix_y_int = indices_flattened[..., 2][mask_indices]
    rgb = images_flattened[mask_indices]

    return self._make_ray_batch(
        pix_x_int, pix_y_int, cam_idx, lossmult=None, rgb=rgb
    )

  def generate_ray_batch(self, cam_idx: int) -> utils.Batch:
    """Generate ray batch for a specified camera in the dataset."""
    # Generate rays for all pixels in the image.
    if self._flattened and not self.render_path:
      pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
          self.widths[cam_idx], self.heights[cam_idx]
      )
      rgb = self.images[cam_idx]
      return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx, rgb=rgb)
    else:
      pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
          self.width, self.height
      )
      return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx)  # pytype: disable=wrong-arg-types  # numpy-scalars

  def get_train_cameras(
      self, config: configs.Config, return_jax_cameras: bool = False
  ) -> tuple[chex.Array, chex.Array, Any, Any, Any] | camera_lib.Camera:
    """Returns cameras to be used for training.

    Args:
      config: The config to use.
      return_jax_cameras: If True, will return JAX camera instances rather than
        the camera tuple.

    Returns:
      A camera tuple consistent with `self.cameras` or a JAX camera instance if
      `return_jax_cameras` is True.
    """
    if config.use_identity_cameras:
      cameras = self._get_identity_cameras()
    elif config.use_perturbed_cameras:
      cameras = self._get_perturbed_cameras(config)
    else:
      cameras = self.cameras

    pixtocams, poses, distortion_params = cameras[:3]
    # Set the distortion params to not be None of we are optimizing for cameras.
    if config.optimize_cameras and not distortion_params:
      distortion_params = {
          'k1': 0.0,
          'k2': 0.0,
          'k3': 0.0,
      }
      distortion_params = jax.tree_util.tree_map(
          lambda x: np.zeros(self.cameras[0].shape[0]), distortion_params
      )

    cameras = (pixtocams, poses, distortion_params, *cameras[3:])

    if return_jax_cameras:
      image_sizes = np.array([(x.shape[1], x.shape[0]) for x in self.images])
      return jax.vmap(self.jax_camera_from_tuple_fn)(cameras, image_sizes)

    return cameras

  def _get_perturbed_cameras(
      self, config: configs.Config
  ) -> tuple[chex.Array, chex.Array, Any, Any, Any]:
    """Returns perturbed cameras."""
    rng = jax.random.PRNGKey(0)
    perturbed_cameras = maya_camera_utils.perturb_cameras(
        rng,
        self.jax_cameras,
        sigma_look_at=config.camera_perturb_sigma_look_at,
        sigma_position=config.camera_perturb_sigma_position,
        sigma_dolly_z=config.camera_perturb_sigma_dolly_z,
        sigma_focal_length=config.camera_perturb_sigma_focal_length,
        single_dolly=config.camera_perturb_intrinsic_single,
        dolly_use_average=config.camera_perturb_dolly_use_average,
    )
    if (
        perturbed_cameras.has_radial_distortion
        and config.camera_perturb_zero_distortion
    ):
      perturbed_cameras = perturbed_cameras.replace(
          radial_distortion=np.zeros_like(perturbed_cameras.radial_distortion)
      )
    camera_tuple = jax.vmap(camera_utils.tuple_from_jax_camera)(
        perturbed_cameras
    )
    return (*camera_tuple, *self.cameras[3:])

  def _get_identity_cameras(
      self,
  ) -> tuple[chex.Array, chex.Array, Any, Any, Any]:
    """Returns a set of cameras that are the identity."""
    pixtocams, poses = self.cameras[:2]
    poses = np.broadcast_to(np.eye(3, 4)[None], poses.shape).copy()
    poses[:, 2, 3] = 1.0
    swap_y_z = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ])
    poses = swap_y_z @ poses

    height, width = self.images[0].shape[:2]
    default_focal = width / (2 * np.tan(np.radians(72 / 2)))
    pixtocams = np.linalg.inv(
        np.array([
            [default_focal, 0, width / 2],
            [0, default_focal, height / 2],
            [0, 0, 1],
        ])
    )
    pixtocams = np.broadcast_to(
        pixtocams[None], (poses.shape[0], *pixtocams.shape)
    )
    return pixtocams, poses, None, *self.cameras[3:]


class RayBatcher(threading.Thread):
  """Thread for providing ray batch data during training and testing.

  Queues batches of ray and color data created by a Dataset object.

  The public interface mimics the behavior of a standard machine learning
  pipeline dataset provider that can provide infinite batches of data to the
  training/testing pipelines without exposing any details of how the batches are
  loaded/created or how this is parallelized. The initializer
  begins the thread using its parent start() method. After the initializer
  returns, the caller can request batches of data straight away.

  The internal self._queue is initialized as queue.Queue(3), so the infinite
  loop in run() will block on the call self._queue.put(self._next_fn()) once
  there are 3 elements. The main thread training job runs in a loop that pops 1
  element at a time off the front of the queue. The RayBatcher thread's run()
  loop will populate the queue with 3 elements, then wait until a batch has been
  removed and push one more onto the end.

  This repeats indefinitely until the main thread's training loop completes
  (typically tens/hundreds of thousands of iterations), then the main thread
  will exit and the RayBatcher thread will automatically be killed since it is a
  daemon.
  """

  def __init__(self, dataset: Dataset):
    super().__init__()

    self._queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True  # Sets parent Thread to be a daemon.
    self.split = dataset.split

    self.dataset = dataset
    self._test_camera_idx = 0
    self._n_examples = dataset._n_examples

    # Seed the queue with one batch to avoid race condition.
    if self.split == utils.DataSplit.TRAIN:
      self._next_fn = dataset._next_train
    else:
      self._next_fn = self._next_test
    self._queue.put(self._next_fn())
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: utils.Batch, contains `rays` and their associated metadata.
    """
    x = self._queue.get()
    if self.split == utils.DataSplit.TRAIN:
      return utils.shard(x)
    else:
      # Do NOT move test `rays` to device, since it may be very large.
      return x

  def _next_test(self) -> utils.Batch:
    """Sample next test batch (one full image)."""
    # Use the next camera index.
    cam_idx = self._test_camera_idx
    self._test_camera_idx = (self._test_camera_idx + 1) % self._n_examples
    return self.dataset.generate_ray_batch(cam_idx)

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: utils.Batch, contains `rays` and their associated metadata.
    """
    x = copy.copy(self._queue.queue[0])  # Make a copy of front of queue.
    if self.split == utils.DataSplit.TRAIN:
      return utils.shard(x)
    else:
      return jax.device_put(x)

  def run(self):
    while True:
      self._queue.put(self._next_fn())


class Blender(Dataset):
  """Blender Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    if config.render_path:
      raise ValueError('render_path cannot be used for the blender dataset.')

    _, camtoworlds, pixtocams, distortion_params, camtype, nameprefixes = (
        load_ngp_posedata(
            config, self.data_dir, f'transforms_{self.split.value}.json'
        )
    )

    def get_imgs(nameprefix):
      fprefix = os.path.join(self.data_dir, nameprefix)

      def get_img(f, fprefix=fprefix, is_16bit=False):
        if f.endswith('.exr'):
          image = image_io.load_exr(fprefix + f)
        else:
          image = image_io.load_img(fprefix + f, is_16bit)
        if config.factor > 1:
          image = image_utils.downsample(image, config.factor)
        return image

      if self._use_tiffs:
        channels = [get_img(f'_{ch}.tiff') for ch in ['R', 'G', 'B', 'A']]
        # Convert image to sRGB color space.
        image = image_utils.linear_to_srgb(np.stack(channels, axis=-1))
      elif self._use_exrs:
        image = get_img('.exr')
      else:
        image = get_img('.png') / 255.0

      if self._load_disps:
        disp_image = get_img('_disp.tiff', is_16bit=True)[..., :1] / 65535.0
      else:
        disp_image = None
      if self._load_normals:
        normal_image = get_img('_normal.png')[..., :3] * 2.0 / 255.0 - 1.0
      else:
        normal_image = None

      return image, disp_image, normal_image

    all_imgs = ParallelMap(get_imgs, nameprefixes)
    images, disp_images, normal_images = zip(*all_imgs)

    self.images = np.stack(images, axis=0)

    if self._use_exrs:
      # im_median = np.median(self.images)
      im_median = 0.65

      for i in range(self.images.shape[0]):
        if config.linear_to_srgb:
          self.images[i] = np.clip(
              image_utils.linear_to_srgb((self.images[i] / im_median) * 0.65),
              0.0,
              float('inf'),
              # 1.0
          )

    if self._load_disps:
      self.disp_images = np.stack(disp_images, axis=0)
    if self._load_normals:
      self.normal_images = np.stack(normal_images, axis=0)
      self.alphas = self.images[..., -1]

    rgb, alpha = self.images[..., :3], self.images[..., -1:]
    self.images = rgb * alpha + (1.0 - alpha)  # Use a white background.
    self.height, self.width = self.images[0].shape[:2]
    self.camtoworlds = camtoworlds
    self.pixtocams = pixtocams
    self.distortion_params = distortion_params
    self.camtype = camtype
    self.reload_mesh(config)

  def reload_mesh(self, config):
    if config.meshfile is not None and config.meshfile != '':
      f = os.path.join(LOCAL_MESHDIR, os.path.basename(config.meshfile))
      gfile.Copy(config.meshfile, f, overwrite=True)
      trimesh_mesh = trimesh.load_mesh(f, process=False)


class LLFF(Dataset):
  """LLFF Dataset."""

  def _load_renderings(self, config: configs.Config):
    """Load images from disk."""
    if config.image_subdir is None:
      image_subdir = 'images'
    else:
      image_subdir = config.image_subdir
    colmap_image_dir = os.path.join(self.data_dir, image_subdir)
    # Set up downscaling factor.
    factor = 1 if config.factor == 0 else config.factor
    # Train raw at full resolution because of the Bayer mosaic pattern.
    rawnerf_training = (
        config.rawnerf_mode and self.split == utils.DataSplit.TRAIN
    )
    if factor == 1 or rawnerf_training:
      image_dir_suffix = ''
      print('*** using full-resolution images')
    else:
      image_dir_suffix = f'_{config.factor}'
      print(f'*** using {factor}x downsampled images')

    bounds = None

    if config.llff_load_from_poses_bounds:
      print('*** Loading from poses_bounds.npy.')
      image_names = sorted(utils.listdir(colmap_image_dir))
      poses, pixtocams, distortion_params, camtype, bounds = load_llff_posedata(
          self.data_dir
      )
    elif config.load_ngp_format_poses:
      print('*** Loading NGP format poses', flush=True)
      image_names, poses, pixtocams, distortion_params, camtype, _ = (
          load_ngp_posedata(config, self.data_dir)
      )
    elif config.arcore_format_pose_file is not None:
      print('*** Loading ARCore format poses', flush=True)
      image_names, poses, pixtocams, distortion_params, camtype = (
          load_arcore_posedata(self.data_dir, config.arcore_format_pose_file)
      )
    else:
      print('*** Loading from COLMAP', flush=True)
      # Copy COLMAP data to local disk for faster loading.
      remote_colmap_dir = find_colmap_data(self.data_dir, config.colmap_subdir)
      CopyDir(remote_colmap_dir, LOCAL_COLMAP_DIR, overwrite=True)

      # Load poses.
      scenemanager = NeRFSceneManager(LOCAL_COLMAP_DIR)
      image_names, poses, pixtocams, distortion_params, camtype = (
          scenemanager.process(config.load_colmap_points)
      )
      if config.load_colmap_points:
        self.points = scenemanager.points3D
      print(f'*** Loaded camera parameters for {len(image_names)} images')

    # Previous NeRF results were generated with images sorted by filename,
    # use this flag to ensure metrics are reported on the same test set.
    if config.load_alphabetical:
      inds = np.argsort(image_names)
      image_names = [image_names[i] for i in inds]
      poses, pixtocams, distortion_params = camera_utils.gather_cameras(
          (poses, pixtocams, distortion_params), inds
      )
      print('*** image names sorted alphabetically')

    # Scale the inverse intrinsics matrix by the image downsampling factor.
    pixtocams = pixtocams @ np.diag([factor, factor, 1.0])
    pixtocams = pixtocams.astype(np.float32)
    self.camtype = camtype

    raw_testscene = False
    if config.rawnerf_mode:
      # Load raw images and metadata.
      images, metadata, raw_testscene = raw_utils.load_raw_dataset(
          self.split,
          self.data_dir,
          image_names,
          config.exposure_percentile,
          factor,
      )
      self.metadata = metadata

    else:
      # Load images.
      image_dir = os.path.join(self.data_dir, image_subdir + image_dir_suffix)
      print(f'*** Loading images from {image_dir}')
      for d in [image_dir, colmap_image_dir]:
        if not utils.file_exists(d):
          raise ValueError(f'Image folder {d} does not exist.')
      # Downsampled images may have different names vs images used for COLMAP,
      # so we need to map between the two sorted lists of files.
      colmap_files = sorted(utils.listdir(colmap_image_dir))
      file_indices = [i for i, f in enumerate(colmap_files) if f in image_names]

      def load_indexed_images(basedir):
        files = sorted(utils.listdir(basedir))
        paths = [os.path.join(basedir, files[i]) for i in file_indices]
        images = ParallelMap(image_io.load_img, paths)
        return images

      images = load_indexed_images(image_dir)
      # A lot of the code assumes 3 channels so drop any alphas.
      images = [z[..., :3] / 255.0 for z in images]
      print(f'*** Loaded {len(images)} images from disk')

      if (
          not config.render_path
          and config.use_masks is not configs.MaskInput.NONE
      ):
        images = np.array(images)
        if config.use_masks is configs.MaskInput.PNG:
          mask_dir = self.data_path / ('masks' + image_dir_suffix)
          if not utils.file_exists(mask_dir):
            raise ValueError(f'Masks folder {mask_dir} does not exist.')
          mask_images = np.stack(load_indexed_images(mask_dir), axis=0) / 255.0
        elif config.use_masks is configs.MaskInput.PROTO:
          mask_dir = self.data_path / 'masks'
          all_proto_paths = [mask_dir / f'{f}.proto' for f in image_names]
          empty_mask = np.ones(images.shape[1:])

          load_mask_fn = functools.partial(
              mask_utils.load_mask,
              scale_factor=config.factor,
          )
          mask_images = ParallelMap(load_mask_fn, all_proto_paths)
          if all(m is None for m in mask_images):
            raise ValueError(
                f'Set {config.use_masks=}, but no mask found in {mask_dir}'
            )
          mask_images = [empty_mask if m is None else m for m in mask_images]
          mask_images = np.stack(mask_images, axis=0)
        else:
          raise ValueError(f'Unsupported mask: {config.use_masks}')

        mask_images = mask_images > config.mask_threshold
        if len(mask_images.shape) == 3 and len(images.shape) == 4:
          mask_images = np.expand_dims(mask_images, axis=3)
        if mask_images.shape[:-1] != images.shape[:-1] or len(
            mask_images.shape
        ) != len(images.shape):
          raise ValueError(
              f'Masks images shape {mask_images.shape} and '
              + f'rgb images shape {images.shape} differ.'
          )
      else:
        mask_images = None

      # EXIF data is usually only present in the original JPEG images.
      jpeg_paths = [os.path.join(colmap_image_dir, f) for f in image_names]
      exifs = ParallelMap(image_io.load_exif, jpeg_paths)
      self.exifs = exifs
      if 'ExposureTime' in exifs[0] and 'ISOSpeedRatings' in exifs[0]:
        gather_exif_value = lambda k: np.array([float(x[k]) for x in exifs])
        shutters = gather_exif_value('ExposureTime')
        isos = gather_exif_value('ISOSpeedRatings')
        self.exposures = shutters * isos / 1000.0
        self.max_exposure = np.max(self.exposures)
      print(f'*** Loaded EXIF data for {len(exifs)} images')

      # Also load semantic data if the path is provided.
      if config.semantic_dir:
        semantic_files = sorted(utils.listdir(config.semantic_dir))
        colmap_to_extra = dict(zip(colmap_files, semantic_files))
        semantic_paths = [
            os.path.join(config.semantic_dir, colmap_to_extra[f])
            for f in image_names
        ]
        if config.semantic_format == 'npy':
          semantic_images = ParallelMap(utils.load_npy, semantic_paths)
          semantic_images = np.stack(semantic_images, axis=0)
        elif config.semantic_format == 'image':
          semantic_images = ParallelMap(image_io.load_img, semantic_paths)
          semantic_images = np.stack(semantic_images, axis=0) / 255
        else:
          raise ValueError(
              "semantic_format must be 'image' or 'npy' but is"
              f' {config.semantic_format}'
          )
        if len(semantic_images.shape) == 3:
          semantic_images = np.expand_dims(semantic_images, axis=-1)

    self.colmap_to_world_transform = np.eye(4)

    meters_per_colmap = (
        camera_utils.get_meters_per_colmap_from_calibration_images(
            config, poses, image_names
        )
    )
    self.scene_metadata = {'meters_per_colmap': meters_per_colmap}

    # Separate out 360 versus forward facing scenes.
    if config.forward_facing:
      # Set the projective matrix defining the NDC transformation.
      self.pixtocam_ndc = pixtocams.reshape(-1, 3, 3)[0]
      # Rescale according to a default bd factor.
      if bounds is None:
        bounds = np.array([0.01, 1.0])
        print(
            'Warning: Config.forward_facing=True but no scene bounds found.'
            'Defaulting to bounds [0.01, 1.0].'
        )
      scale = 1.0 / (bounds.min() * 0.75)
      poses[:, :3, 3] *= scale
      self.colmap_to_world_transform = np.diag([scale] * 3 + [1])
      bounds *= scale
      # Recenter poses.
      poses, transform = camera_utils.recenter_poses(poses)
      self.colmap_to_world_transform = (
          transform @ self.colmap_to_world_transform
      )
      # Forward-facing spiral render path.
      self.render_poses = camera_utils.generate_spiral_path(
          poses, bounds, n_frames=config.render_path_frames
      )
    else:
      # Rotate/scale poses to align ground with xy plane and fit to unit cube.
      if config.transform_poses_fn is None:
        transform_poses_fn = camera_utils.transform_poses_pca
      else:
        transform_poses_fn = config.transform_poses_fn
      poses, transform = transform_poses_fn(poses)
      self.colmap_to_world_transform = transform
      print('*** Constructed COLMAP-to-world transform.')

      if config.render_spline_keyframes is not None:
        self.spline_indices, self.render_poses, self.render_exposures = (
            camera_utils.create_render_spline_path(
                config, image_names, poses, self.exposures
            )
        )
        print(
            f'*** Constructed {len(self.render_poses)} render poses via '
            'spline interpolation.'
        )
      else:
        # Automatically generated inward-facing elliptical render path.
        self.render_poses = camera_utils.generate_ellipse_path(
            poses,
            n_frames=config.render_path_frames,
            z_variation=config.z_variation,
            z_phase=config.z_phase,
            rad_mult_min=config.rad_mult_min,
            rad_mult_max=config.rad_mult_max,
            render_rotate_xaxis=config.render_rotate_xaxis,
            render_rotate_yaxis=config.render_rotate_yaxis,
            lock_up=config.lock_up,
        )
        print(
            f'*** Constructed {len(self.render_poses)} render poses via '
            'ellipse path'
        )

    if raw_testscene:
      # For raw testscene, the first image sent to COLMAP has the same pose as
      # the ground truth test image. The remaining images form the training set.
      raw_testscene_poses = {
          utils.DataSplit.TEST: poses[:1],
          utils.DataSplit.TRAIN: poses[1:],
      }
      poses = raw_testscene_poses[self.split]

    self.poses = poses

    # Select the split.
    all_indices = np.arange(len(images))
    test_indices = all_indices[all_indices % config.llffhold == 0]
    if config.llff_use_all_images_for_training or raw_testscene:
      train_indices = all_indices
    elif (
        config.render_spline_keyframes or config.render_spline_keyframes_choices
    ):
      train_indices, test_indices = self._split_indices_with_spline_keyframes(
          config, all_indices, test_indices, image_names
      )
    else:
      train_indices = all_indices[all_indices % config.llffhold != 0]

    split_indices = {
        utils.DataSplit.TEST: test_indices,
        utils.DataSplit.TRAIN: train_indices,
    }
    print(
        '*** Constructed train/test split: '
        f'#train={len(train_indices)} #test={len(test_indices)}'
    )

    indices = split_indices[self.split]
    # All per-image quantities must be re-indexed using the split indices.
    images = [z for i, z in enumerate(images) if i in indices]
    poses, self.pixtocams, self.distortion_params = camera_utils.gather_cameras(
        (poses, pixtocams, distortion_params), indices
    )
    if self.exposures is not None:
      self.exposures = self.exposures[indices]
    if config.rawnerf_mode and self.metadata is not None:
      for key in ['exposure_idx', 'exposure_values']:
        self.metadata[key] = self.metadata[key][indices]
    if config.semantic_dir and semantic_images is not None:
      self.semantic_images = semantic_images[indices]

    if config.multiscale_train_factors is not None:
      all_images = images
      all_pixtocams = [self.pixtocams]
      lcm = np.lcm.reduce(config.multiscale_train_factors)
      print(f'*** Cropping images to a multiple of {lcm}')

      def crop(z):
        sh = z.shape
        return z[: (sh[0] // lcm) * lcm, : (sh[1] // lcm) * lcm]

      def downsample(z, factor):
        down_sh = tuple(np.array(z.shape[:-1]) // factor) + z.shape[-1:]
        return np.array(jax.image.resize(z, down_sh, 'bicubic'))

      images = [crop(z) for z in images]
      lossmult = [1.0] * len(images)
      # Warning: we use box filter downsampling here, for now.
      for factor in config.multiscale_train_factors:
        print(f'*** Downsampling by factor of {factor}x')
        all_images += [downsample(z, factor) for z in images]
        all_pixtocams.append(self.pixtocams @ np.diag([factor, factor, 1.0]))
        # Weight by the scale factor. In mip-NeRF I think we weighted by the
        # pixel area (factor**2) but empirically this seems to weight coarser
        # scales too heavily.
        lossmult += [factor] * len(images)

      n_copies = 1 + len(config.multiscale_train_factors)
      copy_inds = np.concatenate([np.arange(len(poses))] * n_copies, axis=0)
      _, poses, self.distortion_params = camera_utils.gather_cameras(
          (self.pixtocams, poses, self.distortion_params), copy_inds
      )
      self.lossmult = np.array(lossmult, dtype=np.float32)
      if self.exposures is not None:
        self.exposures = np.concatenate([self.exposures] * n_copies, axis=0)

      images = all_images
      self.pixtocams = np.concatenate(all_pixtocams, axis=0).astype(np.float32)

    heights = [z.shape[0] for z in images]
    widths = [z.shape[1] for z in images]
    const_height = np.all(np.array(heights) == heights[0])
    const_width = np.all(np.array(widths) == widths[0])
    if const_height and const_width:
      images = np.stack(images, axis=0)
    else:
      self.images_flattened, self.indices_flattened = flatten_data(images)
      self.heights = heights
      self.widths = widths
      self._flattened = True
      print(f'*** Flattened images into f{len(self.images_flattened)} pixels')

    self.images = images
    if config.use_masks and mask_images is not None:
      mask_images = mask_images[indices]
      self.mask_images = mask_images
    self.camtoworlds = self.render_poses if config.render_path else poses
    self.image_names = [
        image_name for i, image_name in enumerate(image_names) if i in indices
    ]
    self.height, self.width = images[0].shape[:2]
    print('*** LLFF successfully loaded!')
    print(f'*** split={self.split}')
    print(f'*** #images/poses/exposures={len(images)}')
    print(f'*** #camtoworlds={len(self.camtoworlds)}')
    print(f'*** resolution={(self.height, self.width)}')

  def _split_indices_with_spline_keyframes(
      self,
      config: configs.Config,
      all_indices: np.ndarray,
      test_indices: np.ndarray,
      all_image_names: List[str],
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Constructs train, test split indices when spline keyframes are present.

    When using keyframe-based spline paths, we want to avoid training on
    keyframes for two reasons: to use them for validation and to minimize the
    number of blurred pixels used in training (spline keyframes may be
    blurred). We add splint keyframes to the test split here.

    Args:
      config: Config object.
      all_indices: indices of all images available for train and test.
      test_indices: indices of additional test images.
      all_image_names: filenames for all images.

    Returns:
      train_indices: image indices to use in the train split.
      test_indices: image indices to use in the test split.
    """

    def _sorted_union(subsets):
      result = set()
      for subset in subsets:
        result = result.union(subset)
      return list(sorted(result))

    def _sorted_complement(superset, subset):
      return list(sorted(set(superset) - set(subset)))

    # Identify all sources for keyframes.
    spline_keyframe_sources = []
    if config.render_spline_keyframes:
      print(
          'Adding images from config.render_spline_keyframes to test '
          f'split: {config.render_spline_keyframes}'
      )
      spline_keyframe_sources.append(config.render_spline_keyframes)
    if config.render_spline_keyframes_choices:
      print(
          'Adding images from config.render_spline_keyframes_choices '
          f'to test split: {config.render_spline_keyframes_choices}'
      )
      spline_keyframe_sources.extend(
          config.render_spline_keyframes_choices.split(',')
      )

    spline_keyframe_indices = _sorted_union(
        [
            camera_utils.identify_file_indices(source, all_image_names)
            for source in spline_keyframe_sources
        ]
    )
    test_indices = _sorted_union([test_indices, spline_keyframe_indices])
    train_indices = _sorted_complement(all_indices, test_indices)

    return np.array(train_indices), np.array(test_indices)


class TanksAndTemplesNerfPP(Dataset):
  """Subset of Tanks and Temples Dataset as processed by NeRF++."""

  def _load_renderings(self, config):
    """Load images from disk."""
    if config.render_path:
      split_str = 'camera_path'
    else:
      split_str = self.split.value

    basedir = os.path.join(self.data_dir, split_str)

    def load_files(dirname, load_fn, shape=None):
      files = [
          os.path.join(basedir, dirname, f)
          for f in sorted(utils.listdir(os.path.join(basedir, dirname)))
      ]
      mats = np.array([load_fn(utils.open_file(f)) for f in files])
      if shape is not None:
        mats = mats.reshape(mats.shape[:1] + shape)
      return mats

    poses = load_files('pose', np.loadtxt, (4, 4))
    # Flip Y and Z axes to get correct coordinate frame.
    poses = np.matmul(poses, np.diag(np.array([1, -1, -1, 1])))

    intrinsics = load_files('intrinsics', np.loadtxt, (4, 4))

    if not config.render_path:
      images = load_files('rgb', lambda f: np.array(Image.open(f))) / 255.0
      self.images = images
      self.height, self.width = self.images[0].shape[:2]

    else:
      # Hack to grab the image resolution from a test image
      d = os.path.join(self.data_dir, 'test', 'rgb')
      f = os.path.join(d, sorted(utils.listdir(d))[0])
      shape = image_io.load_img(f).shape
      self.height, self.width = shape[:2]
      self.images = None

    self.camtoworlds = poses
    self.focal = intrinsics[0, 0, 0]
    self.pixtocams = np.linalg.inv(intrinsics)[..., :3, :3]


class TanksAndTemplesFVS(Dataset):
  """Subset of Tanks and Temples Dataset as processed by Free View Synthesis."""

  def _load_renderings(self, config):
    """Load images from disk."""
    render_only = config.render_path and self.split == utils.DataSplit.TEST

    basedir = os.path.join(self.data_dir, 'dense')
    sizes = [f for f in sorted(utils.listdir(basedir)) if f.startswith('ibr3d')]
    sizes = sizes[::-1]

    if config.factor >= len(sizes):
      raise ValueError(f'Factor {config.factor} larger than {len(sizes)}')

    basedir = os.path.join(basedir, sizes[config.factor])
    open_fn = lambda f: utils.open_file(os.path.join(basedir, f))

    files = [f for f in sorted(utils.listdir(basedir)) if f.startswith('im_')]
    if render_only:
      files = files[:1]
    load_image_fn = lambda f: np.array(Image.open(open_fn(f)))
    images = ParallelMap(load_image_fn, files)
    images = np.array(images) / 255.0

    names = ['Ks', 'Rs', 'ts']
    intrinsics, rot, trans = (np.load(open_fn(f'{n}.npy')) for n in names)

    # Convert poses from colmap world-to-cam into our cam-to-world.
    w2c = np.concatenate([rot, trans[..., None]], axis=-1)
    c2w_colmap = np.linalg.inv(camera_utils.pad_poses(w2c))[:, :3, :4]
    c2w = c2w_colmap @ np.diag(np.array([1, -1, -1, 1]))

    # Reorient poses so z-axis is up
    poses, _ = camera_utils.transform_poses_pca(c2w)
    self.poses = poses

    # A lot of the code assumes 3 channels so drop any alphas.
    self.images = images[..., :3]
    self.height, self.width = self.images[0].shape[:2]
    self.camtoworlds = poses
    self.focal = intrinsics[0, 0, 0]
    self.pixtocams = np.linalg.inv(intrinsics)[..., :3, :3]

    if render_only:
      render_path = camera_utils.generate_ellipse_path(
          poses,
          config.render_path_frames,
          z_variation=config.z_variation,
          z_phase=config.z_phase,
      )
      self.images = None
      self.camtoworlds = render_path
      self.render_poses = render_path
    else:
      # Select the split.
      all_indices = np.arange(images.shape[0])
      indices = {
          utils.DataSplit.TEST: all_indices[all_indices % config.llffhold == 0],
          utils.DataSplit.TRAIN: all_indices[
              all_indices % config.llffhold != 0
          ],
      }[self.split]

      self.images = self.images[indices][..., :3]
      self.camtoworlds = self.camtoworlds[indices]


class DTU(Dataset):
  """DTU Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    if config.render_path:
      raise ValueError('render_path cannot be used for the DTU dataset.')

    def load_image(i):
      # Set light condition string accordingly.
      if config.dtu_light_cond < 7:
        light_str = f'{config.dtu_light_cond}_r' + (
            '5000' if i < 50 else '7000'
        )
      else:
        light_str = 'max'

      # Load image.
      fname = os.path.join(self.data_dir, f'rect_{i:03d}_{light_str}.png')
      image = image_io.load_img(fname) / 255.0
      if config.factor > 1:
        image = image_utils.downsample(image, config.factor)

      # Load projection matrix from file.
      fname = path.join(
          self.data_dir, f'../../Calibration/cal18/pos_{i:03d}.txt'
      )
      with utils.open_file(fname, 'rb') as f:
        projection = np.loadtxt(f, dtype=np.float32)

      # Decompose projection matrix into pose and camera matrix.
      camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
      camera_mat = camera_mat / camera_mat[2, 2]
      pose = np.eye(4, dtype=np.float32)
      pose[:3, :3] = rot_mat.transpose()
      pose[:3, 3] = (t[:3] / t[3])[:, 0]
      camtoworld = pose[:3]

      if config.factor > 0:
        # Scale camera matrix according to downsampling factor.
        camera_mat = (
            np.diag([1.0 / config.factor, 1.0 / config.factor, 1.0]).astype(
                np.float32
            )
            @ camera_mat
        )
      pixtocam = np.linalg.inv(camera_mat)

      return image, camtoworld, pixtocam

    # Find out whether the particular scan has 49 or 65 images.
    n_images = len(utils.listdir(self.data_dir)) // 8

    ret = ParallelMap(load_image, range(1, n_images + 1))
    images, camtoworlds, pixtocams = zip(*ret)

    pixtocams = np.stack(pixtocams)
    camtoworlds = np.stack(camtoworlds)
    images = np.stack(images)

    def rescale_poses(poses):
      """Rescales camera poses according to maximum x/y/z value."""
      s = np.max(np.abs(poses[:, :3, -1]))
      out = np.copy(poses)
      out[:, :3, -1] /= s
      return out

    # Flip y and z axes to get poses in OpenGL coordinate system.
    camtoworlds = camtoworlds @ np.diag([1, -1, -1, 1]).astype(np.float32)
    # Center and scale poses.
    camtoworlds, _ = camera_utils.transform_poses_focus(camtoworlds)
    camtoworlds = rescale_poses(camtoworlds)

    all_indices = np.arange(images.shape[0])
    split_indices = {
        utils.DataSplit.TEST: all_indices[all_indices % config.llffhold == 0],
        utils.DataSplit.TRAIN: all_indices[all_indices % config.llffhold != 0],
    }
    indices = split_indices[self.split]

    self.images = images[indices]
    self.height, self.width = images.shape[1:3]
    self.camtoworlds = camtoworlds[indices]
    self.pixtocams = pixtocams[indices]


class PixelRig(Dataset):
  """Dataset to be used with Pixel rig data.

  The rig has 5 cameras arranged in a cross:
     4
   2 1 3
     5
  with the central camera (1) at the origin of the world coordinate system.
  """

  def _load_renderings(self, config):
    """Load images from disk."""
    images_dir = self.data_dir
    cameras_dir = images_dir.replace('scaled_images', 'scaled_camera_pose')
    image_filenames = utils.listdir(images_dir)
    camera_filenames = utils.listdir(cameras_dir)

    image_filenames.sort()
    camera_filenames.sort()
    assert len(image_filenames) == len(camera_filenames)
    images = []
    cams = []
    poses = []
    pixtocams = []
    for image_filename, camera_filename in zip(
        image_filenames, camera_filenames
    ):
      # Read image.
      image_path = os.path.join(images_dir, image_filename)
      images.append(image_io.load_img(image_path) / 255.0)
      # Read camera.
      camera_path = os.path.join(cameras_dir, camera_filename)
      with utils.open_file(camera_path, 'rb') as camerain:
        camera = vision_sfm.Camera.FromProto(
            camera_pb2.CameraProto.FromString(camerain.read())
        )
      cams.append(camera)
      pose = np.linalg.inv(camera.GetCameraFromWorldTransform())[:3, :4]
      # Flip from OpenCV to OpenGL coordinate space.
      pose = pose @ np.diag([1, -1, -1, 1])
      # Scale poses to use config's world space near bound, since NDC space
      # assumes world space near = 1.
      pose[:3, -1] *= 1.0 / config.near
      poses.append(pose)
      self.near = 0.0  # self.near must be 0 for NDC space rays.
      self.far = 1.0  # self.far must be 1 for NDC space rays.
      # Create inverse intrinsic matrix.
      pixtocams.append(
          np.linalg.inv(
              camera_utils.intrinsic_matrix(
                  camera.FocalLength(),
                  camera.FocalLength() * camera.PixelAspectRatio(),
                  camera.PrincipalPointX(),
                  camera.PrincipalPointY(),
              )
          )
      )

    poses = np.stack(poses, axis=0)
    # Rigid rotation on world space to ensure poses are facing "forward",
    # as this is required for the NDC transform.
    poses = np.diag([1, -1, -1]) @ poses

    # Generate simple circular render path for visualization.
    # The pixel rig data always has the central camera at the coordinate frame
    # origin, we rely on this assumption here.
    radius = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
    t_vec = lambda th: radius * np.array([np.cos(th), np.sin(th), 0.0])
    n_frames = config.render_path_frames
    render_poses = [
        np.concatenate([np.eye(3), t_vec(theta)[:, None]], axis=-1)
        for theta in np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    ]
    self.render_poses = np.stack(render_poses, axis=0)

    if config.render_path:
      self.camtoworlds = self.render_poses
      self.pixtocams = pixtocams[0]
    else:
      self.camtoworlds = poses
      self.pixtocams = np.stack(pixtocams, axis=0)
    self.images = np.stack(images, axis=0)
    self.images = self.images[..., :3]
    self.height, self.width = self.images[0].shape[:2]

    # Create projection matrix to use for NDC space.
    # We use a centered principal point because the NDC function assumes it.
    self.pixtocam_ndc = np.linalg.inv(
        camera_utils.intrinsic_matrix(
            cams[0].FocalLength(),
            cams[0].FocalLength(),
            self.width / 2.0,
            self.height / 2.0,
        )
    )


class RTMV(Dataset):
  """RTMV dataset."""

  def _load_renderings(self, config):
    """Load images from disk.

    RTMV dataset does not have train/test splits.
    As a result, all images are loaded at once.

    Args:
      config: The config object to use.
    """
    basedir = self.data_dir
    filenames = sorted(utils.listdir(basedir))

    image_filenames = [f for f in filenames if f.endswith('.exr')]
    depth_filenames = [f for f in filenames if f.endswith('.depth.exr')]
    seg_filenames = [f for f in filenames if f.endswith('.seg.exr')]
    image_filenames = [f for f in image_filenames if f not in depth_filenames]
    image_filenames = [f for f in image_filenames if f not in seg_filenames]
    camera_filenames = [f for f in filenames if f.endswith('.json')]

    image_filenames.sort()
    camera_filenames.sort()
    assert len(image_filenames) == len(camera_filenames)

    def get_img(f):
      image = image_io.load_exr(f)
      if config.factor > 1:
        image = image_utils.downsample(image, config.factor)
      return image

    images = []
    poses = []
    for image_filename, camera_filename in zip(
        image_filenames, camera_filenames
    ):
      # Read image.
      image_path = os.path.join(basedir, image_filename)
      channels = get_img(image_path)
      image = image_utils.linear_to_srgb(channels)

      images.append(image)

      # Read camera
      camera_path = os.path.join(basedir, camera_filename)
      with utils.open_file(camera_path, 'r') as fp:
        meta = json.load(fp)
        camera_data = meta['camera_data']
        poses.append(np.array(camera_data['cam2world']).T)

    self.height, self.width = image.shape[:2]
    if config.factor > 1:
      focal = float(camera_data['intrinsics']['fx']) / config.factor

    self.images = np.stack(images, axis=0)
    rgb, alpha = self.images[..., :3], self.images[..., -1:]
    self.images = rgb * alpha + (1.0 - alpha)  # Use a white background.
    self.focal = focal
    pixtocams = camera_utils.get_pixtocam(self.focal, self.width, self.height)

    # Generate simple circular render path for visualization.
    # The RTMV dataset has the scene mostly centered at the coordinate origin.
    # The render path has radius as 0.8 and elevated a bit higher than the
    # xy plane.
    radius = 0.8

    def circular(rad):
      t_vec = radius * camera_utils.normalize(
          np.array([np.cos(rad), np.sin(rad), 0.6])
      )
      return t_vec

    n_frames = config.render_path_frames
    render_poses = [
        camera_utils.viewmatrix(circular(t), np.array([0, 0, 1]), circular(t))
        for t in np.linspace(0, 2 * np.pi, n_frames, endpoint=True)
    ]
    self.render_poses = np.stack(render_poses, axis=0)

    if config.render_path:
      self.camtoworlds = self.render_poses
      self.pixtocams = pixtocams[0]
    else:
      self.camtoworlds = np.stack(poses, axis=0)[:, :3]
      self.pixtocams = np.stack(pixtocams, axis=0)


class Aerial(Dataset):
  """Dataset to be used with Aerial data."""

  def _load_renderings(self, config):
    """Load images from disk."""
    images_dir = os.path.join(self.data_dir, 'rgb')
    cameras_dir = os.path.join(self.data_dir, 'cameras')
    image_filenames = utils.listdir(images_dir)
    camera_filenames = utils.listdir(cameras_dir)

    assert len(image_filenames) == len(camera_filenames)
    image_filenames.sort()
    camera_filenames.sort()

    image_paths = [os.path.join(images_dir, name) for name in image_filenames]
    images = ParallelMap(image_io.load_img, image_paths)
    images = np.stack(images, axis=0) / 255

    camera_paths = [
        os.path.join(cameras_dir, name) for name in camera_filenames
    ]

    def load_camera_proto(camera_path):
      with utils.open_file(camera_path, 'rb') as camerain:
        camera = vision_sfm.Camera.FromProto(
            camera_pb2.CameraProto.FromString(camerain.read())
        )
      return camera

    cams = ParallelMap(load_camera_proto, camera_paths)

    poses = []
    pixtocams = []
    for _, camera in enumerate(cams):
      pose = np.linalg.inv(camera.GetCameraFromWorldTransform())[:3, :4]
      # Flip from OpenCV to OpenGL coordinates.
      pose = pose @ np.diag(np.array([1, -1, -1, 1]))
      # Scale the poses so that the Aerial scene fits within [-2, 2]^3. This
      # constant scale is set manually by the export code.
      pose[:3, -1] *= 1.0 / config.world_scale
      poses.append(pose)

      # Create inverse intrinsic matrix.
      pixtocams.append(np.linalg.inv(camera.GetCalibrationMatrix()))
    pixtocams = np.stack(pixtocams, axis=0)
    poses = np.stack(poses, axis=0)

    # Select the split.
    all_indices = np.arange(images.shape[0])
    train_indices = all_indices % config.llffhold != 0
    split_indices = {
        utils.DataSplit.TEST: all_indices[all_indices % config.llffhold == 0],
        utils.DataSplit.TRAIN: train_indices,
    }
    indices = split_indices[self.split]

    # All per-image quantities must be re-indexed using the split indices.
    images = images[indices]
    poses = poses[indices]
    pixtocams = pixtocams[indices]

    self.images = np.stack(images, axis=0)
    self.images = self.images[..., :3]
    self.pixtocams = pixtocams.astype(np.float32)
    self.camtoworlds = poses
    self.height, self.width = images.shape[1:3]

    if config.render_path:
      # Use the 'orbit_cameras', if available.
      render_cameras_dir = os.path.join(self.data_dir, 'orbit_cameras')
      if utils.isdir(render_cameras_dir):
        render_camera_filenames = utils.listdir(render_cameras_dir)
        render_camera_filenames.sort()
        render_camera_paths = [
            os.path.join(render_cameras_dir, name)
            for name in render_camera_filenames
        ]
        render_cams = ParallelMap(load_camera_proto, render_camera_paths)
        render_poses = []
        for _, camera in enumerate(render_cams):
          pose = np.linalg.inv(camera.GetCameraFromWorldTransform())[:3, :4]
          # Flip from OpenCV to OpenGL coordinates.
          pose = pose @ np.diag(np.array([1, -1, -1, 1]))
          # Scale the poses so that the Aerial scene fits within [-2, 2]^3. This
          # constant scale is set manually by the export code.
          pose[:3, -1] *= 1.0 / config.world_scale
          render_poses.append(pose)

          # Create inverse intrinsic matrix.
          self.pixtocams = np.linalg.inv(camera.GetCalibrationMatrix())
          self.width = camera.ImageSizeX()
          self.height = camera.ImageSizeY()
          self.focal = camera.FocalLength()
        self.camtoworlds = np.stack(render_poses, axis=0)
      else:
        # Generate simple circular render path for visualization.
        radius = 1.0  # Hard code this --- we know the scene is in [-1,1]^3.
        t_vec = lambda th: np.array(  # pylint: disable=g-long-lambda
            [radius * np.cos(th), radius * np.sin(th), radius]
        )
        n_frames = config.render_path_frames
        up = np.array([0.0, 0.0, 1.0])
        render_poses = [
            camera_utils.viewmatrix(t_vec(theta), up, t_vec(theta))
            for theta in np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
        ]
        self.camtoworlds = np.stack(render_poses, axis=0)
        self.focal = 3.0 * self.width
        self.pixtocams = np.array([
            [1.0 / self.focal, 0, -0.5 * self.width / self.focal],
            [0, -1.0 / self.focal, 0.5 * self.height / self.focal],
            [0, 0, -1.0],
        ])


class Maya(Dataset):
  """Dataset to be used with synthetic renderings from maya."""

  def _load_renderings(self, config):
    data_dir = gpath.GPath(config.data_dir, self.split.value)

    # Load image data.
    images = []
    for file in data_dir.iterdir():
      if str(file).endswith('png'):
        image = media.read_image(file)
        image = image.astype(np.float32) / 255.0
        images.append(image)

    self.images = np.stack(images)
    rgb, alpha = self.images[..., :3], self.images[..., -1:]
    self.images = rgb * alpha + (1.0 - alpha)  # Use a white background.

    model_path = gpath.GPath(data_dir, 'model.pb')
    # Load camera data.
    with model_path.open('rb') as f:
      model = model_pb2.ModelProto.FromString(f.read())

    def _convert_to_jax_cameras(
        cameras: Sequence[vision_sfm.Camera],
    ) -> camera_lib.Camera:
      """Converts a sequence of vsfm Cameras to a diffren jax Camera."""
      return misc_utils.tree_collate(
          [camera_io.from_sfm_camera(c) for c in cameras]
      )

    sfm_cameras = [vision_sfm.Camera.FromProto(v.camera) for v in model.views]
    jax_cameras = _convert_to_jax_cameras(sfm_cameras)

    pix_to_cam, cam_to_world, distortion_params = jax.vmap(
        camera_utils.tuple_from_jax_camera
    )(jax_cameras)

    self.pixtocams = np.asarray(pix_to_cam)
    self.camtoworlds = np.asarray(cam_to_world)
    self.distortion_params = distortion_params

    height, width = np.asarray(jax_cameras.image_size)[0]
    self.height = int(height)
    self.width = int(width)
    self.focal = float(jax_cameras.focal_length[0])
    self.poses = self.camtoworlds


class PreloadedData(Dataset):
  """Preloaded Dataset.

  Convenient for debugging in Colab using data that exists as local variables
  rather than on disk.
  """

  def _load_renderings(self, config, **kwargs):
    """Process passed-forward kwargs from constructor."""

    self.images = np.array(kwargs['images']).astype(np.float32)
    self.height, self.width = self.images[0].shape[:2]
    self.camtoworlds = np.array(kwargs['camtoworlds']).astype(np.float32)
    self.pixtocams = np.array(kwargs['pixtocams']).astype(np.float32)

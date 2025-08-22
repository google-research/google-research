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

"""Different datasets implementation plus a general port for all the datasets."""

import abc
import copy
import dataclasses
import functools
import os
import queue
import threading
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from camp_zipnerf.internal import camera_utils as teacher_camera_utils
from camp_zipnerf.internal import datasets as teacher_datasets
from camp_zipnerf.internal import utils as teacher_utils
import jax
import jax.numpy as jnp
import numpy as np
import pycolmap
from smerf.internal import camera_utils
from smerf.internal import configs
from smerf.internal import coord
from smerf.internal import train_utils
from smerf.internal import utils


def load_dataset(split, train_dir, config, cached_dataset=None):
  """Loads a split of a dataset using the data_loader specified by `config`."""
  dataset_dict = {
      'llff': LLFF,
  }
  return dataset_dict[config.dataset_loader](
      split, train_dir, config, cached_dataset
  )


class NeRFSceneManager(pycolmap.SceneManager):
  """COLMAP pose loader."""

  def process(
      self,
  ):
    """Applies NeRF-specific postprocessing to the loaded pose data.

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
    # self.load_points3D()  # For now, we do not need the point cloud data.

    # Assume shared intrinsics between all cameras.
    cam = self.cameras[1]

    # Extract focal lengths and principal point parameters.
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))

    # Extract extrinsic matrices in world-to-camera format.
    imdata = self.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
      im = imdata[k]
      rot = im.R()
      trans = im.tvec.reshape(3, 1)
      w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
      w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    c2w_mats = np.linalg.inv(w2c_mats)
    poses = c2w_mats[:, :3, :4]

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    names = [imdata[k].name for k in imdata]

    # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
    poses = poses @ np.diag([1, -1, -1, 1])

    # Get distortion parameters.
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

    return names, poses, pixtocam, params, camtype


class Dataset(threading.Thread, metaclass=abc.ABCMeta):
  """Dataset Base Class.

  Base class for a NeRF dataset. Creates batches of ray and color data used for
  training or rendering a NeRF model.

  Each subclass is responsible for loading images and camera poses from disk by
  implementing the _load_renderings() method. This data is used to generate
  train and test batches of ray + color data for feeding through the NeRF model.
  The ray parameters are calculated in _generate_rays().

  The public interface mimics the behavior of a standard machine learning
  pipeline dataset provider that can provide infinite batches of data to the
  training/testing pipelines without exposing any details of how the batches are
  loaded/created or how this is parallelized. Therefore, the initializer runs
  all setup, including data loading from disk using _load_renderings(), and
  begins the thread using its parent start() method. After the initializer
  returns, the caller can request batches of data straight away.

  The internal self._queue is initialized as queue.Queue(3), so the infinite
  loop in run() will block on the call self._queue.put(self._next_fn()) once
  there are 3 elements. The main thread training job runs in a loop that pops 1
  element at a time off the front of the queue. The Dataset thread's run() loop
  will populate the queue with 3 elements, then wait until a batch has been
  removed and push one more onto the end.

  This repeats indefinitely until the main thread's training loop completes
  (typically hundreds of thousands of iterations), then the main thread will
  exit and the Dataset thread will automatically be killed since it is a daemon.

  Attributes:
    alphas: np.ndarray, optional array of alpha channel data.
    cameras: tuple summarizing all camera extrinsic/intrinsic/distortion params.
    camtoworlds: np.ndarray, a list of extrinsic camera pose matrices.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.
    data_dir: str, location of the dataset on disk.
    disp_images: np.ndarray, optional array of disparity (inverse depth) data.
    distortion_params: dict, the camera distortion model parameters.
    far: float, far plane value for rays.
    focal: float, focal length from camera intrinsics.
    height: int, height of images.
    images: np.ndarray, array of RGB image data.
    near: float, near plane value for rays.
    normal_images: np.ndarray, optional array of surface normal vector data.
    pixtocams: np.ndarray, one or a list of inverse intrinsic camera matrices.
    pixtocam_ndc: np.ndarray, the inverse intrinsic matrix used for NDC space.
    poses: np.ndarray, optional array of auxiliary camera pose data.
    rays: utils.Rays, ray data for every pixel in the dataset.
    semantic_images: np.ndarray, optional array of semantic data.
    size: int, number of images in the dataset.
    split: str, indicates if this is a "train" or "test" dataset.
    width: int, width of images.
  """

  def __init__(
      self,
      split,
      data_dir,
      config,
      cached_dataset=None,
  ):
    super().__init__()

    # Initialize attributes
    self._queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True  # Sets parent Thread to be a daemon.
    self._patch_size = np.maximum(config.patch_size, 1)
    self._batch_size = config.batch_size // (
        jax.process_count() * config.gradient_accumulation_steps
    )
    if self._patch_size**2 > self._batch_size:
      raise ValueError(
          f'Patch size {self._patch_size}^2 too large for '
          + f'per-process batch size {self._batch_size}'
      )
    self._batching = utils.BatchingMethod(config.batching)
    self._use_tiffs = config.use_tiffs

    self._test_camera_idx = 0
    self._num_border_pixels_to_mask = 0
    self._cast_rays_in_train_step = config.cast_rays_in_train_step
    self._render_spherical = False

    self.split = utils.DataSplit(split)
    self.data_dir = data_dir
    self.near = config.near
    self.far = config.far
    self.distortion_params = None
    self.disp_images = None
    self.normal_images = None
    self.semantic_images = None
    self.alphas = None
    self.poses = None
    self.pixtocam_ndc = None
    self.camtype = camera_utils.ProjectionType.PERSPECTIVE

    # Providing type comments for these attributes, they must be correctly
    # initialized by _load_renderings() (see docstring) in any subclass.
    self.images: np.ndarray = None
    self.camtoworlds: np.ndarray = None
    self.pixtocams: np.ndarray = None
    self.height: int = None
    self.width: int = None

    # Load data from disk using provided config parameters.
    if not cached_dataset:
      self._load_renderings(config)
    else:
      self.images = cached_dataset.images
      self.camtoworlds = cached_dataset.camtoworlds
      self.pixtocams = cached_dataset.pixtocams
      self.height = cached_dataset.height
      self.width = cached_dataset.width
      self.focal = cached_dataset.focal
      self.poses = cached_dataset.poses
      self.distortion_params = cached_dataset.distortion_params

    self._n_examples = self.camtoworlds.shape[0]

    self.cameras = (
        self.pixtocams,
        self.camtoworlds,
        self.distortion_params,
        self.pixtocam_ndc,
    )

    # Seed the queue with one batch to avoid race condition.
    if self.split == utils.DataSplit.TRAIN:
      self._next_fn = self._next_train
    else:
      self._next_fn = self._next_test
    self._queue.put(self._next_fn())
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = self._queue.get()
    if self.split == utils.DataSplit.TRAIN:
      return utils.shard(x)
    else:
      # Do NOT move test `rays` to device, since it may be very large.
      return x

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = copy.copy(self._queue.queue[0])  # Make a copy of front of queue.
    if self.split == utils.DataSplit.TRAIN:
      return utils.shard(x)
    else:
      return jax.device_put(x)

  def run(self):
    while True:
      self._queue.put(self._next_fn())

  @property
  def size(self):
    return self._n_examples

  @abc.abstractmethod
  def _load_renderings(self, config):
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
    """
    # pyformat: enable

  def _make_ray_batch(
      self,
      pix_x_int,
      pix_y_int,
      cam_idx,
      lossmult = None,
  ):
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

    Returns:
      A dict mapping from strings utils.Rays or arrays of image data.
      This is the batch provided for one NeRF train or test iteration.
    """

    broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[Ellipsis, None]
    ray_kwargs = {
        'lossmult': broadcast_scalar(1.0) if lossmult is None else lossmult,
        'near': broadcast_scalar(self.near),
        'far': broadcast_scalar(self.far),
        'cam_idx': broadcast_scalar(cam_idx),
    }

    pixels = utils.Pixels(pix_x_int, pix_y_int, **ray_kwargs)
    if self._cast_rays_in_train_step and self.split == utils.DataSplit.TRAIN:
      # Fast path, defer ray computation to the training loop (on device).
      rays = pixels
    else:
      # Slow path, do ray computation using numpy (on CPU).
      rays = camera_utils.cast_ray_batch(
          self.cameras, pixels, self.camtype, xnp=np
      )

    # Create data batch.
    batch = {}
    batch['rays'] = rays
    batch['rgb'] = self.images[cam_idx, pix_y_int, pix_x_int]
    return utils.Batch(**batch)

  def _next_train(self):
    """Sample next training batch (random rays)."""
    # We assume all images in the dataset are the same resolution, so we can use
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
    lossmult = None

    return self._make_ray_batch(
        pix_x_int, pix_y_int, cam_idx, lossmult=lossmult
    )

  def generate_ray_batch(self, cam_idx):
    """Generate ray batch for a specified camera in the dataset."""
    if self._render_spherical:
      camtoworld = self.camtoworlds[cam_idx]
      rays = camera_utils.cast_spherical_rays(
          camtoworld, self.height, self.width, self.near, self.far, xnp=np
      )
      return utils.Batch(rays=rays)
    else:
      # Generate rays for all pixels in the image.
      pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
          self.width, self.height
      )
      return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx)

  def _next_test(self):
    """Sample next test batch (one full image)."""
    # Use the next camera index.
    cam_idx = self._test_camera_idx
    self._test_camera_idx = (self._test_camera_idx + 1) % self._n_examples
    return self.generate_ray_batch(cam_idx)


class LLFF(Dataset):
  """LLFF Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    colmap_image_dir = os.path.join(self.data_dir, 'images')
    # Set up scaling factor.
    image_dir_suffix = f'_{config.factor}'
    factor = config.factor

    # Load bounds if possible (only used in forward facing scenes).
    posefile = os.path.join(self.data_dir, 'poses_bounds.npy')
    poses_arr = None
    if utils.file_exists(posefile):
      with utils.open_file(posefile, 'rb') as fp:
        poses_arr = np.load(fp)
      bounds = poses_arr[:, -2:]
    else:
      bounds = np.array([0.01, 1.0])

    # Load pre-computed poses_bounds.npy in the format described in
    # https://github.com/Fyusion/LLFF. For example, this can be generated with
    # vision::sfm based pose estimation from the Insitu pipeline.
    if config.llff_load_from_poses_bounds:
      print('Loading from poses_bounds.npy.')
      image_names = sorted(utils.listdir(colmap_image_dir))

      if poses_arr is None:
        raise ValueError('poses_bounds.npy was not loaded correctly.')
      poses_hwf = poses_arr[:, :-2].reshape([-1, 3, 5])
      poses_llff = poses_hwf[:, :, :4]
      # Convert from [down, right, backwards] to [right, up, backwards] coord.
      nerf_to_llff = np.array([
          [0, -1, 0, 0],
          [1, 0, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
      ])
      poses = poses_llff @ nerf_to_llff
      h, w, f = poses_hwf[0, :, 4]
      pixtocam = camera_utils.get_pixtocam(f, w, h)
      distortion_params = None
      camtype = camera_utils.ProjectionType.PERSPECTIVE
    else:
      print('Loading from COLMAP')
      # Copy COLMAP data to local disk for faster loading.
      colmap_dir = os.path.join(self.data_dir, 'sparse/0/')

      # Load poses.
      scenemanager = NeRFSceneManager(colmap_dir)
      colmap_data = scenemanager.process()
      image_names, poses, pixtocam, distortion_params, camtype = colmap_data

    # Previous NeRF results were generated with images sorted by filename,
    # use this flag to ensure metrics are reported on the same test set.
    if config.load_alphabetical:
      inds = np.argsort(image_names)
      image_names = [image_names[i] for i in inds]
      poses = poses[inds]

    # Scale the inverse intrinsics matrix by the image downsampling factor.
    pixtocam = pixtocam @ np.diag([factor, factor, 1.0])
    self.pixtocams = pixtocam.astype(np.float32)
    self.focal = 1.0 / self.pixtocams[0, 0]
    self.distortion_params = distortion_params
    self.camtype = camtype

    # Load images.
    image_dir = os.path.join(self.data_dir, 'images' + image_dir_suffix)
    for d in [image_dir, colmap_image_dir]:
      if not utils.file_exists(d):
        raise ValueError(f'Image folder {d} does not exist.')
    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = sorted(utils.listdir(colmap_image_dir))
    image_files = sorted(utils.listdir(image_dir))
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [
        os.path.join(image_dir, colmap_to_image[f]) for f in image_names
    ]
    images = [utils.load_img(x) for x in image_paths]
    images = np.stack(images, axis=0) / 255.0

    self.colmap_to_world_transform = np.eye(4)

    # Separate out 360 versus forward facing scenes.
    if config.forward_facing:
      # Set the projective matrix defining the NDC transformation.
      self.pixtocam_ndc = self.pixtocams.reshape(-1, 3, 3)[0]
      # Rescale according to a default bd factor.
      scale = 1.0 / (bounds.min() * 0.75)
      poses[:, :3, 3] *= scale
      self.colmap_to_world_transform = np.diag([scale] * 3 + [1])
      bounds *= scale
      # Recenter poses.
      poses, transform = camera_utils.recenter_poses(poses)
      self.colmap_to_world_transform = (
          transform @ self.colmap_to_world_transform
      )
    else:
      # Rotate/scale poses to align ground with xy plane and fit to unit cube.
      poses, transform = camera_utils.transform_poses_pca(poses)
      self.colmap_to_world_transform = transform

    self.poses = poses

    # Select the split.
    all_indices = np.arange(images.shape[0])
    if config.llff_use_all_images_for_training:
      train_indices = all_indices
    else:
      train_indices = all_indices % config.llffhold != 0
    split_indices = {
        utils.DataSplit.TEST: all_indices[all_indices % config.llffhold == 0],
        utils.DataSplit.TRAIN: train_indices,
    }
    indices = split_indices[self.split]
    # All per-image quantities must be re-indexed using the split indices.
    images = images[indices]
    poses = poses[indices]

    self.images = images
    self.camtoworlds = poses
    self.height, self.width = images.shape[1:3]


def cam_to_rays(dataset, cam_idx, xnp=np):
  """Constructs rays for a single image.

  Args:
    dataset: mipnerf360 Dataset instance.
    cam_idx: int. Which camera index render.
    xnp: numpy or jax.numpy. Detetrmines if CPU or device 0 executes ray
      casting logic.

  Returns:
    Rays instance as returned from dataset.generate_ray_batch(). Rays will have
      origins, directions, and viewdirs populated.
  """
  # pylint: disable=protected-access
  cast_rays_now = dataset._cast_rays_now
  try:
    if xnp == np:
      # Constructs camera rays using CPU.
      dataset._cast_rays_now = True
      rays = dataset.generate_ray_batch(cam_idx).rays
    elif xnp == jnp:
      # Constructs camera rays using device=0.
      dataset._cast_rays_now = False
      rays = dataset.generate_ray_batch(cam_idx).rays
      rays = cast_rays(rays, dataset.cameras, dataset.camtype)
    else:
      raise NotImplementedError(xnp)
  finally:
    dataset._cast_rays_now = cast_rays_now
  # pylint: enable=protected-access
  return rays


@functools.partial(jax.jit, static_argnums=(2,))
def cast_rays(
    # Array arguments
    rays,
    cameras,
    # Constant arguments
    camtype,
):
  """Populates additional fields in Rays.

  Args:
    rays: Rays without origins, directions, or viewdirs.
    cameras: Camera parameters from Dataset.
    camtype: Type of camera model from Dataset.

  Returns:
    rays with additional fields populated.
  """
  return teacher_camera_utils.cast_ray_batch(cameras, rays, camtype, xnp=jnp)


pcast_rays = jax.pmap(
    cast_rays, in_axes=(0, 0), static_broadcasted_argnums=(2,)
)


def remove_glo_info(rays):
  """Strips GLO info from rays."""
  maybe_zeros = lambda x: jnp.zeros_like(x) if x is not None else x
  return rays.replace(cam_idx=maybe_zeros(rays.cam_idx))


def remove_exposure_info(
    rays, exposure_config,
):
  """Strips exposure info from rays.

  Replaces exposure_values with the median of a set of given exposures, if
  provided.

  Args:
    rays: Rays from Dataset.
    exposure_config: Config for exposure. See grid_utils.py.

  Returns:
    rays with exposure_idx and exposure_values replaced by defaults.
  """
  maybe_zeros = lambda x: jnp.zeros_like(x) if x is not None else x
  maybe_const = (
      lambda x, c: jnp.full_like(x, c, dtype=x.dtype) if x is not None else x
  )
  default_exposure = exposure_config['default_exposure']

  rays = rays.replace(
      exposure_idx=maybe_zeros(rays.exposure_idx),
      exposure_values=maybe_const(rays.exposure_values, default_exposure),
  )
  return rays


def set_exposure_values(rays, exposure):
  """Set exposure_values for all rays to `exposure`."""
  if exposure is not None:
    shape = rays.origins[Ellipsis, :1].shape
    rays = rays.replace(exposure_values=jnp.full(shape, exposure))
  return rays


# TODO(duckworthd): Fold this into a single pmap call.
def preprocess_rays(
    rays,
    mode,
    merf_config,
    dataset,
    pcameras = None,
    prng = None,
):
  """Preprocesses rays for model.

  Args:
    rays: Rays returned by a mipnerf360 dataset. Leading dimensions must be
      [num_local_devices, batch_size, ...].
    mode: 'train' or 'test'.
    merf_config: ...
    dataset: Dataset instance
    pcameras: Cameras corresponding to this dataset, replicated across all
      devices. Leading dimensions must be [num_local_devices, ...].
    prng: Random number generator. Leading dimensions must be
      [num_local_devices, ...].

  Returns:
    utils.Rays instance. Rays are guaranteed to have origins, directions, and
      sm_idxs populated.
  """
  assert mode in ['train', 'test'], mode

  # Get ray origins, directions, etc.
  if rays.origins is None:
    assert pcameras is not None
    assert dataset is not None
    rays = pcast_rays(rays, pcameras, dataset.camtype)

  # Always remove GLO info.
  rays = remove_glo_info(rays)

  # Teachers can get exposure information from two sources: the rays and the
  # predictions of their own internal Exposure MLP. Disable the former here.
  if not configs.use_exposure_in_teacher(merf_config):
    rays = remove_exposure_info(rays, merf_config.exposure_config)

  # Jitter ray position and direction
  if mode == 'train' and merf_config.enable_ray_jitter:
    rays = train_utils.pjitter_rays(prng, rays, merf_config)

  # Assign nearest submodel to each ray *after* jittering.
  sm_idxs = coord.nearest_submodel(
      t_positions=rays.origins,
      config=merf_config,
      grid_config=merf_config.grid_config,
  )

  # Return utils.Rays instance with sm_idxs attached.
  fields = dataclasses.asdict(rays)
  fields['sm_idxs'] = sm_idxs
  return utils.Rays(**fields)


class TestRaysPrefetcher(threading.Thread):
  """Prefetches test camera rays in daemon thread.

  Instances of this class provide an iterator over rays for full images. All
  computation is offloaded to a separate thread. Use this if the bulk of your
  runtime is dominated by on-device computations.

  Instances of this object can be used exactly once. Create a new instance to
  iterate again.
  """

  END_OF_QUEUE = 'END_OF_QUEUE'

  def __init__(self, dataset, cam_idxs, config, size=3):
    """Initializes TestRaysPrefetcher.

    Args:
      dataset: mipnerf360 dataset instance.
      cam_idxs: list[int]. Which cameras to render.
      config: Config instance.
      size: Number of cameras to enqueue at once.
    """
    super().__init__()

    self.daemon = True  # Sets parent Thread to be a daemon.

    size = min(size, len(cam_idxs))
    self._queue = queue.Queue(size)  # Set prefetch buffer to 3 batches.
    self.dataset = dataset
    self.cam_idxs = cam_idxs
    self.config = config

    # Initialize queue
    self.start()

  def __iter__(self):
    """Returns iterator over enqueued rays."""
    return self

  def __next__(self):
    """Returns the next set of rays.

    If an exception was encountered during ray construction, it will be raised
    here.

    Raises StopIteration after the last camera is yielded.
    """
    next_value = self._queue.get()
    if isinstance(next_value, Exception):
      raise next_value
    if next_value == self.END_OF_QUEUE:
      raise StopIteration
    return next_value

  def run(self):
    """Prepares all camera rays."""
    for cam_idx in self.cam_idxs:
      try:
        # TODO(duckworthd): Prepare rays on device.
        rays = cam_to_rays(self.dataset, cam_idx)  # f32[H, W, C]
        rays = preprocess_rays(
            rays=rays,
            mode='test',
            merf_config=self.config,
            dataset=self.dataset,
        )
        self._queue.put(rays)
      except Exception as err:  # pylint: disable=broad-exception-caught
        self._queue.put(err)
    self._queue.put(self.END_OF_QUEUE)

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

"""Utilities for distillation training."""

import contextlib
import functools
import os
import re
import textwrap
from typing import Optional

from absl import flags
from camp_zipnerf.internal import configs as mipnerf360_configs
from camp_zipnerf.internal import utils as mipnerf360_utils
from etils import epath
import gin
import jax
import numpy as np
from smerf.internal import configs as merf_configs
from smerf.internal import coord as merf_coord


_TEACHER_NAMESPACES = {
    'Config': 'camp_zipnerf.internal.configs',
    'HashEncoding': 'camp_zipnerf.internal.grid_utils',
    'MLP': 'camp_zipnerf.internal.models',
    'Model': 'camp_zipnerf.internal.models',
    'experiment_name': 'spin.utils.sweep_utils',
}

_MERF_NAMESPACES = {
    # configs.py
    'Config': 'smerf.internal.configs',
    # contract.py
    'contract': 'smerf.internal.coord',
    'uncontract': 'smerf.internal.coord',
    # hash_encoding.py
    'Multi3DGrid': 'smerf.internal.hash_encoding',
    'MultiHashEncoding': 'smerf.internal.hash_encoding',
    'MultiPropHashEncoding': 'smerf.internal.hash_encoding',
    # math.py
    'feature_activation': 'smerf.internal.math',
    # models.py
    'DeferredMLP': 'smerf.internal.models',
    'Model': 'smerf.internal.models',
    'PropMLP': 'smerf.internal.models',
    'MultiPropMLP': 'smerf.internal.models',
    'DensityAndFeaturesMLP': 'smerf.internal.models',
    'MultiDensityAndFeaturesMLP': 'smerf.internal.models',
    'MultiPostProcessingMLP': 'smerf.internal.models',
    # schedule.py
    'ConstSchedule': 'smerf.internal.schedule',
    'LogLerpSchedule': 'smerf.internal.schedule',
}


def load_config(save_config = True):
  """Loads and initializes Gin config from flags."""
  gin_bindings = flags.FLAGS.gin_bindings
  if flags.FLAGS.is_xm_sweep:
    gin_bindings += _additional_gin_bindings_for_sweep(gin_bindings)

  # Parse Gin config.
  gin.parse_config_files_and_bindings(
      flags.FLAGS.gin_configs, gin_bindings, skip_unknown=True
  )
  teacher_config = None
  merf_config = merf_configs.Config()

  if merf_config.distill_teacher_ckpt_dir:
    # TODO(duckworthd): Operate directly on
    # gin.configs._CONFIG[(namespace, full namespace)][cls_or_fn] = <value>
    base_merf_bindings = gin.config_str()
    additional_merf_bindings = """
      Config.data_dir = 'NOT_USED'
    """

    # The user may configure objects in both 'smerf' and 'camp_zipnerf' in their
    # original Gin config. If there is a naming conflict, prefer the 'smerf'
    # version of the object.
    merf_bindings = _set_namespaces(
        _MERF_NAMESPACES,
        base_merf_bindings + additional_merf_bindings,
    )

    # Read teacher config directly from file. Do not parse.
    teacher_config_path = (
        epath.Path(merf_config.distill_teacher_ckpt_dir) / 'config.gin'
    )
    assert teacher_config_path.exists(), teacher_config_path
    base_teacher_bindings = teacher_config_path.read_text()

    # Override some fields based on their value in the MERF config.
    additional_teacher_bindings = textwrap.dedent(f"""
      Config.batch_size = {_mipnerf360_batch_size(merf_config, 'train')}
      Config.factor = {merf_config.factor}
      Config.near = {merf_config.near}
      Config.far = {merf_config.far}
      Config.patch_size = {merf_config.patch_size}
      Config.render_chunk_size = {_mipnerf360_batch_size(merf_config, 'test')}
      eval/Config.near_plane_meters = None  # Don't clip when camera gets close to a surface
    """)

    # Process base & additional teacher bindings.
    teacher_bindings = {
        'base': base_teacher_bindings,
        'additional': additional_teacher_bindings,
    }
    for k, bindings in teacher_bindings.items():
      bindings = _set_namespaces(_TEACHER_NAMESPACES, bindings)
      bindings = _strip_deprecated_name_suffixes(
          ['Config.interlevel_loss_blurs'],  # Deleted in cl/553482273
          bindings,
      )
      teacher_bindings[k] = bindings

    # Initialize Gin using new bindings.
    gin.clear_config()
    gin.parse_config_files_and_bindings(
        None,
        # Apply Gin bindings in this order: teacher base, MERF, teacher
        # additional. The order ensures that teacher properties computed from
        # --gin_bindings take precedence over the contents of MERF's gin file.
        (
            f'{teacher_bindings["base"]}'
            f'\n{merf_bindings}'
            f'\n{teacher_bindings["additional"]}'
        ),
        skip_unknown=True,
    )
    teacher_config = mipnerf360_configs.Config()
    merf_config = merf_configs.Config()

  # Save Gin config.
  if save_config:
    raise NotImplementedError('Call distill.save_config() manually.')

  return merf_config, teacher_config


def save_config(logdir):
  """Saves Gin config to logdir."""
  if jax.host_id() == 0:
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / 'config.gin').write_text(gin.config_str())


def _additional_gin_bindings_for_sweep(gin_bindings):
  """Constructs additional Gin bindings if necessary."""
  assert flags.FLAGS.is_xm_sweep, f'{flags.FLAGS.is_xm_sweep=}'
  assert flags.FLAGS.xm_xid >= 0, f'{flags.FLAGS.xm_xid=}'
  assert flags.FLAGS.xm_wid >= 0, f'{flags.FLAGS.xm_wid=}'

  # Load user-specified Gin config to determine root directories for all
  # experiments in this sweep.
  gin.parse_config_files_and_bindings(
      flags.FLAGS.gin_configs, gin_bindings, skip_unknown=True
  )
  merf_config = merf_configs.Config()
  gin.clear_config()

  # Determine subdirectory for this work unit.
  wid = flags.FLAGS.xm_wid
  experiment_name = f'{wid:04d}'
  additional_gin_bindings = []

  # Modify checkpoint_dir.
  if (
      merf_config.checkpoint_dir
      and merf_config.one_checkpoint_dir_per_work_unit
  ):
    checkpoint_dir = os.path.join(merf_config.checkpoint_dir, experiment_name)
    additional_gin_bindings.append(
        f'smerf.internal.configs.Config.checkpoint_dir = "{checkpoint_dir}"',
    )

  # Modify baking_checkpoint_dir.
  if (
      merf_config.baking_checkpoint_dir is not None
      and merf_config.one_baking_checkpoint_dir_per_work_unit
  ):
    baking_checkpoint_dir = os.path.join(
        merf_config.baking_checkpoint_dir, experiment_name
    )
    additional_gin_bindings.append(
        'smerf.internal.configs.Config.baking_checkpoint_dir ='
        f' "{baking_checkpoint_dir}"'
    )

  return additional_gin_bindings


@contextlib.contextmanager
def _gin_parse(gin_configs=None, gin_bindings=None):
  config_str = gin.config_str()
  try:
    gin.clear_config()
    gin.parse_config_files_and_bindings(
        gin_configs, gin_bindings, skip_unknown=True
    )
    yield
  finally:
    gin.clear_config()
    gin.parse_config(config_str)


def _set_namespaces(name_to_namespace, gin_bindings):
  """Apply transform to each gin binding statement's namespace."""
  return _transform(
      gin_bindings,
      functools.partial(_set_namespace, name_to_namespace=name_to_namespace),
  )


def _strip_deprecated_name_suffixes(
    deprecated_name_suffixes, gin_bindings
):
  """Remove deprecated names from gin bindings."""
  return _transform(
      gin_bindings,
      functools.partial(
          _strip_deprecated_name_suffix,
          deprecated_name_suffixes=deprecated_name_suffixes,
      ),
  )


def _set_namespace(scopes, name, value, *, name_to_namespace):
  """Set the namespace of each name."""
  # Example:
  # --> name = camp_zipnerf.internal.models.Model.bg_intensity_range
  # --> old_namespace = 'camp_zipnerf.internal.models'
  # --> cls_or_fn = 'Model'
  # --> field = 'bg_intensity_range'
  *field_prefix, field = name.split('.')
  if field_prefix:
    *old_namespace, cls_or_fn = field_prefix
    old_namespace = '.'.join(old_namespace)
  else:
    old_namespace = ''
    cls_or_fn = None

  if cls_or_fn not in name_to_namespace:
    # No rule has been registered for this class or function. Keep the
    # namespace it already has.
    new_namespace = old_namespace
  else:
    new_namespace = name_to_namespace[cls_or_fn]
    if (
        new_namespace.endswith(old_namespace)
        or old_namespace.endswith(new_namespace)
    ):
      # The new and old namespaces refer to the same class or function. Use
      # the new namespace, which should be more complete.
      #
      # old_namespace = 'internal.models'
      # new_namespace = 'smerf.internal.models'
      # result        = 'smerf.internal.models'
      pass
    else:
      # The new and old namespace for this class or function aren't compatible.
      # Use the old namespace.
      #
      # old_namespace = 'camp_zipnerf.internal.models'
      # new_namespace = 'smerf.internal.models'
      # result        = 'camp_zipnerf.internal.models'
      new_namespace = old_namespace

  if cls_or_fn is not None:
    name = f'{cls_or_fn}.{field}'
  if new_namespace:
    name = f'{new_namespace}.{name}'
  return scopes, name, value


def _strip_deprecated_name_suffix(
    scopes, name, value, *, deprecated_name_suffixes
):
  for name_suffix in deprecated_name_suffixes:
    if name.endswith(name_suffix):
      return None
  return scopes, name, value


def _transform(s, transform_fn):
  """Applies a transformation function to each line in a Gin config."""
  result = []
  for line in s.splitlines():
    m = re.search('^([a-zA-Z0-9./_]+) = (.+)$', line.strip())
    if m is not None:
      name, value = m.groups()
      *scopes, name = name.split('/')
      transform_result = transform_fn(scopes, name, value)
      if transform_result is None:
        continue
      scopes, name, value = transform_result
      line = f'{name} = {value}'
      if scopes:
        scopes = '/'.join(scopes)
        line = f'{scopes}/{line}'
    result.append(line)
  return '\n'.join(result)


def _mipnerf360_batch_size(config, split):
  """Computes largest valid batch size for mipnerf360."""
  if split == 'train':
    # Batch must be evenly divisible by number of devices and patch size.
    factor = (
        jax.local_device_count()
        * config.patch_size
        * config.patch_size
    )
  elif split == 'test':
    # Batch must be evenly divisible by number of devices.
    factor = jax.local_device_count()
  else:
    raise NotImplementedError(split)

  # Actual batch size is evenly split across gradient accumulation steps.
  mipnerf360_batch_size = (
      config.batch_size // config.gradient_accumulation_steps
  )
  # Largest multiple evenly divisibly by factor.
  mult = mipnerf360_batch_size // factor
  return mult * factor


def create_prender_teacher(model, teacher_config):
  """Creates pmap'd render function for teacher."""
  _ = teacher_config

  def render_teacher(rng, state, batch):
    """Returns model predictions for each queried point."""
    # Extract rays. Rays must already be cast.
    rays = batch.rays
    assert rays.origins is not None

    # Render pixels.
    renderings, ray_history = model.apply(
        state.params,
        rng,
        rays,
        train_frac=1.0,
        compute_extras=True,
        zero_glo=True,
        train=False,  # Ensures ExposureMLP is used if available.
    )

    # Assemble supervision outputs. Use NerfMLP, not PropMLP.
    r = renderings[-1]
    h = ray_history[-1]
    result = {
        'tdist': h['tdist'],  # [...,p+1], range=[near, far]
        'weights': h['weights'],  # [...,p], range=[0,1]
        'rendered_rgb': r['rgb'],  # f32[...,3]
        'exposure_prediction': r.get(
            'exposure_prediction'
        ),  # f32[...,1] or None
    }

    return result

  prender_teacher = jax.pmap(
      render_teacher,
      axis_name='batch',
      in_axes=(0, 0, 0),
  )

  return prender_teacher


def create_prender_student(
    teacher_model,
    student_model,
    merf_config,
    alpha_threshold,
    return_ray_results,
):
  """Creates pmap'ed render_fn for models.render_image.

  Rays must already be cast before using the function returned here.

  Args:
    teacher_model: Teacher model instance.
    student_model: Student model instance.
    merf_config: Config for student model.
    alpha_threshold: float. Smallest allowed alpha value for a ray segment
      during rendering.
    return_ray_results: Boolean. If True, return additional results in student
      model.

  Returns:
    Function for rendering rays.
  """

  def render_eval_fn(teacher_params, student_params, train_frac, _, rays):
    """Renders student model with teacher and/or student's PropMLPs.

    Args:
      teacher_params: Teacher's model parameters.
      student_params: Student's model parameters.
      train_frac: float in [0, 1]. Percentage of training finished.
      _: ignored
      rays: Rays to render. Ray casting must already be applied.

    Returns:
      See models.Model.
    """
    # Rays must already be cast.
    assert rays.origins is not None

    # Don't override tdist unless requested to do so.
    tdist_override = None

    need_teacher_samples = (
        merf_config.distill_use_teacher_tdist
        or merf_config.distill_use_teacher_exposure
    )
    if need_teacher_samples:
      # Render teacher
      teacher_renderings, teacher_ray_history = teacher_model.apply(
          teacher_params,
          None,  # Deterministic.
          rays,
          train_frac=train_frac,
          compute_extras=True,
          zero_glo=True,
          train=False,  # Ensures ExposureMLP is used if available.
      )

      # Override tdist if necessary.
      if merf_config.distill_use_teacher_tdist:
        tdist_override = teacher_ray_history[-1]['tdist']

      # Replace camera's exposure values with teacher's predictions if they're
      # available. If the teacher doesn't use an ExposureMLP, this quantity will
      # be None.
      if merf_config.distill_use_teacher_exposure:
        exposure_values = teacher_renderings[-1].get('exposure_prediction')
        if exposure_values is not None:
          rays = rays.replace(exposure_values=exposure_values)

    # Render student.
    student_renderings, student_ray_history = student_model.apply(
        student_params,
        None,  # Deterministic.
        rays,
        train_frac=train_frac,
        return_ray_results=return_ray_results,
        alpha_threshold=alpha_threshold,
        tdist_override=tdist_override,
    )

    return jax.lax.all_gather(
        (student_renderings, student_ray_history),
        axis_name='batch',
    )

  # pmap only over variables and rays.
  render_eval_pfn = jax.pmap(
      render_eval_fn,
      in_axes=(0, 0, None, None, 0),
      axis_name='batch',
  )
  return render_eval_pfn


def alive_sm_idxs(origins, config, grid_config):
  """How many cameras are assigned to each submodel?

  Args:
    origins: f32[n, 3]. Camera origins.
    config: configs.Config instance.
    grid_config: See grid_utils.initialize_grid_config()

  Returns:
    sm_idxs: i32[k]. Submodels containing at least one camera.
    counts: i32[k]. Number of cameras assigned to each submodel.
  """
  # Construct one ray per camera.
  rays = mipnerf360_utils.Rays(origins=origins)

  # Assumption: all camera origins lie in [-1, 1]^3.
  assert np.all((rays.origins >= -1) & (rays.origins <= 1))

  # Determine which submodel each camera is assigned to.
  sm_idxs = merf_coord.rays_to_sm_idxs(rays, config, grid_config)

  # Count the number of cameras assigned to each submodel.
  sm_idxs, counts = np.unique(sm_idxs, return_counts=True)

  return sm_idxs, counts


def log_dir(config):
  """Where to write prebaked assets."""
  return epath.Path(config.checkpoint_dir)


def baked_log_dir(config):
  """Where to write baked assets."""
  result = log_dir(config)
  if config.baking_checkpoint_dir is not None:
    result = epath.Path(config.baking_checkpoint_dir)
  return result


def should_write_prebaked_assets(config):
  """True if job should write prebaked assets."""
  return (
      config.enable_train
      or config.enable_eval
      or config.enable_video
      or config.enable_render_path_video
  )


def should_write_baked_assets(config):
  """True if job should write baked assets."""
  return config.enable_baking or config.enable_baked_eval

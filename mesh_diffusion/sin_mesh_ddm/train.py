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

"""Train single mesh latent texture diffusion."""

import functools
import gc
import os
from typing import Any, Tuple, Union

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import periodic_actions
from clu import platform
from etils import epath
import flax
from flax.core import freeze
import flax.jax_utils as flax_utils
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

from mesh_diffusion.latents import dataset as ds
from mesh_diffusion.latents import io_utils as mio
from mesh_diffusion.latents import ring_vae
from mesh_diffusion.sin_im_ddm import diffusion as df
from mesh_diffusion.sin_im_ddm import net
from mesh_diffusion.sin_mesh_ddm import input_pipeline


@flax.struct.dataclass
class TrainState:
  step: int
  params: Any
  opt_state: optax.OptState
  encoder_params: Any
  decoder_params: Any
  key: Any
  sigma: Any


def merge_batch_stats(replicated_state):
  """Merge model batch stats."""
  if jax.tree.leaves(replicated_state.batch_stats):
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    return replicated_state.replace(
        batch_stats=cross_replica_mean(replicated_state.batch_stats)
    )
  else:
    return replicated_state


def create_train_state(
    config: ml_collections.ConfigDict,
    geom_data: Any,
    key: np.ndarray,
    num_multi_steps: int,
) -> Tuple[nn.Module, nn.Module, nn.Module, Any, TrainState]:
  """Create train state."""
  param_key, en_key, dec_key, state_key = jax.random.split(key, 4)

  # Encoder + Decoder
  trainer_dict = checkpoint.load_state_dict(config.trainer_checkpoint_dir)

  enc_params = trainer_dict['params']['Checkpointring_encoder_0']
  dec_params = trainer_dict['params']['Checkpointring_decoder_0']

  encoder = ring_vae.ring_encoder(
      features=128, latent_dim=config.latent_dim, num_heads=8
  )
  decoder = ring_vae.ring_decoder(features=512)

  ring_dummy = (
      jnp.zeros((50, config.attn_neigh, 2), dtype=jnp.float32),
      jnp.zeros((50, config.attn_neigh, 3), dtype=jnp.float32),
  )

  shape_dummy = (
      jnp.zeros((1000, config.latent_dim), dtype=jnp.float32),
      jnp.ones((1000,), jnp.int32),
      jnp.zeros((1000, 2), jnp.float32),
  )

  _ = encoder.init(en_key, *ring_dummy)
  _ = decoder.init(dec_key, *shape_dummy)

  enc_params = freeze(enc_params)
  dec_params = freeze(dec_params)

  unet_args = dict(
      features=config.unet_features,
      mlp_layers=config.mlp_layers,
      hdim=config.hdim,
  )

  # DDM Model

  if config.obj_labels:
    model = net.fc_unet_2_lab(**unet_args)
    geom_data = geom_data + (geom_data[0],)
  else:
    model = net.fc_unet_2(**unet_args)
    # model = net.fc_next(**unet_args)

  dummy_latents = jnp.zeros(
      (geom_data[0][0].shape[0], config.latent_dim), dtype=jnp.complex64
  )

  variables = model.init(param_key, dummy_latents, 3, *geom_data)

  if config.train:
    params = variables['params']
    # sigma = 0
  else:
    model_dict = checkpoint.load_state_dict(config.model_checkpoint_dir)
    params = model_dict['params']
    params = freeze(params)
    # if 'sigma' in model_dict:
    #   sigma = model_dict['sigma']
    # else:
    #   sigma = 0

  if config.schedule == 'constant':
    scheduler = config.learning_rate
  elif config.schedule == 'linear':
    scheduler = optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=config.end_learning_rate,
        transition_steps=config.num_steps,
    )
  else:
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=config.init_learning_rate,
        peak_value=config.learning_rate,
        end_value=config.end_learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.num_steps,
    )

  optimizer = optax.adam(learning_rate=scheduler, **config.adam_args)

  if num_multi_steps > 1:
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=num_multi_steps)

  return (
      model,
      encoder,
      decoder,
      optimizer,
      TrainState(
          step=0,
          params=params,
          opt_state=optimizer.init(params),
          encoder_params=enc_params,
          decoder_params=dec_params,
          key=state_key,
          sigma=1.0,
      ),
  )


@flax.struct.dataclass
class Metrics(metrics.Collection):
  loss: metrics.Average.from_output('loss')
  mean_grads: metrics.Average.from_output('mean_grads')
  max_grads: metrics.Average.from_output('max_grads')
  mean_updates: metrics.Average.from_output('mean_updates')
  max_updates: metrics.Average.from_output('max_updates')


def encode_signal(encoder: nn.Module, params: Any, enc_data: Any):
  """Encode signal."""
  ring_logs = enc_data[0]
  ring_vals = enc_data[1]
  mean, ln_var = encoder.apply({'params': params}, ring_logs, ring_vals)
  return mean, ln_var


def train_step(
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    geom_data: Any,
    state: TrainState,
    zt: Array,
    epst: Array,
    t: Any,
) -> Tuple[TrainState, metrics.Collection]:
  """Perform a single training step.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    optimizer: Optax optimizer.
    geom_data: Geometry inputs to DDM
    state: State of the model (optimizer and state).
    zt: Noisy variational latent codes |V| x latent_dim (complex)
    epst: ground truth offset
    t: timestep

  Returns:
    The new model state and dictionary with metrics.
  """

  def get_l1_loss(params):
    """L1 loss."""
    eps_pred = model.apply({'params': params}, zt, t, *geom_data)
    if isinstance(eps_pred, tuple):
      eps_pred, vq_loss = eps_pred
      diff = jnp.abs(eps_pred - epst)
      loss = jnp.mean(diff) + vq_loss
    else:
      diff = jnp.abs(eps_pred - epst)
      loss = jnp.mean(diff)
    return loss

  params = state.params

  loss_val, grads = jax.value_and_grad(get_l1_loss)(params)

  grads = jax.lax.pmean(grads, axis_name='batch')
  grads = jax.tree.map(jnp.conj, grads)

  updates, new_opt_state = optimizer.update(grads, state.opt_state, params)
  new_params = optax.apply_updates(params, updates)

  gravel, _ = jax.flatten_util.ravel_pytree(grads)
  uravel, _ = jax.flatten_util.ravel_pytree(updates)
  gravel = jnp.abs(gravel)
  uravel = jnp.abs(uravel)
  g_max = jnp.max(gravel)
  g_mean = jnp.mean(gravel)
  u_max = jnp.max(uravel)
  u_mean = jnp.mean(uravel)

  new_state = state.replace(
      step=state.step + 1,
      params=new_params,
      opt_state=new_opt_state,
  )

  metrics_update = Metrics.single_from_model_output(
      loss=loss_val,
      mean_grads=g_mean,
      max_grads=g_max,
      max_updates=u_max,
      mean_updates=u_mean,
  )

  return new_state, metrics_update


def train_step_z0(
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    geom_data: Any,
    state: TrainState,
    zt: Array,
    z0: Array,
    t: Any,
) -> Tuple[TrainState, metrics.Collection]:
  """Perform a single training step.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    optimizer: Optax optimizer.
    geom_data: Geometry inputs to DDM.
    state: State of the model (optimizer and state).
    zt: Noisy variational latent codes |V| x latent_dim (x 2).
    z0: z0.
    t: timestep.

  Returns:
    The new model state and dictionary with metrics.
  """
  key = state.key

  def get_l1_loss(params):
    """L1 loss."""
    z0_pred = model.apply({'params': params}, zt, t, *geom_data)

    if isinstance(z0_pred, tuple):
      z0_pred, vq_loss = z0_pred
      diff = jnp.abs(z0 - z0_pred)

      loss = jnp.mean(diff) / jnp.mean(jnp.abs(z0))

      loss = loss + vq_loss
    else:
      diff = jnp.abs(z0 - z0_pred)

      loss = jnp.mean(diff) / jnp.mean(jnp.abs(z0))
    return loss

  params = state.params

  loss_val, grads = jax.value_and_grad(get_l1_loss)(params)

  grads = jax.lax.pmean(grads, axis_name='batch')
  grads = jax.tree.map(jnp.conj, grads)

  updates, new_opt_state = optimizer.update(grads, state.opt_state, params)
  new_params = optax.apply_updates(params, updates)

  gravel, _ = jax.flatten_util.ravel_pytree(grads)
  uravel, _ = jax.flatten_util.ravel_pytree(updates)
  gravel = jnp.abs(gravel)
  uravel = jnp.abs(uravel)
  g_max = jnp.max(gravel)
  g_mean = jnp.mean(gravel)
  u_max = jnp.max(uravel)
  u_mean = jnp.mean(uravel)

  new_state = state.replace(
      step=state.step + 1, params=new_params, opt_state=new_opt_state, key=key
  )

  metrics_update = Metrics.single_from_model_output(
      loss=loss_val,
      mean_grads=g_mean,
      max_grads=g_max,
      max_updates=u_max,
      mean_updates=u_mean,
  )

  return new_state, metrics_update


def predict(
    model: nn.Module,
    geom_data: Any,  # Tuple[Array, ...],
    params: Any,
    zt: Array,
    t: int,
) -> Array:
  """Predict the noise at a single timestep.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    geom_data: Geometry inputs to DDM
    params: Model params
    zt: Noisy variational latent codes |V| x latent_dim (x 2)
    t: timestep

  Returns:
    epst: The predicted added noise
  """
  out = model.apply({'params': params}, zt, t, *geom_data)

  if isinstance(out, tuple):
    out = out[0]

  return out


def sample_latents(
    model: nn.Module,
    geom_data: Any,
    params: Any,
    key: np.ndarray,
    shape: Tuple[int, ...],
    timestep: int,
    sigma: float = 1.0,
    schedule: Any = 'linear',
    inpaint_data: Any = None,
):
  """Predict the noise at a single timestep.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    geom_data: Geometry inputs to DDM.
    params: Model params.
    key: RNGKey.
    shape: Shape out output latents B X |V| x latent_dim.
    timestep: timestep.
    sigma: Noise scale factor.
    schedule: schedule.
    inpaint_data: inpaint data.

  Returns:
    latents: Generated latent codes.
  """
  logging.info('Running sampling pmap...')
  p_predict_noise = jax.pmap(
      functools.partial(predict, model=model), axis_name='batch'
  )
  logging.info('Running sampling pmap... Done.')
  beta, alpha, alpha_bar, alpha_bar_prev, ln_var = df.get_diffusion_parameters(
      timestep, schedule
  )
  key, init_key = jax.random.split(key)
  zt = sigma * jax.random.normal(init_key, shape)
  for t in range(timestep, 0, -1):
    time_vec = t * jnp.ones(zt.shape[0], dtype=jnp.float32)
    zt_c = jax.lax.complex(zt[..., 0], zt[..., 1])
    epst = p_predict_noise(
        params=params, geom_data=geom_data, zt=zt_c, t=time_vec
    )
    key, noise_key = jax.random.split(key)
    noise = sigma * jax.random.normal(noise_key, zt.shape)
    if t <= 1:
      noise = 0.0 * noise
    epst = jnp.concatenate(
        (jnp.real(epst)[..., None], jnp.imag(epst)[..., None]), axis=-1
    )
    zt = df.predict_prev(
        beta[t - 1],
        alpha[t - 1],
        alpha_bar[t - 1],
        alpha_bar_prev[t - 1],
        ln_var[t - 1],
        zt,
        epst,
        noise,
    )
    if inpaint_data is not None:
      zt, key = df.encode_inpaint_mask(
          zt,
          inpaint_data[0],
          inpaint_data[1],
          alpha_bar_prev[t - 1],
          key,
          sigma,
      )

  return jax.lax.complex(zt[..., 0], zt[..., 1]), key


def sample_latents_z0(
    model: nn.Module,
    geom_data: Any,
    params: Any,
    key: np.ndarray,
    shape: Tuple[int, ...],
    timestep: int,
    sigma: float = 1.0,
    schedule: Any = 'linear',
    inpaint_data: Any = None,
):
  """Predict the noise at a single timestep.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    geom_data: Geometry inputs to DDM
    params: Model params
    key: RNGKey
    shape: Shape out output latents B X |V| x latent_dim (x 2)
    timestep: timestep
    sigma: sigma
    schedule: schedule.
    inpaint_data: inpaint_data.

  Returns:
    latents: Generated latent codes
  """
  logging.info('Running sampling pmap...')
  p_predict = jax.pmap(
      functools.partial(predict, model=model), axis_name='batch'
  )
  logging.info('Running sampling pmap... Done.')
  beta, alpha, alpha_bar, alpha_bar_prev, ln_var = df.get_diffusion_parameters(
      timestep, schedule
  )
  key, init_key = jax.random.split(key)
  zt = sigma * jax.random.normal(init_key, shape)
  for t in range(timestep, 0, -1):
    time_vec = t * jnp.ones(zt.shape[0], dtype=jnp.int32)
    zt_c = jax.lax.complex(zt[..., 0], zt[..., 1])
    z0_c = p_predict(params=params, geom_data=geom_data, zt=zt_c, t=time_vec)
    key, noise_key = jax.random.split(key)
    noise = sigma * jax.random.normal(noise_key, zt.shape)
    if t <= 1:
      noise = 0.0 * noise
    z0 = jnp.concatenate(
        (jnp.real(z0_c)[..., None], jnp.imag(z0_c)[..., None]), axis=-1
    )
    zt = df.predict_prev_from_z0(
        beta[t - 1],
        alpha[t - 1],
        alpha_bar[t - 1],
        alpha_bar_prev[t - 1],
        ln_var[t - 1],
        zt,
        z0,
        noise,
    )
    if inpaint_data is not None:
      zt, key = df.encode_inpaint_mask(
          zt,
          inpaint_data[0],
          inpaint_data[1],
          alpha_bar_prev[t - 1],
          key,
          sigma,
      )
  return jax.lax.complex(zt[..., 0], zt[..., 1]), key


def decode_latents(decoder: nn.Module, params: Any, geom_data: Any):
  """Decode."""
  return decoder.apply({'params': params}, *geom_data)


def reconstruct(
    p_decode_latents: Any, decoder_data: Any, latents: Array) -> Array:
  """Reconstruct."""
  # vert_inds = decoder_data[0]
  pix_tris = decoder_data[1]
  pix_bary = decoder_data[2]
  pix_logs = decoder_data[3]
  # valid_mask = decoder_data[4]
  data_i = decoder_data[5]
  data_j = decoder_data[6]
  tex_dim = decoder_data[7]

  pix_tris = jnp.reshape(pix_tris, (pix_tris.shape[0], -1))
  pix_logs = jnp.reshape(pix_logs, (pix_logs.shape[0], -1, 2))
  pix_recon = np.zeros(
      (pix_tris.shape[0], pix_tris.shape[1], 3), dtype=pix_logs.dtype
  )

  num_points = pix_tris.shape[1]

  n_splits = (num_points // 100000) + 1
  p_range = np.arange(num_points)

  p_splits = np.array_split(p_range, n_splits)

  for l in range(len(p_splits)):
    geom_data = (
        latents,
        pix_tris[:, p_splits[l]],
        pix_logs[:, p_splits[l], ...],
    )
    pix_r = p_decode_latents(geom_data=geom_data)

    pix_recon[:, p_splits[l], :] = np.asarray(pix_r)

  pix_recon = np.reshape(pix_recon, (pix_recon.shape[0], -1, 3, 3))
  pix_recon = np.sum(pix_recon * pix_bary[..., None], axis=2)

  pix_recon = np.clip(255.0 * np.asarray(pix_recon), 0, 255)
  pix_recon = pix_recon.astype(np.uint8)

  h = int(tex_dim[0, 0])
  w = int(tex_dim[0, 1])
  tr = np.zeros((latents.shape[0], h, w, 3), dtype=np.uint8)

  for l in range(latents.shape[0]):
    tr[l, data_i[l, :], data_j[l, :], :] = pix_recon[l, ...]
  return tr


def visualize_labels(decoder_data: Any, labels: Array) -> Array:
  """Visualize labels."""

  colors = [
      [128, 0, 0],
      [230, 25, 75],
      [250, 190, 212],
      [170, 110, 40],
      [245, 130, 48],
      [255, 255, 25],
      [255, 250, 200],
      [60, 180, 75],
      [170, 255, 195],
      [0, 128, 128],
      [70, 240, 240],
      [0, 0, 128],
      [0, 130, 200],
      [220, 190, 255],
      [240, 50, 230],
  ]
  colors = np.asarray(colors, dtype=np.uint8)

  # vert_inds = decoder_data[0]
  pix_tris = decoder_data[1]
  # pix_bary = decoder_data[2]
  # pix_logs = decoder_data[3]
  # valid_mask = decoder_data[4]
  data_i = decoder_data[5]
  data_j = decoder_data[6]
  tex_dim = decoder_data[7]

  compose_fn = input_pipeline.compose_labels
  pix_labels = jax.vmap(compose_fn, (0, 0), 0)(labels, pix_tris)[..., 0]
  pix_colors = colors[pix_labels]

  h = int(tex_dim[0, 0])
  w = int(tex_dim[0, 1])
  tr = np.zeros((labels.shape[0], h, w, 3), dtype=np.uint8)

  for l in range(labels.shape[0]):
    tr[l, data_i[l, :], data_j[l, :], :] = pix_colors[l, ...]
  return tr


def get_rng(seed: Union[None, int, Tuple[int, int]]) -> np.ndarray:
  """Returns a JAX RNGKey."""
  if seed is None:
    # Case 1: No random seed given, use XManager ID.
    # All processes (and restarts) get exactly the same seed but every work unit
    # and experiment is different.
    work_unit = platform.work_unit()
    rng = (work_unit.experiment_id, work_unit.id)
  elif isinstance(seed, int):
    # Case 2: Single integer given.
    rng = (0, seed)
  else:
    # Case 3: Tuple[int, int] given.
    if not isinstance(seed, (tuple, list)) or len(seed) != 2:
      raise ValueError(
          'Random seed must be an integer or tuple of 2 integers '
          f'but got {seed!r}'
      )
    rng = seed
  # JAX RNGKeys are arrays of np.uint32 and shape [2].
  return np.asarray(rng, dtype=np.uint32)


def extract_inds(z, inds):
  """Index."""
  return z[inds, :]


def ema(avg_params, new_params, decay, step):
  """Ema."""
  factor = min(decay, (step + 1) / (step + 10))
  new_avg = jax.tree_util.tree_map(
      lambda x, y: factor * x + (1 - factor) * y, avg_params, new_params
  )
  return new_avg


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  workdir = epath.Path(workdir)
  workdir.mkdir(parents=True, exist_ok=True)

  key = get_rng(config.seed)
  logging.info('Using random seed %s.', key)

  # Learning rate schedule.
  num_train_steps = config.num_steps

  timestep = config.timestep
  batch_size = config.batch_size
  batch_dim = jax.local_device_count()
  num_multi_steps = batch_size // batch_dim
  if num_multi_steps == 0:
    num_multi_steps = 1
  num_train_samples = config.num_train_samples
  num_test_samples = config.num_test_samples
  logging.info('num_train_steps=%d', num_train_steps)
  logging.info('num_diffusion_timesteps=%d', timestep)
  logging.info('batch_size=%d', batch_size)

  ## Setup data loaders
  geom_data, gtest_data = input_pipeline.get_geom_data(config)
  # geom_data.cache()

  g_data_init = gtest_data.take(1).get_single_element()
  geom_init = input_pipeline.get_conv_data(config, g_data_init)

  # Initalize models + train state
  key, init_key = jax.random.split(key, 2)

  model, encoder, decoder, optimizer, state = create_train_state(
      config, geom_init, init_key, num_multi_steps
  )

  # Latent codes + decoder params
  encoder_params = state.encoder_params
  decoder_params = state.decoder_params

  if state.sigma == 0:
    compute_sigma = True
  else:
    compute_sigma = False

  # Diffusion parameters
  _, _, ab_t, _, _ = df.get_diffusion_parameters(
      timestep, config.ddm_schedule
  )

  # Set up checkpointing of the model and the input pipeline.
  checkpoint_dir = workdir / 'checkpoints'
  ckpt = checkpoint.MultihostCheckpoint(
      os.fspath(checkpoint_dir), max_to_keep=2
  )
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step) + 1

  # Distribute training + eval steps
  state = flax_utils.replicate(state)

  logging.info('Running pmap...')
  if not config.predict_z0:
    p_train_step = jax.pmap(
        functools.partial(train_step, model=model, optimizer=optimizer),
        axis_name='batch',
    )
  else:
    p_train_step = jax.pmap(
        functools.partial(train_step_z0, model=model, optimizer=optimizer),
        axis_name='batch',
    )

  devices = jax.devices()
  print('Devices = ', flush=True)
  print(devices, flush=True)

  p_encode = jax.pmap(
      functools.partial(encode_signal, encoder=encoder, params=encoder_params),
      axis_name='batch',
  )

  p_decode = jax.pmap(
      functools.partial(decode_latents, decoder=decoder, params=decoder_params),
      axis_name='batch',
  )

  logging.info('Running pmap... Done.')

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0, asynchronous=False
  )
  if initial_step == 1:
    writer.write_hparams(dict(config))

  logging.info('Starting training loop at step %d.', initial_step)
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer
  )
  if jax.process_index() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(num_profile_steps=5, logdir=workdir),
    ]
  train_metrics = None

  num_geom_repeat = ds.get_num_repeat(
      num_train_samples, num_train_steps + 1, batch_dim
  )
  print('Num geom repeat = {}'.format(num_geom_repeat), flush=True)
  num_eval_steps = num_train_steps // config.eval_every_steps
  num_gtest_repeat = ds.get_num_repeat(
      num_test_samples, num_eval_steps + 1, config.num_recon_samples
  )

  logging.info('Precomputing latents...')

  # Batch size for precomputing latents is the largest factor common to
  # num_train_samples and num_test_samples, and is not greater than
  # the number of devices.
  bs = batch_dim  # batch_dim == number of devices.
  while (num_train_samples % bs > 0) or (num_test_samples % bs > 0):
    bs = bs - 1

  geom_b_data = geom_data.batch(bs, deterministic=True)
  geom_iter = iter(geom_b_data)
  latents = np.zeros(
      (num_train_samples, config.num_verts, config.latent_dim),
      dtype=np.complex64,
  )

  # For larger meshes, the device memory may not be enough to push all the nodes
  # through the encoder, so we encode in chunks.
  num_splits = 4  # This should be a configurable parameter.
  node_inds = np.array_split(np.arange(config.num_verts), num_splits)

  if config.obj_labels or (config.spec or config.inpaint_labels):
    gtest_b_data = gtest_data.batch(bs, deterministic=True)
    gtest_iter = iter(gtest_b_data)
    geom_nodes = np.zeros(
        (num_train_samples, config.num_verts, 3), dtype=np.float32
    )
    gtest_nodes = np.zeros(
        (num_test_samples, config.num_verts, 3), dtype=np.float32
    )

    if config.inpaint_labels:
      gtest_latents = np.zeros(
          (num_test_samples, config.num_verts, config.latent_dim),
          dtype=np.complex64,
      )

    for l in range(num_test_samples // bs):
      gtest_batch = next(gtest_iter)
      gtest_nodes[(l * bs) : (l + 1) * bs, ...] = np.asarray(
          input_pipeline.get_nodes(gtest_batch)
      )

      if config.inpaint_labels:
        enc_data = input_pipeline.get_encoder_data(gtest_batch)
        ring_logs = enc_data[0]
        ring_vals = enc_data[1]
        for s in range(num_splits):
          l_mean, _ = p_encode(
              enc_data=(
                  ring_logs[:, node_inds[s], ...],
                  ring_vals[:, node_inds[s], ...],
              )
          )
          gtest_latents[(l * bs) : (l + 1) * bs, node_inds[s], ...] = (
              np.asarray(l_mean)
          )

  for l in range(num_train_samples // bs):
    print('Processing {}-th example...'.format(l), flush=True)
    geom_batch = next(geom_iter)
    enc_data = input_pipeline.get_encoder_data(geom_batch)
    ring_logs = enc_data[0]
    ring_vals = enc_data[1]

    for s in range(num_splits):
      l_mean, _ = p_encode(
          enc_data=(
              ring_logs[:, node_inds[s], ...],
              ring_vals[:, node_inds[s], ...],
          )
      )
      latents[(l * bs) : (l + 1) * bs, node_inds[s], ...] = np.asarray(l_mean)

    if config.obj_labels or config.spec:
      geom_nodes[(l * bs) : (l + 1) * bs, ...] = np.asarray(
          input_pipeline.get_nodes(geom_batch)
      )

  mean_mag = np.mean(np.abs(latents))
  sigma = np.sqrt(2.0) * mean_mag

  if compute_sigma:
    state = flax_utils.unreplicate(state)
    state = state.replace(sigma=sigma)
    state = flax_utils.replicate(state)

  latents = tf.convert_to_tensor(latents)
  geom_latents = tf.data.Dataset.from_tensor_slices(latents)

  logging.info('Precomputing latents... Done.')

  if (not config.obj_labels) and (not config.spec):
    geom_data = tf.data.Dataset.zip((geom_data, geom_latents))

    if config.inpaint_labels:
      inpaint_labels = input_pipeline.load_and_map_mask(
          config, np.reshape(gtest_nodes, (-1, 3)), inpaint=True
      )
      inpaint_labels = np.reshape(
          inpaint_labels, (num_test_samples, config.num_verts)
      )

      inpaint_labels = tf.data.Dataset.from_tensor_slices(inpaint_labels)
      gtest_latents = tf.data.Dataset.from_tensor_slices(gtest_latents)

      gtest_data = tf.data.Dataset.zip(
          (gtest_data, gtest_latents, inpaint_labels)
      )
  else:

    logging.info('Loading and mapping labels...')

    if config.obj_labels:
      geom_labels = input_pipeline.load_and_map_mask(
          config, np.reshape(geom_nodes, (-1, 3))
      )
      gtest_labels = input_pipeline.load_and_map_mask(
          config, np.reshape(gtest_nodes, (-1, 3))
      )

      geom_labels = np.reshape(
          geom_labels, (num_train_samples, config.num_verts)
      )
      gtest_labels = np.reshape(
          gtest_labels, (num_test_samples, config.num_verts)
      )

    elif config.spec:

      geom_labels = input_pipeline.load_and_map_spec(
          config, np.reshape(geom_nodes, (-1, 3))
      )
      gtest_labels = input_pipeline.load_and_map_spec(
          config, np.reshape(gtest_nodes, (-1, 3))
      )

      geom_labels = np.reshape(
          geom_labels,
          (num_train_samples, config.num_verts, config.spec_features),
      )
      gtest_labels = np.reshape(
          gtest_labels,
          (num_test_samples, config.num_verts, config.spec_features),
      )

    geom_labels = tf.data.Dataset.from_tensor_slices(geom_labels)
    gtest_labels = tf.data.Dataset.from_tensor_slices(gtest_labels)

    if config.inpaint_labels:
      inpaint_labels = input_pipeline.load_and_map_mask(
          config, np.reshape(gtest_nodes, (-1, 3)), inpaint=True
      )
      inpaint_labels = np.reshape(
          inpaint_labels, (num_test_samples, config.num_verts)
      )

      inpaint_labels = tf.data.Dataset.from_tensor_slices(inpaint_labels)
      gtest_latents = tf.data.Dataset.from_tensor_slices(gtest_latents)

      gtest_data = tf.data.Dataset.zip(
          (gtest_data, gtest_labels, gtest_latents, inpaint_labels)
      )
    else:
      gtest_data = tf.data.Dataset.zip((gtest_data, gtest_labels))

    logging.info('Loading and mapping labels... Done.')

    geom_data = tf.data.Dataset.zip((geom_data, geom_latents, geom_labels))

  geom_data = geom_data.cache()
  geom_data = geom_data.repeat(100 * num_geom_repeat)
  geom_data = geom_data.shuffle(100, reshuffle_each_iteration=True)
  geom_data = geom_data.batch(
      batch_dim,
      drop_remainder=True,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False,
  )
  geom_data = geom_data.prefetch(tf.data.AUTOTUNE)

  gtest_data = gtest_data.cache()
  gtest_data = gtest_data.repeat(100 * num_gtest_repeat)
  gtest_data = gtest_data.shuffle(
      num_test_samples, reshuffle_each_iteration=True
  )
  gtest_data = gtest_data.batch(
      batch_dim,
      drop_remainder=True,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False,
  )
  gtest_data = gtest_data.prefetch(tf.data.AUTOTUNE)

  geom_iter = iter(geom_data)
  gtest_iter = iter(gtest_data)
  # Reconstruct inital geometry for comparison

  if (not config.obj_labels) and (not config.spec):
    geom_batch, latents_batch = next(geom_iter)
  else:
    geom_batch, latents_batch, labels_batch = next(geom_iter)
    labels_batch = jnp.asarray(labels_batch.numpy())

  latents_batch = jnp.asarray(latents_batch.numpy())

  recon_data = input_pipeline.get_decoder_data(geom_batch)

  init_recon = reconstruct(
      p_decode_latents=p_decode, decoder_data=recon_data, latents=latents_batch
  )
  init_images = {}
  init_images['Init recon'] = init_recon[0, ...]
  writer.write_images(1, init_images)
  if config.obj_labels:
    label_viz = visualize_labels(decoder_data=recon_data, labels=labels_batch)
    label_im = {}
    label_im['Labels'] = label_viz[0, ...]
    writer.write_images(1, label_im)

  with metric_writers.ensure_flushes(writer):
    if config.train:
      for step in range(initial_step, num_train_steps + 1):
        # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
        # devices.
        is_last_step = step == num_train_steps

        with jax.profiler.StepTraceAnnotation('train', step_num=step):
          if (not config.obj_labels) and (not config.spec):
            geom_batch, latents_batch = next(geom_iter)
            latents_batch = jnp.asarray(latents_batch.numpy())
            conv_data = input_pipeline.get_conv_data(config, geom_batch)
          else:
            geom_batch, latents_batch, signal_batch = next(geom_iter)
            latents_batch = jnp.asarray(latents_batch.numpy())
            signal_batch = jnp.asarray(signal_batch.numpy())
            conv_data = input_pipeline.get_conv_data(
                config, geom_batch, signal_batch
            )

          recon_data = input_pipeline.get_decoder_data(geom_batch)
          z0 = latents_batch
          zt, epst, t, key = df.encode_batched_tangent_latent_samples(
              z0, ab_t, key, sigma=sigma
          )

          key = np.asarray(key)
          if not config.predict_z0:
            state, metrics_update = p_train_step(
                geom_data=conv_data, state=state, zt=zt, epst=epst, t=t
            )
          else:
            state, metrics_update = p_train_step(
                geom_data=conv_data, state=state, zt=zt, z0=z0, t=t
            )

          metric_update = flax_utils.unreplicate(metrics_update)

        ## EMA
        if config.use_ema:
          if step == config.start_ema_after:
            state = flax_utils.unreplicate(state)
            params = state.params
            state = state.replace(ema_state=params)
            state = flax_utils.replicate(state)
          elif (
              step > config.start_ema_after
              and (step - config.start_ema_after) % config.update_ema_every == 0
          ):
            state = flax_utils.unreplicate(state)
            new_avg = ema(
                state.ema_state,
                state.params,
                config.ema_decay,
                step - config.start_ema_after + 1,
            )
            state = state.replace(ema_state=new_avg, params=new_avg)
            state = flax_utils.replicate(state)

        train_metrics = (
            metric_update
            if train_metrics is None
            else train_metrics.merge(metric_update)
        )

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
        for h in hooks:
          h(step)

        if step % config.log_loss_every_steps == 0 or is_last_step:
          writer.write_scalars(step, train_metrics.compute())
          train_metrics = None

        if (
            step % config.eval_every_steps == 0 or is_last_step
        ) and config.sample_during_training:
          with report_progress.timed('eval'):
            logging.info('Sampling diffusion scale to generate latent codes...')

            if (not config.obj_labels) and (not config.spec):
              if not config.inpaint_labels:
                gtest_batch = next(gtest_iter)
                inpaint_data = None
              else:
                gtest_batch, gtest_latents, gtest_inpaint = next(gtest_iter)
                gtest_latents = jnp.asarray(gtest_latents.numpy())
                gtest_inpaint = jnp.asarray(gtest_inpaint.numpy())
                inpaint_data = (gtest_latents, gtest_inpaint)

              geom_test_data = input_pipeline.get_conv_data(config, gtest_batch)

            else:
              if not config.inpaint_labels:
                gtest_batch, tsignal_batch = next(gtest_iter)
                inpaint_data = None
              else:
                gtest_batch, tsignal_batch, gtest_latents, gtest_inpaint = next(
                    gtest_iter
                )
                gtest_latents = jnp.asarray(gtest_latents.numpy())
                gtest_inpaint = jnp.asarray(gtest_inpaint.numpy())
                inpaint_data = (gtest_latents, gtest_inpaint)

              tsignal_batch = jnp.asarray(tsignal_batch.numpy())
              geom_test_data = input_pipeline.get_conv_data(
                  config, gtest_batch, tsignal_batch
              )

            recon_test_data = input_pipeline.get_decoder_data(gtest_batch)
            if not config.predict_z0:
              gen_latents, key = sample_latents(
                  model=model,
                  geom_data=geom_test_data,
                  params=state.params,
                  key=key,
                  shape=z0.shape + (2,),
                  timestep=timestep,
                  sigma=sigma,
                  schedule=config.ddm_schedule,
                  inpaint_data=inpaint_data,
              )
            else:
              gen_latents, key = sample_latents_z0(
                  model=model,
                  geom_data=geom_test_data,
                  params=state.params,
                  key=key,
                  shape=z0.shape + (2,),
                  timestep=timestep,
                  sigma=sigma,
                  schedule=config.ddm_schedule,
                  inpaint_data=inpaint_data,
              )

            logging.info('Sampling done, generated latent codes.')
            logging.info('Reconstructing with decoder...')

            gen_tex = reconstruct(
                p_decode_latents=p_decode,
                decoder_data=recon_test_data,
                latents=gen_latents,
            )
            logging.info('Reconstruction complete.')
            gen_images = {}
            for k in range(batch_dim):
              gen_images['Sample_{}'.format(k)] = gen_tex[k, ...]

            writer.write_images(step, gen_images)

        if step % config.checkpoint_every_steps == 0 or is_last_step:
          with report_progress.timed('checkpoint'):
            # state = merge_batch_stats(state)
            ckpt.save(flax_utils.unreplicate(state))

      logging.info('Finishing training at step %d', num_train_steps)

    # Pure sampling
    else:
      del geom_iter
      del geom_data
      gc.collect()
      latent_shape = (batch_dim, config.num_verts, config.latent_dim, 2)
      num_sample_steps = config.num_samples // batch_dim
      im_count = 0
      for step in range(initial_step, initial_step + num_sample_steps):
        with report_progress.timed('eval'):
          logging.info('Sampling diffusion scale to generate latent codes...')
          if (not config.obj_labels) and (not config.spec):
            if not config.inpaint_labels:
              gtest_batch = next(gtest_iter)
              inpaint_data = None
            else:
              gtest_batch, gtest_latents, gtest_inpaint = next(gtest_iter)
              gtest_latents = jnp.asarray(gtest_latents.numpy())
              gtest_inpaint = jnp.asarray(gtest_inpaint.numpy())
              inpaint_data = (gtest_latents, gtest_inpaint)

            geom_test_data = input_pipeline.get_conv_data(config, gtest_batch)

          else:
            if not config.inpaint_labels:
              gtest_batch, tsignal_batch = next(gtest_iter)
              inpaint_data = None
            else:
              gtest_batch, tsignal_batch, gtest_latents, gtest_inpaint = next(
                  gtest_iter
              )
              gtest_latents = jnp.asarray(gtest_latents.numpy())
              gtest_inpaint = jnp.asarray(gtest_inpaint.numpy())
              inpaint_data = (gtest_latents, gtest_inpaint)

            tsignal_batch = jnp.asarray(tsignal_batch.numpy())
            geom_test_data = input_pipeline.get_conv_data(
                config, gtest_batch, tsignal_batch
            )

          recon_test_data = input_pipeline.get_decoder_data(gtest_batch)
          if not config.predict_z0:
            gen_latents, key = sample_latents(
                model=model,
                geom_data=geom_test_data,
                params=state.params,
                key=key,
                shape=latent_shape,
                timestep=timestep,
                sigma=sigma,
                schedule=config.ddm_schedule,
                inpaint_data=inpaint_data,
            )
          else:
            gen_latents, key = sample_latents_z0(
                model=model,
                geom_data=geom_test_data,
                params=state.params,
                key=key,
                shape=latent_shape,
                timestep=timestep,
                sigma=sigma,
                schedule=config.ddm_schedule,
                inpaint_data=inpaint_data,
            )

          logging.info('Sampling done, generated latent codes.')
          logging.info('Reconstructing with decoder...')

          gen_tex = reconstruct(
              p_decode_latents=p_decode,
              decoder_data=recon_test_data,
              latents=gen_latents,
          )
          logging.info('Reconstruction complete.')
          gen_images = {}
          for k in range(batch_dim):
            gen_images['Sample_{}'.format(k)] = gen_tex[k, ...]

            if config.sample_save_dir:
              gen_im_file = config.sample_save_dir + '{}.jpg'.format(im_count)
              gen_png_file = config.sample_save_dir + '{}.png'.format(im_count)
              im_count = im_count + 1
              mio.save_png(gen_tex[k, ...], gen_im_file)
              mio.save_png(gen_tex[k, ...], gen_png_file)

          writer.write_images(step, gen_images)

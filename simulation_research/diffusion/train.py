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

"""Train and evaluate functions to train the diffusion model."""
from functools import partial  # pylint: disable=g-importing-member
import os
import pickle
import time

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import periodic_actions
import jax
from jax import jit
from jax import random
from jax import vmap
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

from simulation_research.diffusion import diffusion
from simulation_research.diffusion import ode_datasets
from simulation_research.diffusion import samplers
from simulation_research.diffusion import unet


def train_and_evaluate(config, workdir):
  """Execute diffusion model training and evaluation loop.

  See diffusion.py for more details.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    trained score function s(xₜ,t)=∇logp(xₜ).
  """

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0)
  report_progress = periodic_actions.ReportProgress(writer=writer)

  key = random.PRNGKey(config.seed)
  # Construct the dataset
  timesteps = config.dataset_timesteps
  ds = getattr(ode_datasets, config.dataset)(N=config.ds + config.bs)
  trajectories = ds.Zs[config.bs:, :timesteps]
  test_x = ds.Zs[:config.bs, :timesteps]
  data_std = trajectories.std()
  T_long = ds.T_long[:timesteps]  # pylint: disable=invalid-name
  dataset = tf.data.Dataset.from_tensor_slices(trajectories)
  dataiter = dataset.shuffle(len(dataset)).batch(config.bs).as_numpy_iterator
  assert trajectories.shape[1] == timesteps, "inconsistent data sizes"

  # initialize the model
  x = test_x  # (bs, N, C)
  modelconfig = unet.unet_64_config(
      x.shape[-1], base_channels=config.channels, attention=config.attention)
  model = unet.UNet(modelconfig)
  noise = getattr(diffusion, config.noisetype)
  difftype = getattr(diffusion, config.difftype)(noise)
  # whether or not to condition on initial timesteps
  cond_fn = lambda z: (z[:, :3] if config.ic_conditioning else None)

  # save the config and the data_std (used for normalization)
  with tf.Open(os.path.join(workdir, "config.pickle"), "wb") as f:
    pickle.dump(config, f)
  with tf.io.gfile.Open(os.path.join(workdir, "data_std.pickle"), "wb") as f:
    pickle.dump(data_std, f)
  # setup checkpoint saving
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir, {}, max_to_keep=2)

  ## train the model
  score_fn = diffusion.train_diffusion(
      model,
      dataiter,
      data_std,
      config.epochs,
      diffusion=difftype,
      lr=config.lr,
      writer=writer,
      report=report_progress,
      ckpt=ckpt,
      cond_fn=cond_fn)

  ## evaluate the model
  kstart = 3  # timepoint at which to start measuring errors

  @jit
  def log_prediction_metric(qs):
    """Log geometric mean of rollout relative error computed over trajectory.

    Takes trajectory qs, uses qs[kstart] as initial condition and integrates
    from there using the dataset ODE. Compares integrated vs qs.

    Args:
      qs: the trajectory (length, dimensions) to evaluate

    Returns:
      the log of the geomean of the rollout error
    """
    k = kstart
    traj = qs[k:]
    times = T_long[k:]
    traj_gt = ds.integrate(traj[0], times)
    return jnp.log(rel_err(traj, traj_gt)[1:len(times) // 2]).mean()

  @jit
  def pmetric(qs):
    """Geomean of rollout relative error, also taken over minibatch."""
    log_metric = vmap(log_prediction_metric)(qs)
    std_err = jnp.exp(log_metric.std() / jnp.sqrt(log_metric.shape[0]))
    return jnp.exp(log_metric.mean()), std_err  # also returns stderr

  eval_scorefn = partial(score_fn, cond=cond_fn(test_x))
  nll = samplers.compute_nll(difftype, eval_scorefn, key, test_x).mean()
  stoch_samples = samplers.sde_sample(
      difftype, eval_scorefn, key, test_x.shape, nsteps=1000, traj=False)
  err = pmetric(stoch_samples)[0]

  logging.info(f"{noise.__name__} gets NLL {nll:.3f} and err {err:.3f}")  # pylint: disable=logging-fstring-interpolation
  eval_metrics_cpu = jax.tree.map(np.array, {"NLL": nll, "err": err})
  writer.write_scalars(config.epochs, eval_metrics_cpu)
  report_progress(config.epochs, time.time())
  return score_fn


@jit
def rel_err(x, y):
  """Computes |x-y|/|x|+|y| with L1 norm taken along axis=-1."""
  return jnp.abs(x - y).sum(-1) / (jnp.abs(x).sum(-1) + jnp.abs(y).sum(-1))

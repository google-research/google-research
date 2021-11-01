# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Training and evalution for score-based generative models."""

import functools
import gc
import io
import os
import time
from typing import Any

from . import datasets
from . import evaluation
from . import losses
from . import models  # Keep this import for registering all model definitions.
from . import sampling
from . import utils
from .models import utils as mutils
from absl import logging
import flax
import flax.deprecated.nn as nn
import flax.jax_utils as flax_utils
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from .models import ddpm, ncsnv2, ncsnv3
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan


def train(config, workdir):
  """Runs a training loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  tf.io.gfile.makedirs(workdir)
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)
  rng = jax.random.PRNGKey(config.seed)
  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  if jax.host_id() == 0:
    writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  model_name = config.model.name
  ncsn_def = mutils.get_model(model_name).partial(config=config)
  rng, run_rng = jax.random.split(rng)
  # Whether the generative model is conditioned on class labels
  class_conditional = "conditional" in config.training.loss.lower()
  with nn.stateful() as init_model_state:
    with nn.stochastic(run_rng):
      input_shape = (jax.local_device_count(), config.data.image_size,
                     config.data.image_size, 3)
      input_list = [(input_shape, jnp.float32), (input_shape[:1], jnp.int32)]
      if class_conditional:
        input_list.append(input_list[-1])
      _, initial_params = ncsn_def.init_by_shape(
          model_rng, input_list, train=True)
      ncsn = nn.Model(ncsn_def, initial_params)

  optimizer = losses.get_optimizer(config).create(ncsn)

  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  del ncsn, init_model_state  # Do not keep a copy of the initial model.

  # Create checkpoints directory and the initial checkpoint
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  ckpt = utils.Checkpoint(
      checkpoint_dir,
      max_to_keep=None)
  ckpt.restore_or_initialize(state)

  # Save intermediate checkpoints to resume training automatically
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
  ckpt_meta = utils.Checkpoint(
      checkpoint_meta_dir,
      max_to_keep=1)
  state = ckpt_meta.restore_or_initialize(state)
  initial_step = int(state.step)
  rng = state.rng

  # Build input pipeline.
  rng, ds_rng = jax.random.split(rng)
  train_ds, eval_ds, _ = datasets.get_dataset(ds_rng, config)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  scaler = datasets.get_data_scaler(config)  # data normalizer
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Distribute training.
  optimize_fn = losses.optimization_manager(config)
  if config.training.loss.lower() == "ddpm":
    # Use score matching loss with DDPM-type perturbation.
    ddpm_params = mutils.get_ddpm_params()
    train_step = functools.partial(losses.ddpm_loss, ddpm_params=ddpm_params,
                                   train=True, optimize_fn=optimize_fn)
    eval_step = functools.partial(losses.ddpm_loss, ddpm_params=ddpm_params,
                                  train=False)
  else:
    # Use score matching loss with NCSN-type perturbation.
    sigmas = mutils.get_sigmas(config)
    # Whether to use a continuous distribution of noise levels
    continuous = "continuous" in config.training.loss.lower()
    train_step = functools.partial(
        losses.ncsn_loss,
        sigmas=sigmas,
        class_conditional=class_conditional,
        continuous=continuous,
        train=True,
        optimize_fn=optimize_fn,
        anneal_power=config.training.anneal_power)
    eval_step = functools.partial(
        losses.ncsn_loss,
        sigmas=sigmas,
        class_conditional=class_conditional,
        continuous=continuous,
        train=False,
        anneal_power=config.training.anneal_power)

  p_train_step = jax.pmap(train_step, axis_name="batch")
  p_eval_step = jax.pmap(eval_step, axis_name="batch")
  state = flax_utils.replicate(state)

  num_train_steps = config.training.n_iters

  logging.info("Starting training loop at step %d.", initial_step)
  rng = jax.random.fold_in(rng, jax.host_id())
  for step in range(initial_step, num_train_steps + 1):
    # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
    # devices.

    # Convert data to JAX arrays. Use ._numpy() to avoid copy.
    batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))  # pylint: disable=protected-access

    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
    next_rng = jnp.asarray(next_rng)
    loss, state = p_train_step(next_rng, state, batch)
    loss = flax.jax_utils.unreplicate(loss)

    # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)

    if jax.host_id() == 0 and step % 50 == 0:
      logging.info("step: %d, training_loss: %.5e", step, loss)
      writer.scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption.
    if step % config.training.snapshot_freq_for_preemption == 0 and jax.host_id(
    ) == 0:
      saved_state = flax_utils.unreplicate(state)
      saved_state = saved_state.replace(rng=rng)
      ckpt_meta.save(saved_state)

    # Report the loss on an evaluation dataset.
    if step % 100 == 0:
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), next(eval_iter))  # pylint: disable=protected-access
      eval_loss, _ = p_eval_step(next_rng, state, eval_batch)
      eval_loss = flax.jax_utils.unreplicate(eval_loss)
      if jax.host_id() == 0:
        logging.info("step: %d, eval_loss: %.5e", step, eval_loss)
        writer.scalar("eval_loss", eval_loss, step)

    # Save a checkpoint periodically and generate samples.
    if (step +
        1) % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      if jax.host_id() == 0:
        saved_state = flax_utils.unreplicate(state)
        saved_state = saved_state.replace(rng=rng)
        ckpt.save(saved_state)

      # Generate and save samples
      if config.training.snapshot_sampling:
        rng, sample_rng = jax.random.split(rng)
        init_shape = tuple(train_ds.element_spec["image"].shape)
        samples = sampling.get_samples(sample_rng,
                                       config,
                                       flax_utils.unreplicate(state),
                                       init_shape,
                                       scaler,
                                       inverse_scaler,
                                       class_conditional=class_conditional)
        this_sample_dir = os.path.join(
            sample_dir, "iter_{}_host_{}".format(step, jax.host_id()))
        tf.io.gfile.makedirs(this_sample_dir)

        if config.sampling.final_only:  # Do not save intermediate samples
          sample = samples[-1]
          image_grid = sample.reshape((-1, *sample.shape[2:]))
          nrow = int(np.sqrt(image_grid.shape[0]))
          sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
          with tf.io.gfile.GFile(
              os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
            np.save(fout, sample)

          with tf.io.gfile.GFile(
              os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
            utils.save_image(image_grid, fout, nrow=nrow, padding=2)
        else:  # Save all intermediate samples produced during sampling.
          for i, sample in enumerate(samples):
            image_grid = sample.reshape((-1, *sample.shape[2:]))
            nrow = int(np.sqrt(image_grid.shape[0]))
            sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, "sample_{}.np".format(i)),
                "wb") as fout:
              np.save(fout, sample)

            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, "sample_{}.png".format(i)),
                "wb") as fout:
              utils.save_image(image_grid, fout, nrow=nrow, padding=2)


def evaluate(config,
             workdir,
             eval_folder = "eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create eval_dir
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)

  # Build input pipeline.
  rng, ds_rng = jax.random.split(rng)
  _, eval_ds, _ = datasets.get_dataset(ds_rng, config, evaluation=True)
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  model_name = config.model.name
  ncsn_def = mutils.get_model(model_name).partial(config=config)
  rng, run_rng = jax.random.split(rng)
  class_conditional = "conditional" in config.training.loss.lower()
  with nn.stateful() as init_model_state:
    with nn.stochastic(run_rng):
      input_shape = tuple(eval_ds.element_spec["image"].shape[1:])
      input_list = [(input_shape, jnp.float32), (input_shape[:1], jnp.int32)]
      if class_conditional:
        input_list.append(input_list[-1])
      _, initial_params = ncsn_def.init_by_shape(
          model_rng, input_list, train=True)
      ncsn = nn.Model(ncsn_def, initial_params)

  optimizer = losses.get_optimizer(config).create(ncsn)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  del ncsn, init_model_state  # Do not keep a copy of the initial model.

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  if config.training.loss.lower() == "ddpm":
    # Use the score matching loss with DDPM-type perturbation.
    ddpm_params = mutils.get_ddpm_params()
    eval_step = functools.partial(
        losses.ddpm_loss, ddpm_params=ddpm_params, train=False)
  else:
    # Use the score matching loss with NCSN-type perturbation.
    sigmas = mutils.get_sigmas(config)
    continuous = "continuous" in config.training.loss.lower()
    eval_step = functools.partial(
        losses.ncsn_loss,
        sigmas=sigmas,
        continuous=continuous,
        class_conditional=class_conditional,
        train=False,
        anneal_power=config.training.anneal_power)

  p_eval_step = jax.pmap(eval_step, axis_name="batch")

  rng = jax.random.fold_in(rng, jax.host_id())

  # A data class for checkpointing.
  @flax.struct.dataclass
  class EvalMeta:
    ckpt_id: int
    round_id: int
    rng: Any

  # Add one additional round to get the exact number of samples as required.
  num_rounds = config.eval.num_samples // config.eval.batch_size + 1

  eval_meta = EvalMeta(ckpt_id=config.eval.begin_ckpt, round_id=-1, rng=rng)
  eval_meta = checkpoints.restore_checkpoint(
      eval_dir, eval_meta, step=None, prefix=f"meta_{jax.host_id()}_")

  if eval_meta.round_id < num_rounds - 1:
    begin_ckpt = eval_meta.ckpt_id
    begin_round = eval_meta.round_id + 1
  else:
    begin_ckpt = eval_meta.ckpt_id + 1
    begin_round = 0

  rng = eval_meta.rng
  # Use inceptionV3 for images with higher resolution
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  logging.info("begin checkpoint: %d", begin_ckpt)
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    ckpt_filename = os.path.join(checkpoint_dir, "ckpt-{}.flax".format(ckpt))

    # Wait if the target checkpoint hasn't been produced yet.
    waiting_message_printed = False
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed and jax.host_id() == 0:
        logging.warn("Waiting for the arrival of ckpt-%d.flax", ckpt)
        waiting_message_printed = True
      time.sleep(10)

    # In case the file was just written and not ready to read from yet.
    try:
      state = utils.load_state_dict(ckpt_filename, state)
    except:
      time.sleep(60)
      try:
        state = utils.load_state_dict(ckpt_filename, state)
      except:
        time.sleep(120)
        state = utils.load_state_dict(ckpt_filename, state)

    pstate = flax.jax_utils.replicate(state)
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

    # Compute the loss function on the full evaluation dataset.
    all_losses = []
    for i, batch in enumerate(eval_iter):
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)  # pylint: disable=protected-access
      eval_loss, _ = p_eval_step(next_rng, pstate, eval_batch)
      eval_loss = flax.jax_utils.unreplicate(eval_loss)
      all_losses.append(eval_loss)
      if (i + 1) % 1000 == 0 and jax.host_id() == 0:
        logging.info("Finished %dth step loss evaluation", i + 1)

    all_losses = jnp.asarray(all_losses)

    state = jax.device_put(state)
    # Sampling and computing statistics for Inception scores, FIDs, and KIDs.
    # Designed to be pre-emption safe. Automatically resumes when interrupted.
    for r in range(begin_round, num_rounds):
      if jax.host_id() == 0:
        logging.info("sampling -- ckpt: %d, round: %d", ckpt, r)
      rng, sample_rng = jax.random.split(rng)
      init_shape = tuple(eval_ds.element_spec["image"].shape)

      this_sample_dir = os.path.join(
          eval_dir, f"ckpt_{ckpt}_host_{jax.host_id()}")
      tf.io.gfile.makedirs(this_sample_dir)
      samples = sampling.get_samples(sample_rng, config, state, init_shape,
                                     scaler, inverse_scaler,
                                     class_conditional=class_conditional)
      samples = samples[-1]
      samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
      samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, 3))
      with tf.io.gfile.GFile(
          os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, samples=samples)
        fout.write(io_buffer.getvalue())

      gc.collect()
      latents = evaluation.run_inception_distributed(samples, inception_model,
                                                     inceptionv3=inceptionv3)
      gc.collect()
      with tf.io.gfile.GFile(
          os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
        fout.write(io_buffer.getvalue())

      eval_meta = eval_meta.replace(ckpt_id=ckpt, round_id=r, rng=rng)
      # Save an intermediate checkpoint directly if not the last round.
      # Otherwise save eval_meta after computing the Inception scores and FIDs
      if r < num_rounds - 1:
        checkpoints.save_checkpoint(
            eval_dir,
            eval_meta,
            step=ckpt * num_rounds + r,
            keep=1,
            prefix=f"meta_{jax.host_id()}_")

    # Compute inception scores, FIDs and KIDs.
    if jax.host_id() == 0:
      # Load all statistics that have been previously computed and saved.
      all_logits = []
      all_pools = []
      for host in range(jax.host_count()):
        this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}_host_{host}")

        stats = tf.io.gfile.glob(
            os.path.join(this_sample_dir, "statistics_*.npz"))
        wait_message = False
        while len(stats) < num_rounds:
          if not wait_message:
            logging.warn("Waiting for statistics on host %d", host)
            wait_message = True
          stats = tf.io.gfile.glob(
              os.path.join(this_sample_dir, "statistics_*.npz"))
          time.sleep(1)

        for stat_file in stats:
          with tf.io.gfile.GFile(stat_file, "rb") as fin:
            stat = np.load(fin)
            if not inceptionv3:
              all_logits.append(stat["logits"])
            all_pools.append(stat["pool_3"])

      if not inceptionv3:
        all_logits = np.concatenate(
            all_logits, axis=0)[:config.eval.num_samples]
      all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

      # Load pre-computed dataset statistics.
      data_stats = evaluation.load_dataset_stats(config)
      data_pools = data_stats["pool_3"]

      if hasattr(config.eval, "num_partitions"):
        # Divide samples into several partitions and compute FID/KID/IS on them.
        assert not inceptionv3
        fids = []
        kids = []
        inception_scores = []
        partition_size = config.eval.num_samples // config.eval.num_partitions
        tf_data_pools = tf.convert_to_tensor(data_pools)
        for i in range(config.eval.num_partitions):
          this_pools = all_pools[i * partition_size:(i + 1) * partition_size]
          this_logits = all_logits[i * partition_size:(i + 1) * partition_size]
          inception_scores.append(
              tfgan.eval.classifier_score_from_logits(this_logits))
          fids.append(
              tfgan.eval.frechet_classifier_distance_from_activations(
                  data_pools, this_pools))
          this_pools = tf.convert_to_tensor(this_pools)
          kids.append(
              tfgan.eval.kernel_classifier_distance_from_activations(
                  tf_data_pools, this_pools).numpy())

        fids = np.asarray(fids)
        inception_scores = np.asarray(inception_scores)
        kids = np.asarray(kids)
        with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_all_{ckpt}.npz"),
                               "wb") as f:
          io_buffer = io.BytesIO()
          np.savez_compressed(
              io_buffer, all_losses=all_losses, mean_loss=all_losses.mean(),
              ISs=inception_scores, fids=fids, kids=kids)
          f.write(io_buffer.getvalue())

      else:
        # Compute FID/KID/IS on all samples together.
        if not inceptionv3:
          inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
        else:
          inception_score = -1

        fid = tfgan.eval.frechet_classifier_distance_from_activations(
            data_pools, all_pools)
        # Hack to get tfgan KID work for eager execution.
        tf_data_pools = tf.convert_to_tensor(data_pools)
        tf_all_pools = tf.convert_to_tensor(all_pools)
        kid = tfgan.eval.kernel_classifier_distance_from_activations(
            tf_data_pools, tf_all_pools).numpy()
        del tf_data_pools, tf_all_pools

        logging.info(
            "ckpt-%d --- loss: %.6e, inception_score: %.6e, FID: %.6e, KID: %.6e",
            ckpt, all_losses.mean(), inception_score, fid, kid)

        with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                               "wb") as f:
          io_buffer = io.BytesIO()
          np.savez_compressed(
              io_buffer, all_losses=all_losses, mean_loss=all_losses.mean(),
              IS=inception_score, fid=fid, kid=kid)
          f.write(io_buffer.getvalue())
    else:
      # For host_id() != 0.
      # Use file existence to emulate synchronization across hosts.
      if hasattr(config.eval, "num_partitions"):
        assert not inceptionv3
        while not tf.io.gfile.exists(
            os.path.join(eval_dir, f"report_all_{ckpt}.npz")):
          time.sleep(1.)

      else:
        while not tf.io.gfile.exists(
            os.path.join(eval_dir, f"report_{ckpt}.npz")):
          time.sleep(1.)

    # Save eval_meta after computing IS/KID/FID to mark the end of evaluation
    # for this checkpoint.
    checkpoints.save_checkpoint(
        eval_dir,
        eval_meta,
        step=ckpt * num_rounds + r,
        keep=1,
        prefix=f"meta_{jax.host_id()}_")

    begin_round = 0

  # Remove all meta files after finishing evaluation.
  meta_files = tf.io.gfile.glob(
      os.path.join(eval_dir, f"meta_{jax.host_id()}_*"))
  for file in meta_files:
    tf.io.gfile.remove(file)

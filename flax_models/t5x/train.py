# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""This script pre-trains or fine-tunes a Transformer using the T5 data pipeline."""
from concurrent.futures import thread
import functools
import importlib
import os
from typing import Any, Mapping, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging

# Set Linen to add profiling information when constructing Modules.
# Must be set before flax imports.
# pylint:disable=g-import-not-at-top
os.environ['FLAX_PROFILE'] = 'true'
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import lax
from jax import random
from jax.interpreters.sharded_jit import sharded_jit
import jax.numpy as jnp
import ml_collections
from ml_collections import config_flags
import numpy as np
import t5
from t5.experimental.p5x import checkpoint_importer
from t5.experimental.p5x import input_pipeline
from t5.experimental.p5x import models
from t5.experimental.p5x import partitions
from t5.experimental.p5x import train_lib
import tensorflow as tf

# pylint:disable=g-long-lambda


FLAGS = flags.FLAGS
CFG = None
PyTreeDef = type(jax.tree_structure(None))
TransformerConfig = models.TransformerConfig
jax.config.parse_flags_with_absl()

flags.DEFINE_string(
    'model_dir', default=None, help='Directory to store model data.')

flags.DEFINE_string(
    'data_dir', default=None, help='Tensorflow datasets directory.')

config_flags.DEFINE_config_file(
    name='config',
    default='configs/t5_small_glue.py',
    help_string='training config file.')

ConfigDict = ml_collections.ConfigDict


def get_configs(
    config
):
  """Get train, eval, and predict model configs.

  Args:
    config: The config dict for the experiment.

  Returns:
    A triple (train_config, eval_config, predict_config).
  """
  train_config = TransformerConfig(
      vocab_size=config.vocab_size,
      output_vocab_size=config.vocab_size,
      share_embeddings=config.share_embeddings,
      logits_via_embedding=config.logits_via_embedding,
      dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
      emb_dim=config.emb_dim,
      num_heads=config.num_heads,
      num_layers=config.num_layers,
      qkv_dim=config.qkv_dim,
      mlp_dim=config.mlp_dim,
      mlp_activations=config.mlp_activations,
      position_embeddings='relative',
      relative_attention_num_buckets=config.relative_attention_num_buckets,
      relative_attention_max_distance=config.relative_attention_max_distance,
      max_len=max(config.max_input_length, config.max_target_length,
                  config.max_eval_input_length, config.max_eval_target_length),
      dropout_rate=config.dropout_rate,
      attention_dropout_rate=config.attention_dropout_rate,
      deterministic=False,
      decode=False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6))
  eval_config = train_config.replace(deterministic=True)  # pytype: disable=attribute-error
  predict_config = train_config.replace(  # pytype: disable=attribute-error
      deterministic=True,
      decode=True,
      max_decode_len=config.max_eval_target_length)

  return (train_config, eval_config, predict_config)


def get_initial_params(rng, config,
                       transformer_config,
                       optimizer_def):
  """Get the initial parameter tree."""
  input_shape = (config.batch_size, CFG.max_input_length)
  target_shape = (config.batch_size, CFG.max_target_length)
  initial_variables = models.Transformer(transformer_config).init(
      rng, jnp.ones(input_shape, jnp.float32),
      jnp.ones(target_shape, jnp.float32))
  # apply an optimizer to the parameters
  return optimizer_def.create(initial_variables['params'])


def main(argv):
  global CFG
  CFG = FLAGS.config

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Guarantee that the JAX bfloat16 extension is used rather than TF bfloat16.
  _ = np.array(jnp.array([1.0], dtype=jnp.bfloat16))

  # Use hardware RNG for bernoulli randoms in dropout mask creation.
  if CFG.hardware_rng:
    models.set_hardware_bernoulli()

  if 'module_import' in CFG and CFG.module_import:
    for module in CFG.module_import:
      importlib.import_module(module)

  if 'additional_task_cache_dirs' in CFG and CFG.additional_task_cache_dirs:
    t5.data.add_global_cache_dirs(CFG.additional_task_cache_dirs)

  num_partitions = CFG.num_partitions
  topology = train_lib.compute_multihost_topology(num_partitions)
  batch_size = CFG.batch_size
  eval_batch_size = CFG.eval_batch_size
  per_replica_set_eval_batch_size = eval_batch_size // topology.num_replica_sets
  if batch_size % topology.num_replicas:
    raise ValueError('Batch size must be divisible by the number of replicas.')

  steps_per_epoch = CFG.steps_per_epoch
  logging.info('steps per epoch: %d', steps_per_epoch)

  broadcast = functools.partial(
      train_lib.broadcast,
      num_replicas=topology.per_replica_set_num_replicas,
      num_partitions=topology.per_host_num_partitions,
      devices=topology.this_host_device_assignment)

  if jax.host_id() == 0:
    tf.io.gfile.makedirs(FLAGS.model_dir)
    tf.io.gfile.copy(FLAGS['config'].config_filename,
                     os.path.join(FLAGS.model_dir, 'config.py'),
                     overwrite=True)
    train_summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.model_dir, 'train'))
    eval_summary_writer = tensorboard.SummaryWriter(
        os.path.join(FLAGS.model_dir, 'eval'))
  else:
    train_summary_writer = None
    eval_summary_writer = None

  # Write summaries in background thread to avoid blocking on device sync
  if CFG.infeed:
    # Infeed is currently synchronous, so do it in a background thread too
    infeed_pool = thread.ThreadPoolExecutor(jax.local_device_count(), 'infeed')

  (train_ds, eval_ds), eval_cache = input_pipeline.get_datasets_and_cache(
      CFG, topology.num_replica_sets, topology.replica_set_id,
      topology.per_replica_set_host_id)

  vocab = input_pipeline.get_vocabulary(CFG.mixture_or_task_name)
  encoder = vocab.tf_tokenizer
  eos_id = vocab.tokenizer.eos_id()

  def decode_tokens(toks,
                    eos_id = eos_id,
                    max_id = 32000):
    """Decode tokens back to unicode."""
    del eos_id
    # TODO(levskaya): T5 doesn't seem to emit EOS tokens?  double check this
    # is the best decoding function or just switch to using tf_decode.
    # valid_toks = toks[:np.argmax(toks == eos_id) + 1].astype(np.int32)
    valid_toks = toks.astype(np.int32)
    valid_toks[valid_toks >= max_id] = 3
    return encoder.detokenize(valid_toks).numpy().decode('utf-8')

  logging.info('Initializing model, optimizer, and step functions.')

  train_config, eval_config, predict_config = get_configs(CFG)

  rng = random.PRNGKey(CFG.random_seed)
  rng, init_rng = random.split(rng)
  # This is used for infeed conversion from feature dict <--> tuple
  train_keys = [
      'inputs', 'targets', 'inputs_position', 'targets_position',
      'inputs_segmentation', 'targets_segmentation'
  ]
  device_train_input_shape = tuple([
      (batch_size // topology.num_replicas,
       CFG.max_input_length if 'inputs' in k else CFG.max_target_length)
      for k in train_keys
  ])

  learning_rate_fn = train_lib.create_learning_rate_scheduler(
      factors=CFG.schedule,
      base_learning_rate=CFG.learning_rate,
      warmup_steps=CFG.warmup_steps)

  # First, we only abstractly initialize the optimizer and model parameters,
  # since the parameters may not even fit in device memory!
  # TODO(jekbradbury): make optimizer_defs compare by value so it can be created
  # in get_initial_params without causing pytree incompatibility
  optimizer_def = optim.Adafactor(
      CFG.learning_rate, decay_rate=0.8, step_offset=CFG.step_offset)
  initialize_params_fn = functools.partial(
      get_initial_params,
      config=CFG,
      transformer_config=eval_config,
      optimizer_def=optimizer_def)
  optimizer = jax.eval_shape(initialize_params_fn, init_rng)
  # tuple-like pytree leaves for global_arg_shapes
  optimizer_shapes = jax.tree_map(lambda x: partitions.Spec(*x.shape),
                                  optimizer)

  # Build parameter partition annotations for preserving partitions from train
  # to eval.
  if num_partitions > 1:
    optimizer_partitions = optimizer.restore_state(
        partitions.set_partitions(num_partitions, optimizer.state_dict()))
    per_host_optimizer_partitions = optimizer.restore_state(
        partitions.set_partitions(topology.per_host_num_partitions,
                                  optimizer.state_dict()))

  # Restore unreplicated optimizer + model state from last checkpoint.
  # TODO(jekbradbury,levskaya): implement sharded native checkpoint/restore
  existing_checkpoint_found = False
  if CFG.restore_checkpoints:
    existing_checkpoint_found = train_lib.checkpoint_exists(FLAGS.model_dir)
    optimizer = checkpoints.restore_checkpoint(FLAGS.model_dir, optimizer)

  # Import a pretrained-T5 checkpoint only if we didn't import a local
  # "native" checkpoint (e.g. due to resuming a pre-empted finetuning run.)
  # TODO(jekbradbury,levskaya): implement sharded T5 checkpoint/restore
  if CFG.restore_t5_checkpoint and not existing_checkpoint_found:
    optimizer = checkpoint_importer.restore_from_t5_checkpoint(
        optimizer, CFG.restore_t5_checkpoint)

  if CFG.restore_t5_checkpoint or existing_checkpoint_found:
    if num_partitions > 1:
      # Until checkpoint/restore is sharded, the restored checkpoint is global
      # and we need to slice each sharded parameter into the chunk containing
      # only the partitions that are present on this host.
      def per_host_chunk(x, spec):
        if spec is None or spec is x:  # unsharded or not a parameter
          return x
        if spec[0] == 1:
          dim_size = x.shape[1]
        elif spec[1] == 1:
          dim_size = x.shape[0]
        else:
          raise NotImplementedError()
        chunk_size = (
            dim_size * topology.per_host_num_partitions // num_partitions)
        lower = topology.per_replica_set_host_id * chunk_size
        upper = (topology.per_replica_set_host_id + 1) * chunk_size
        if spec[0] == 1:
          return x[:, lower:upper]
        else:
          return x[lower:upper]

      optimizer = jax.tree_multimap(per_host_chunk, optimizer,
                                    optimizer_partitions)
  else:
    # If pretraining and no checkpoint imported, we jit the (sharded-) init
    # function to minimize fragmentation. We use the same pmap(sharded_jit)
    # setup as the training step/loop to initialize everything "in-place" and
    # avoid communication or OOM.
    if num_partitions > 1:
      initialize_params_fn = sharded_jit(
          initialize_params_fn,
          in_parts=None,
          local_in_parts=None,
          out_parts=optimizer_partitions,
          local_out_parts=per_host_optimizer_partitions,
          # devices=one_replica_device_assignment,
      )
      initialize_params_fn = jax.pmap(
          initialize_params_fn,
          'batch',
          in_axes=0,
          axis_size=topology.num_replicas,
          devices=topology.device_assignment)
      init_rng = broadcast(init_rng)
      optimizer = initialize_params_fn(init_rng)
      # We maintain the optimizer in unbroadcasted form (i.e. with no leading
      # replica axis). This is equivalent to the as-yet-nonexistent pmap kwarg
      # out_axes=None.
      optimizer = train_lib.unbroadcast(optimizer)
    else:
      optimizer = jax.jit(initialize_params_fn)(init_rng)

  # ---------------------------------------------------------------------------
  # Compile multidevice versions of train/eval/predict step and cache init fn.
  # ---------------------------------------------------------------------------

  # We can use either a single train-step for a host training loop:

  # train_step(optimizer, batch, prev_metrics, dropout_rng, **kwargs)
  #  --> new_optimizer, metrics, new_dropout_rng
  def p_train_step(optimizer, batch,
                   prev_metrics,
                   dropout_rng):
    return train_lib.train_step(
        optimizer,
        batch,
        prev_metrics,
        dropout_rng,
        config=train_config,
        learning_rate_fn=learning_rate_fn,
        num_microbatches=CFG.microbatches,
        label_smoothing=CFG.label_smoothing,
        z_loss=CFG.z_loss,
        use_bfloat16=CFG.use_bfloat16)

  if num_partitions > 1:
    p_train_step = sharded_jit(
        p_train_step,
        in_parts=(optimizer_partitions, None, None, None),
        local_in_parts=(per_host_optimizer_partitions, None, None, None),
        out_parts=(optimizer_partitions, None, None),
        local_out_parts=(per_host_optimizer_partitions, None, None))
  # TODO(levskaya): the in_axes spec below might be wrong, double-check.
  p_train_step = jax.pmap(
      p_train_step,
      axis_name='batch',
      in_axes=(None, 0, 0, 0),
      donate_argnums=(0,),
      global_arg_shapes=(optimizer_shapes, None, None, None),
      axis_size=topology.num_replicas,
      devices=topology.device_assignment)  # pytype: disable=wrong-arg-types

  # OR, we use an on-device loop that feeds the training step via infeed queue.
  def device_train_loop_cond(
      args
  ):
    """Stopping criterion for on-device loop."""
    _, _, _, _, step, epoch = args
    return step // steps_per_epoch == epoch

  def device_train_loop_body(
      args
  ):
    """On-device loop body."""
    optimizer, dropout_rngs, metrics, token, step, epoch = args
    # Ordering input data from infeed requires threading a symbolic token
    # through the computation.
    input_data, token = lax.infeed(
        token,
        shape=tuple(
            [jax.ShapedArray(s, jnp.int32) for s in device_train_input_shape]))
    # Rebuild input dict from infeed data tuple.
    batch = {k: v for k, v in zip(train_keys, input_data)}
    # Run the train_step function and return the loop state.
    optimizer, metrics, dropout_rngs = train_lib.train_step(
        optimizer,
        batch,
        metrics,
        dropout_rngs,
        train_config,
        learning_rate_fn,
        num_microbatches=CFG.microbatches,
        label_smoothing=CFG.label_smoothing,
        z_loss=CFG.z_loss)
    step += 1
    return optimizer, dropout_rngs, metrics, token, step, epoch

  def device_train_loop(optimizer, dropout_rngs,
                        metrics, step,
                        epoch):
    # Create symbolic token for threading infeed data.
    token = lax.create_token(step)
    # Run on-device loop.
    optimizer, dropout_rngs, metrics, _, step, _ = lax.while_loop(
        device_train_loop_cond, device_train_loop_body,
        (optimizer, dropout_rngs, metrics, token, step, epoch))
    return optimizer, dropout_rngs, metrics, step

  if num_partitions > 1:
    device_train_loop = sharded_jit(
        device_train_loop,
        in_parts=(optimizer_partitions, None, None, None, None),
        local_in_parts=(per_host_optimizer_partitions, None, None, None, None),
        out_parts=(optimizer_partitions, None, None, None),
        local_out_parts=(per_host_optimizer_partitions, None, None, None))
  p_train_epoch = jax.pmap(
      device_train_loop,
      axis_name='batch',
      in_axes=(None, 0, 0, None, None),
      donate_argnums=(0,),
      global_arg_shapes=(optimizer_shapes, None, None, None, None),
      axis_size=topology.num_replicas,
      devices=topology.device_assignment)  # pytype: disable=wrong-arg-types

  # Reduction psum for metric data.

  def p_allreduce_metrics(x):
    return lax.psum(x, axis_name='batch')

  if num_partitions > 1:
    p_allreduce_metrics = sharded_jit(
        p_allreduce_metrics,
        in_parts=None,
        local_in_parts=None,
        out_parts=None,
        local_out_parts=None,
        num_partitions=num_partitions,
        local_num_partitions=topology.per_host_num_partitions)
  p_allreduce_metrics = jax.pmap(
      p_allreduce_metrics,
      axis_name='batch',
      global_arg_shapes=None,
      axis_size=topology.num_replicas,
      devices=topology.device_assignment)

  # Training evaluation computation.

  # eval_step(params, batch, config, label_smoothing=0.0) --> metrics
  def p_eval_step(params, batch):
    return train_lib.eval_step(
        params, batch, config=eval_config, label_smoothing=CFG.label_smoothing)

  if num_partitions > 1:
    p_eval_step = sharded_jit(
        p_eval_step,
        in_parts=(optimizer_partitions.target, None),
        local_in_parts=(per_host_optimizer_partitions.target, None),
        out_parts=None,
        local_out_parts=None)
  p_eval_step = jax.pmap(
      p_eval_step,
      axis_name='batch',
      in_axes=(None, 0),
      global_arg_shapes=(optimizer_shapes.target, None),
      axis_size=topology.num_replicas,
      devices=topology.device_assignment)  # pytype: disable=wrong-arg-types

  # Fast autoregressive decoding loop.
  # For inference and model evaluation.

  # predict_step(inputs, params,
  #              eos_id, max_decode_len, config, beam_size=4) --> beam_seqs
  def p_pred_step(inputs, params):
    return train_lib.predict_step(inputs, params, eos_id,
                                  CFG.max_eval_target_length, predict_config,
                                  CFG.beam_size)

  if num_partitions > 1:
    p_pred_step = sharded_jit(
        p_pred_step,
        in_parts=(None, optimizer_partitions.target),
        local_in_parts=(None, per_host_optimizer_partitions.target),
        out_parts=None,
        local_out_parts=None)
  p_pred_step = jax.pmap(
      p_pred_step,
      axis_name='batch',
      in_axes=(0, None),
      global_arg_shapes=(None, optimizer_shapes.target),
      axis_size=topology.num_replicas,
      devices=topology.device_assignment)  # pytype: disable=wrong-arg-types

  # ---------------------------------------------------------------------------
  # Main Train Loop
  # ---------------------------------------------------------------------------

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  # There should be a unique dropout key for each replica represented on this
  # host, but the key should be the same for the same replica on other hosts.
  # Again, this is what the replica set abstraction is for.
  dropout_rngs = random.split(
      random.fold_in(rng, topology.replica_set_id),
      topology.per_replica_set_num_replicas)
  # restore step from last checkpoint
  host_step = int(optimizer.state.step)
  empty_metrics = broadcast({
      'loss': 0.0,
      'accuracy': 0.0,
      'learning_rate': 0.0,
      'denominator': 0.0
  })
  if CFG.infeed:
    # TODO(jekbradbury): support something like this for the Python-loop case
    logging.info('Precompiling training loop and moving optimizer to device.')
    optimizer, _, metrics, _ = p_train_epoch(optimizer, dropout_rngs,
                                             empty_metrics,
                                             jnp.array(0, dtype=jnp.int32), 1)
    optimizer = train_lib.unbroadcast(optimizer)
    metrics['loss'].block_until_ready()

  logging.info('Starting training loop.')

  local_devices = jax.local_devices()
  device_step = broadcast(host_step)
  first_epoch = host_step // steps_per_epoch

  # Main Loop over "epochs".
  train_iter = train_ds.as_numpy_iterator()
  for epoch in range(first_epoch, first_epoch + CFG.num_epochs):
    metrics = empty_metrics

    # NOTE: 'optimizer' is unbroadcast by construction at initialization or
    # when loading a checkpoint. It is maintained in 'unbroadcast' state to
    # enable the XLA cross-replica sharding optimization.  The broadcasting is
    # handled automatically by the pmap'd functions that use it.

    # Gather all task evaluation metrics.
    logging.info('Evaluating tasks.')
    if epoch == first_epoch + 1:
      train_lib.sync_devices()
    for task in eval_cache.tasks:
      logging.info('Evaluating task %s', task.name)
      all_predicted, all_bs = [], []
      for pred_batch in eval_cache.preprocessed_examples[task.name]:
        # Handle final odd-sized batch by padding instead of dropping it.
        input_batch, unpadded_batch_size = train_lib.pad_batch_to_size(
            pred_batch['inputs'], per_replica_set_eval_batch_size)
        all_bs.append(unpadded_batch_size)
        # Split batch dimensions for pmap.
        input_batch = jax.tree_map(
            lambda x: x.reshape(
                (topology.per_replica_set_num_replicas, -1) + x.shape[1:]),
            input_batch)
        # Run fast inference on batch.
        all_predicted.append(p_pred_step(input_batch, optimizer.target))

      # Pad out the number of batches so each host has the same number.
      max_host_batch_number = np.max(
          eval_cache.preprocessed_batch_sizes[task.name])
      batch_shortfall = max_host_batch_number - len(all_predicted)
      if batch_shortfall > 0:
        # TODO(levskaya): Fix for case of entirely empty all_predicted.
        # To make sure the cross-host barriers work, we run the program the same
        # number of times on all hosts. The results of this call is ignored, and
        # the predictions are populated with zeros instead.
        p_pred_step(input_batch, optimizer.target)  # Dummy call.
        all_predicted.extend([jnp.zeros_like(all_predicted[0])] *
                             batch_shortfall)
        all_bs.extend([0] * batch_shortfall)
      all_predicted = jnp.concatenate(all_predicted)
      all_bs = jnp.array(all_bs)

      # Collect all batches from across hosts and reverse sharding.
      all_predicted = train_lib.host_allgather(
          all_predicted, topology.num_replica_sets, topology.replica_set_id,
          topology.per_replica_set_host_id == 0)
      seqlength = all_predicted.shape[-1]
      total_examples = np.sum(
          train_lib.host_allgather(all_bs, topology.num_replica_sets,
                                   topology.replica_set_id,
                                   topology.per_replica_set_host_id == 0))
      del all_bs
      assert total_examples == len(eval_cache.examples[task.name]), (
          'Total number of batches incorrect for task %s.' % task.name)
      # De-shard the collected predicted tokens and remove padding.
      all_predicted = np.transpose(all_predicted, (1, 2, 0, 3)).reshape(
          -1, seqlength)[:total_examples]

      # We now run the post-processing and metric-fns on a single host.
      if jax.host_id() == 0:
        assert eval_summary_writer
        raw_predictions = []
        for tokens in all_predicted:
          raw_predictions.append(decode_tokens(tokens))

        # post-process predictions for metric fns
        predictions = [
            task.postprocess_fn(p, example=ex)
            for p, ex in zip(raw_predictions, eval_cache.examples[task.name])
        ]

        for metric_fn in task.metric_fns:
          scores = metric_fn(eval_cache.targets[task.name], predictions)
          for metric_name, metric_value in scores.items():
            tag = f'eval/{task.name}/{metric_name}'
            eval_summary_writer.scalar(tag, metric_value, host_step)
            logging.info('EVAL %s at step %d: %.3f', tag, host_step,
                         metric_value)
          eval_summary_writer.flush()

        # Save text samples for tensorboard.
        exemplars = ''
        for n in np.random.choice(np.arange(len(predictions)), 8):
          tgt_txt = tf.compat.as_text(
              eval_cache.examples[task.name][n]['targets_plaintext'])
          pred_txt = raw_predictions[n]
          exemplars += (f'{eval_cache.inputs[task.name][n]}\n\n'
                        f'target: {tgt_txt}\n\n'
                        f'prediction: {pred_txt}\n\n')
        eval_summary_writer.text(f'{task.name} samples', exemplars, host_step)
        eval_summary_writer.flush()

    # Take an Xprof trace after the first loop has compiled everything.
    if epoch == first_epoch + 1:
      train_lib.sync_devices()

    # For on-device loop, we launch the computation before feeding data.
    logging.info('BEGIN Train loop.')
    if CFG.infeed:
      optimizer, dropout_rngs, metrics, device_step = p_train_epoch(
          optimizer, dropout_rngs, metrics, train_lib.unbroadcast(device_step),
          epoch)
      optimizer = train_lib.unbroadcast(optimizer)

    # Epoch loop.
    while int(host_step // steps_per_epoch) == epoch:
      batch = next(train_iter)
      batch = jax.tree_map(
          lambda x: x.reshape(
              (topology.per_replica_set_num_replicas, -1) + x.shape[1:]), batch)
      # Feed the on-device training loop.
      if CFG.infeed:
        for i, device in enumerate(local_devices):
          # When using infeed to provide data to the computation, we're on our
          # own for feeding the right values to the right devices. Each device
          # should get the minibatch corresponding to its replica, a slice of
          # the larger batch corresponding to the host's replica set.
          if device.platform == 'tpu':
            device_coords = (*device.coords, device.id % 2)
          else:
            device_coords = (device.host_id, i)
          per_replica_set_device_coords = tuple(
              dc % prsm
              for dc, prsm in zip(device_coords, topology.per_replica_set_mesh))
          per_replica_set_replica_coords = tuple(
              prsdc // prm for prsdc, prm in zip(per_replica_set_device_coords,
                                                 topology.per_replica_mesh))
          per_replica_set_replica_id = 0
          for prsm, prm, prsrc in zip(topology.per_replica_set_mesh,
                                      topology.per_replica_mesh,
                                      per_replica_set_replica_coords):
            per_replica_set_replica_id = (
                per_replica_set_replica_id * prsm // prm + prsrc)
          input_tuple = tuple(
              [batch[k][per_replica_set_replica_id] for k in train_keys])
          # Safety check: infeed does not check shape or types but requires
          # them to agree with on-device spec, otherwise the queue and program
          # stalls.
          tuple_shapes = jax.tree_map(jnp.shape, input_tuple)
          tuple_dtypes = jax.tree_map(lambda x: x.dtype, input_tuple)
          assert tuple_shapes == device_train_input_shape, (
              'infeed shape error %s != %s' %
              (tuple_shapes, device_train_input_shape))
          assert tuple(set(tuple_dtypes)) == (jnp.int32,), \
              ('infeed dtype error %s not all of type %s' % (
                  tuple_dtypes, jnp.int32))
          infeed_pool.submit(
              functools.partial(device.transfer_to_infeed, input_tuple))
      # Host training loop.
      else:
        optimizer, metrics, dropout_rngs = p_train_step(optimizer, batch,
                                                        metrics, dropout_rngs)
        optimizer = train_lib.unbroadcast(optimizer)
      host_step += 1
    logging.info('END Train loop.')

    # Maybe save a checkpoint on one host.
    if (CFG.save_checkpoints and
        epoch % CFG.checkpoint_freq == CFG.checkpoint_freq - 1 and
        jax.host_id() == 0):
      checkpoints.save_checkpoint(FLAGS.model_dir, optimizer, host_step)

    # Gather training metrics.
    metrics = p_allreduce_metrics(metrics)
    metrics = jax.tree_map(lambda x: jax.device_get(x[0]), metrics)
    denominator = metrics.pop('denominator')
    summary = jax.tree_map(lambda x: x / denominator, metrics)  # pylint: disable=cell-var-from-loop
    logging.info('train in step: %s, %s', host_step, summary)
    if jax.host_id() == 0:
      assert train_summary_writer
      for key, val in summary.items():
        train_summary_writer.scalar(key, val, host_step)
      train_summary_writer.flush()

    # Gather training evaluation metrics.
    logging.info('Gathering training evaluation metrics.')
    eval_metrics = []
    eval_iter = eval_ds.as_numpy_iterator()
    for _, eval_batch in zip(range(CFG.num_eval_steps), eval_iter):
      eval_batch = jax.tree_map(
          lambda x: x.reshape(
              (topology.per_replica_set_num_replicas, -1) + x.shape[1:]),
          eval_batch)
      metrics = p_eval_step(optimizer.target, eval_batch)
      eval_metrics.append(metrics)
    # average metrics across devices
    eval_metrics = p_allreduce_metrics(eval_metrics)
    eval_metrics = common_utils.get_metrics(eval_metrics)
    # average metrics across steps
    eval_metrics = jax.tree_map(np.sum, eval_metrics)
    eval_denominator = eval_metrics.pop('denominator')
    eval_summary = jax.tree_map(
        lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
        eval_metrics)
    logging.info('eval in step: %s, %s', host_step, eval_summary)
    if jax.host_id() == 0:
      assert eval_summary_writer
      for key, val in eval_summary.items():
        eval_summary_writer.scalar(key, val, host_step)
      eval_summary_writer.flush()

  # Wait until computations are done before exiting
  logging.info('Finished.')
  train_lib.sync_devices()
  # Shut down the infeed threadpool.
  if CFG.infeed:
    infeed_pool.shutdown()


if __name__ == '__main__':
  app.run(main)

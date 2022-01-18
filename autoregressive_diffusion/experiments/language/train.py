# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Language Modeling with a Transformer.

This script trains a Transformer on the text8/enwik8 dataset.
"""

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import functools
import os
import pickle
import time

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax import linen as nn
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf

from autoregressive_diffusion.experiments.language import input_pipeline
from autoregressive_diffusion.experiments.language import language_train_state
from autoregressive_diffusion.experiments.language.architectures import arm
from autoregressive_diffusion.experiments.language.architectures import transformer
from autoregressive_diffusion.model.autoregressive_diffusion import ao_arm
from autoregressive_diffusion.model.autoregressive_diffusion import ardm_utils
from autoregressive_diffusion.model.autoregressive_diffusion import bit_ao
from autoregressive_diffusion.utils import util_fns


def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay',
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: string, factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def extensive_eval(config, test_rng, writer,
                   output_path, model, state, kl_history, test_ds, step,
                   decode_tokens):
  """This function combines all extra eval benchmarks we want to run."""
  if isinstance(model, arm.ARM):
    return

  if config.context_length > 0:
    context_shape = (jax.local_device_count(), 1,
                     config.context_length)
    context = jnp.zeros(context_shape, dtype=jnp.int32)
  else:
    context = None

  # Eval settings.
  is_first_host = jax.process_index() == 0
  num_samples = jax.local_device_count()

  rngs = jax.random.split(test_rng, 6)
  return_rng, rng_sample, rng_sample_naive, rng_sample_policy, rng1, rng2 = rngs
  policy_rngs = jax.random.split(rng1, jax.local_device_count())
  naive_rngs = jax.random.split(rng2, jax.local_device_count())
  del rng1, rng2, rngs

  # Plot loss components over time.
  if is_first_host:
    fname = f'loss_t_{step}.png'
    filename = os.path.join(output_path, 'loss_plots', fname)
    util_fns.plot_loss_components(kl_history, filename, model.num_stages)

  # Sample from the model.
  start = time.time()
  chain = model.sample(
      rng_sample, state.ema_params, num_samples, context=context)
  msg = f'Sampling took {time.time() - start:.2f} seconds'
  logging.info(msg)

  if is_first_host:
    postprocess_and_write_samples(chain, decode_tokens, writer, step)

  del chain

  # Validate and sample using naive policy.
  if model.policy_support:
    budget = 20
    nelbo_policy_naive = eval_policy(model.get_naive_policy(budget),
                                     naive_rngs,
                                     state, model, test_ds)['nelbo']
    naive_dict = {'test_nelbo_policy_naive': nelbo_policy_naive}
    chain_naive = model.sample_with_naive_policy(
        rng_sample_naive, state.ema_params, num_samples, budget, context)
    if is_first_host:
      writer.write_scalars(step, naive_dict)
      postprocess_and_write_samples(chain_naive, decode_tokens, writer, step)

    del chain_naive

  # Val optimal policies.
  if model.policy_support:
    # Check 25, 50 & 100 steps, just because they are interesting to see.
    budgets = [20, 50, 100]

    # Compute policies and costs.
    start = time.time()
    policies, costs = model.compute_policies_and_costs(kl_history[-1], budgets)
    msg = f'Computing policy mats took {time.time() - start:.2f} secs'
    logging.info(msg)

    # Evaluate policy for budget 20 & 50, okay to have the same rng.
    nelbo_policy_20 = eval_policy(policies[0], policy_rngs, state, model,
                                  test_ds)['nelbo']
    nelbo_policy_50 = eval_policy(policies[0], policy_rngs, state, model,
                                  test_ds)['nelbo']
    metric_dict = {'test_nelbo_policy_20': nelbo_policy_20,
                   'test_nelbo_policy_50': nelbo_policy_50}
    budget_results_train = {
        f'train_nelbo_steps_{b}': c
        for b, c in zip(budgets, costs)
    }
    metric_dict.update(budget_results_train)
    if jax.process_index() == 0:
      writer.write_scalars(step, metric_dict)

    # Sample with policy of budget 50.
    chain_policy = model.sample_with_policy(
        rng_sample_policy, state.ema_params, num_samples, policies[1], context)

    if is_first_host:
      postprocess_and_write_samples(chain_policy, decode_tokens, writer, step)

    del chain_policy, policies, costs
  return return_rng


# Primary training / eval / decode step functions.
# -----------------------------------------------------------------------------
def train_step(state,
               batch,
               model,
               learning_rate_fn,
               clip_grad,
               ema_momentum,
               rng=None):
  """Perform a single training step."""
  inputs = batch['inputs']
  context = batch['context'] if model.config.context_length > 0 else None

  step = state.step
  rng_step = jax.random.fold_in(rng, step)

  def loss_fn(params):
    """loss function used for training."""
    elbo, elbo_per_t, nce, t = model.elbo(
        rng_step, params, inputs, train=True, context=context)
    nelbo = -elbo.mean()
    ce = -nce.mean()

    model_type = model.config.model
    if model_type == 'standard_arm' or model_type == 'permute_arm':
      # ARMs do not have an additional CE term. (The elbo is the CE term).
      loss = nelbo

    elif model.config.ce_term > 0:
      loss = nelbo + model.config.ce_term * ce
    else:
      loss = nelbo

    return loss, (elbo, elbo_per_t, nce, t)
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (elbo, elbo_per_t, ce, t)), grad = grad_fn(state.params)

  grad = jax.lax.pmean(grad, 'batch')
  if clip_grad > 0:
    grad, norm = util_fns.clip_by_global_norm(grad, clip_grad)
  new_state = state.apply_gradients(
      grads=grad,
      lr=lr,
      ema_momentum=ema_momentum)

  metrics = {
      'loss': jax.lax.pmean(loss, axis_name='batch'),
      'nelbo': jax.lax.pmean(-elbo, axis_name='batch'),
      'ce': jax.lax.pmean(-ce, axis_name='batch'),
      'learning_rate': lr,
      'grad_norm': norm,
      'nelbo_per_t_batch': jax.lax.all_gather(-elbo_per_t, axis_name='batch'),
      't_batch': jax.lax.all_gather(t, axis_name='batch'),
  }

  return new_state, metrics


def eval_step(rng, params, batch, model):
  """Calculate evaluation metrics on a batch."""

  rng, rng_return = jax.random.split(rng)

  inputs = batch['inputs']
  context = batch['context'] if model.config.context_length > 0 else None

  elbo, _, nce, _ = model.elbo(
      rng, params, inputs, train=False, context=context)

  nelbo = -elbo.mean()
  ce = -nce.mean()

  metrics = {
      'nelbo': jax.lax.pmean(nelbo, axis_name='batch'),
      'ce': jax.lax.pmean(ce, axis_name='batch')
  }
  return metrics, rng_return


def evaluate(p_eval_step, params, eval_ds, rng):
  """Evaluate the target and return a dictionary with the metrics."""
  logging.info('Gathering evaluation metrics.')
  eval_metrics = []

  for eval_batch in eval_ds:
    eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
    eval_batch = common_utils.shard(eval_batch)
    metrics, rng = p_eval_step(rng, params, eval_batch)
    eval_metrics.append(metrics)
  eval_metrics = common_utils.get_metrics(eval_metrics)
  eval_summary = jax.tree_map(np.mean, eval_metrics)
  return eval_summary, rng


# The axes that are broadcasted are the in- and output rng key ones, and the
# model, and the policy. The rng is the first arg, and the last return value.
@functools.partial(
    jax.pmap,
    static_broadcasted_argnums=(3,),
    in_axes=(0, 0, 0, None, None),
    out_axes=(0, 0),
    axis_name='batch')
def eval_step_policy(rng, batch, state, model, policy):
  """Eval a single step."""
  inputs = batch['inputs']
  context = batch['context'] if model.config.context_length > 0 else None
  rng_return, rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
  elbo, _, nce, _ = model.elbo_with_policy(
      rng, state.ema_params, inputs, policy=policy, train=False,
      context=context)
  metrics = {
      'nelbo': jax.lax.pmean(-elbo, axis_name='batch'),
      'ce': jax.lax.pmean(-nce, axis_name='batch')
  }
  return metrics, rng_return


def eval_policy(policy, rng, state, model, test_ds):
  """Evaluate the target with policy and return a dictionary with the metrics."""
  eval_metrics = []

  for eval_batch in test_ds:
    eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
    eval_batch = common_utils.shard(eval_batch)
    metrics, rng = eval_step_policy(rng, eval_batch, state, model, policy)

    # Better to leave metrics on device, and off-load after finishing epoch.
    eval_metrics.append(metrics)

  eval_metrics = common_utils.get_metrics(eval_metrics)
  eval_summary = jax.tree_map(np.mean, eval_metrics)
  return eval_summary


def postprocess_and_write_samples(chain, decode_tokens, writer, step):
  assert chain.shape[-1] == 1
  chain = chain[Ellipsis, 0]

  print_language_sample_chain(chain, decode_tokens)

  exemplars = decode_tokens(chain[-1, 0])
  writer.write_texts(step, {'samples': exemplars})


def generate_prediction(sample_rng, config, model, state, writer,
                        decode_tokens, step):
  """Takes samples from the model."""
  if config.context_length > 0:
    context_shape = (jax.local_device_count(), 1,
                     config.context_length)
    context = jnp.zeros(context_shape, dtype=jnp.int32)
  else:
    context = None

  # Sampling from the model.
  chain = model.sample(
      jax.random.fold_in(sample_rng, step),
      state.ema_params,
      batch_size=jax.local_device_count(),
      context=context)

  postprocess_and_write_samples(chain, decode_tokens, writer, step)


def model_setup(init_rng, config):
  """Sets up the model and initializes params."""
  if config.kernel_init == 'kaiming':
    kernel_init = nn.initializers.kaiming_uniform
  elif config.kernel_init == 'xavier':
    kernel_init = nn.initializers.xavier_uniform
  else:
    raise ValueError

  def get_architecture(num_input_classes, num_output_channels, num_steps,
                       is_causal=False):
    transformer_config = transformer.TransformerConfig(
        vocab_size=num_input_classes,
        output_vocab_size=num_output_channels,
        max_time=num_steps,
        dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        qkv_dim=config.qkv_dim,
        mlp_dim=config.mlp_dim,
        max_len=config.seq_length,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        context_length=config.context_length,
        is_causal=is_causal,
        kernel_init=kernel_init(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    return transformer.TransformerLM(transformer_config)

  config.per_device_batch_size = config.batch_size // jax.process_count()
  input_shape = (config.per_device_batch_size, *config.data_shape)

  if config.model == 'standard_arm':
    model = arm.ARM.create(
        config, get_architecture, random_order=False)
  elif config.model == 'permute_arm':
    model = arm.ARM.create(
        config, get_architecture, random_order=True)
  elif config.model == 'ao_arm':
    config.output_distribution = 'softmax'
    model = ao_arm.ArbitraryOrderARM.create(
        config, get_architecture, absorbing_state=config.num_classes)
  elif config.model == 'bit_ao':
    model = bit_ao.BitUpscaleAutoregressiveDiffusion.create(
        config, get_architecture)
  else:
    raise ValueError(f'Unknown model {config.model}')

  @functools.partial(jax.jit, backend='cpu')
  def init():
    tmp_x = jnp.ones(input_shape, jnp.int32)
    tmp_t = jnp.ones(input_shape[0], jnp.int32)
    if config.context_length > 0:
      context_shape = (config.per_device_batch_size, config.context_length)
      initial_variables = model.init_architecture(
          init_rng, tmp_x, tmp_t, context=jnp.ones(context_shape, jnp.int32))
    else:
      initial_variables = model.init_architecture(
          init_rng, tmp_x, tmp_t)
    return initial_variables

  logging.info('Initializing neural network')
  variables = init()
  return model, variables


def print_language_sample_chain(chain, decode_tokens, keep_frames=5):
  """Prints a language chain to visualize the generation process."""
  if len(chain) > keep_frames:
    linspace = ardm_utils.integer_linspace(0, len(chain), keep_frames)
    chain = chain[linspace]

  exemplars = [decode_tokens(batch[0]) for batch in chain]
  logging.info('Text sampling chain:')
  logging.info('\n'.join(exemplars))


def train_and_evaluate(config, workdir):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  is_first_process = jax.process_index() == 0
  tf.io.gfile.makedirs(workdir)

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  train_ds, eval_ds, test_ds, encoder = input_pipeline.get_datasets(
      config)
  config.seq_length = 250
  vocab_size = int(encoder.vocab_size())
  config.num_classes = vocab_size
  config.data_shape = (config.seq_length, 1)

  logging.info('Training with vocab size %d', vocab_size)

  def decode_tokens(toks):
    return encoder.detokenize(toks)

  start_step = 0
  rng = jax.random.PRNGKey(config.seed)
  rng, init_rng = jax.random.split(rng)
  config.per_device_batch_size = config.batch_size // jax.process_count()

  logging.info('Initializing model, optimizer, and step functions.')
  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  model, initial_variables = model_setup(init_rng, config)

  # Instead of passing the optimizer fns directly, we use a fn that returns
  # the optimizer given a learning rate.
  def tx_fn(lr):
    return optax.adamw(
        lr, b1=0.9, b2=0.99, eps=1e-08, eps_root=0.0,
        weight_decay=config.weight_decay)

  state = language_train_state.TrainState.create(
      params=initial_variables['params'], tx_fn=tx_fn)

  # We access model params only from state below via state.params.
  del initial_variables

  if config.restore_checkpoints:
    # Restore unreplicated model state from last checkpoint.
    state = checkpoints.restore_checkpoint(workdir, state)
    # Grab last step.
    start_step = int(state.step)

  writer = metric_writers.create_default_writer(
      workdir, just_logging=not is_first_process)
  if start_step == 0:
    config_dict = dict(config)
    writer.write_hparams(config_dict)

  if is_first_process and start_step == 0:
    # Dump config file to work dir for easy model loading.
    config_path = os.path.join(workdir, 'config')
    with tf.io.gfile.GFile(config_path, 'wb') as fp:
      pickle.dump(config, fp)

  print('Using state', type(state))
  # Replicate state.
  state = jax_utils.replicate(state)

  learning_rate_fn = create_learning_rate_scheduler(
      factors=config.lr_factors,
      base_learning_rate=config.learning_rate, warmup_steps=config.warmup_steps)

  # Compile multidevice versions of train/eval/predict step fn.
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          learning_rate_fn=learning_rate_fn,
          clip_grad=config.clip_grad,
          ema_momentum=config.get('ema_momentum', 0.999)),
      axis_name='batch',
      in_axes=(0, 0),
      donate_argnums=(0,))
  p_eval_step = jax.pmap(
      functools.partial(
          eval_step, model=model),
      axis_name='batch')

  # Main Train Loop
  # ---------------------------------------------------------------------------

  # We init the first set of train PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  rng = jax.random.fold_in(rng, jax.process_index())
  rng1, rng2, rng3, extensive_eval_rngs, sample_rng = jax.random.split(rng, 5)
  train_rngs = jax.random.split(rng1, jax.local_device_count())
  eval_rngs = jax.random.split(rng2, jax.local_device_count())
  test_rngs = jax.random.split(rng3, jax.local_device_count())
  del rng, rng1, rng2, rng3

  logging.info('Starting training loop.')
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer)
  if is_first_process:
    hooks += [
        report_progress,
        periodic_actions.Profile(logdir=workdir, num_profile_steps=5)
    ]
  train_metrics = []

  # Iterator that does epoch-wise indefinite iteration.
  def iterate_train(train_ds):
    epoch = 1
    while True:
      msg = f'Starting epoch {epoch}'
      logging.info(msg)
      for batch in train_ds:
        yield batch
      epoch += 1

  train_iter = iterate_train(train_ds)

  kl_tracker_train = util_fns.KLTracker(num_steps=model.num_steps)
  kl_history = []

  with metric_writers.ensure_flushes(writer):
    step = start_step
    for step in range(start_step, config.num_train_steps):
      is_last_step = step == config.num_train_steps - 1

      # Shard data to devices and do a training step.
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        batch = common_utils.shard(jax.tree_map(np.asarray, next(train_iter)))
        state, metrics = p_train_step(
            state, batch, rng=train_rngs)
        train_metrics.append(metrics)

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
      for h in hooks:
        h(step)

      # Periodic metric handling.
      if step > 0 and (step % config.eval_every_steps == 0 or is_last_step):
        with report_progress.timed('training_metrics'):
          logging.info('Gathering training metrics.')
          train_metrics = common_utils.get_metrics(train_metrics)

          # First handle loss terms per step.
          t_batch = train_metrics.pop('t_batch')
          nelbo_per_t_batch = train_metrics.pop('nelbo_per_t_batch')
          kl_tracker_train.update(
              t_batch.reshape(-1), nelbo_per_t_batch.reshape(-1))
          kl_values = kl_tracker_train.get_kl_per_t()
          kl_history.append(np.array(kl_values))
          kl_history = kl_history[-100:]  # Keep last 100 items only.

          # Handle remaining `standard` metrics
          summary = jax.tree_map(jnp.mean, train_metrics)
          summary = {'train_' + k: v for k, v in summary.items()}
          writer.write_scalars(step, summary)
          train_metrics = []

        with report_progress.timed('eval'):
          eval_results, eval_rngs = evaluate(
              p_eval_step=p_eval_step,
              params=state.ema_params,
              eval_ds=eval_ds,
              rng=eval_rngs)
          writer.write_scalars(
              step, {'eval_' + k: v for k, v in eval_results.items()})

          test_results, test_rngs = evaluate(
              p_eval_step=p_eval_step,
              params=state.ema_params,
              eval_ds=test_ds,
              rng=test_rngs)
          writer.write_scalars(
              step, {'test_' + k: v for k, v in test_results.items()})

        if step == 1000 or (step > 0 and
                            step % config.detailed_eval_every_steps == 0):
          if is_first_process:
            loss_components_path = os.path.join(workdir, 'loss_components')
            with tf.io.gfile.GFile(loss_components_path, 'wb') as fp:
              pickle.dump(kl_history[-1], fp)

          extensive_eval_rngs = extensive_eval(
              config, extensive_eval_rngs, writer, workdir,
              model, state, kl_history, test_ds, step,
              decode_tokens)

        with report_progress.timed('generate_text'):
          generate_prediction(sample_rng, config, model, state, writer,
                              decode_tokens, step)

      # Save a checkpoint on one host after every checkpoint_freq steps.
      save_checkpoint = (
          step > 0 and
          (step % config.checkpoint_every_steps == 0 or is_last_step))
      if config.save_checkpoints and save_checkpoint and is_first_process:
        with report_progress.timed('checkpoint'):
          checkpoints.save_checkpoint(workdir, jax_utils.unreplicate(state),
                                      step, overwrite=True)

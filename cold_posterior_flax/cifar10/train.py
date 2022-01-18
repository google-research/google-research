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

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train CIFAR10 using either SGMCMC Bayesian inference or SGD MAP."""

import ast
import functools

from absl import logging
from clu import metric_writers
from clu import parameter_overview
from flax import jax_utils
from flax import optim
from flax import serialization
import flax.deprecated.nn
from flax.training import common_utils
import jax
from jax import random
import jax.experimental.optimizers
import jax.nn
import jax.numpy as jnp
import ml_collections

from cold_posterior_flax.cifar10 import input_pipeline
from cold_posterior_flax.cifar10 import schedulers
from cold_posterior_flax.cifar10 import train_functions
from cold_posterior_flax.cifar10.models import new_initializers
from cold_posterior_flax.cifar10.models import new_regularizers
from cold_posterior_flax.cifar10.models import wideresnet
from cold_posterior_flax.cifar10.sample import sym_euler_sgmcmc


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def create_model(prng_key, batch_size, image_size, model_def):
  input_shape = (batch_size, image_size, image_size, 3)
  with flax.deprecated.nn.stateful() as init_state:
    with flax.deprecated.nn.stochastic(jax.random.PRNGKey(0)):
      _, model = model_def.create_by_shape(prng_key,
                                           [(input_shape, jnp.float32)])
  return model, init_state


def create_optimizer(config, model, learning_rate, train_size, sampler_rng):
  """Create optimizer definition based on config flags."""
  if config.optimizer == 'adam':
    optimizer_def = optim.Adam(
        learning_rate=learning_rate, beta1=config.momentum)
  elif config.optimizer == 'momentum':
    optimizer_def = optim.Momentum(
        learning_rate=learning_rate, beta=config.momentum)
  elif config.optimizer == 'sym_euler':
    optimizer_def = sym_euler_sgmcmc.SymEulerSGMCMC(
        train_size,
        sampler_rng,
        learning_rate=learning_rate,
        beta=config.momentum,
        temperature=config.base_temp,
        step_size_factor=1.)
  else:
    raise ValueError('Invalid value %s for config.optimizer.' %
                     config.optimizer)

  if config.weight_norm == 'none':
    pass
  elif config.weight_norm == 'learned':
    optimizer_def = optim.WeightNorm(optimizer_def)
  elif config.weight_norm in ['fixed', 'ws_sqrt', 'learned_b', 'ws']:
    # Applied in layers directly.
    pass
  else:
    raise ValueError('Invalid value %s for config.weight_norm.' %
                     config.weight_norm)

  optimizer = optimizer_def.create(model)

  if not config.debug_run:
    optimizer = optimizer.replicate()
  return optimizer


def make_training_steps(config, learning_rate_fn, l2_reg, train_size,
                        temperature_fn, step_size_fn):
  """Build the training step functions."""

  def loss_fn(model, state, batch, prng_key):
    """Loss function used for training."""
    with flax.deprecated.nn.stateful(state) as new_state:
      with flax.deprecated.nn.stochastic(prng_key):
        logits, model_penalty, summaries = model(batch['image'])
    ce_loss = train_functions.cross_entropy_loss(logits, batch['label'])

    # l2_reg not used when using prior
    no_prior = (
        config.kernel_prior == 'none' and config.bias_prior == 'none' and
        config.kernel_prior == 'none')
    assert l2_reg == 0 or no_prior, 'Either set priors or l2_reg > 0, not both.'
    weight_penalty_params = jax.tree_leaves(model.params)
    weight_l2 = sum(
        [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
    weight_penalty = l2_reg * 0.5 * weight_l2

    kernels = optim.ModelParamTraversal(lambda n, _: n.endswith('kernel'))
    kernel_prior_penalty = sum(map(kernel_prior(), kernels.iterate(model)))

    scales = optim.ModelParamTraversal(
        lambda n, _: n.endswith('scale') or n.endswith('gamma'))
    scale_prior_penalty = sum(map(scale_prior(), scales.iterate(model)))

    biases = optim.ModelParamTraversal(
        lambda n, _: any(map(n.endswith, ['bias', 'rebias', 'beta'])))
    bias_prior_penalty = sum(map(bias_prior(), biases.iterate(model)))
    prior_penalty = kernel_prior_penalty + bias_prior_penalty + scale_prior_penalty
    prior_penalty *= config.prior_reg

    loss = (
        ce_loss + weight_penalty + config.std_penalty_mult * model_penalty +
        prior_penalty)

    return loss, (new_state, logits, prior_penalty, model_penalty,
                  weight_penalty, summaries)

  def kernel_prior():
    if config.kernel_prior == 'he_normal':
      return functools.partial(
          new_regularizers.he_normal_prior_regularizer,
          scale=config.kernel_prior_scale / train_size)
    if config.kernel_prior == 'normal':
      return functools.partial(
          new_regularizers.normal_prior_regularizer,
          scale=config.kernel_prior_scale / train_size)
    elif config.kernel_prior == 'none':
      return lambda x: 0.
    else:
      raise ValueError('Invalid value "%s" for config.kernel_prior' %
                       config.kernel_prior)

  def bias_prior():
    if config.bias_prior == 'normal':
      return functools.partial(
          new_regularizers.normal_prior_regularizer,
          scale=config.bias_prior_scale / train_size)
    elif config.bias_prior == 'none':
      return lambda x: 0.
    else:
      raise ValueError('Invalid value "%s" for config.bias_prior' %
                       config.bias_prior)

  def scale_prior():
    if config.scale_prior == 'normal':
      if config.activation_f in [
          'bias_scale_relu_norm', 'bias_scale_SELU_norm'
      ]:
        if config.softplus_scale:
          mean = new_initializers.inv_softplus(1.0)
        else:
          mean = 1.0
        return functools.partial(
            new_regularizers.normal_prior_regularizer,
            scale=config.scale_prior_scale / train_size,
            mean=mean)
      else:
        return functools.partial(
            new_regularizers.normal_prior_regularizer,
            scale=config.scale_prior_scale / train_size,
            mean=1.0)
    elif config.scale_prior == 'none':
      return lambda x: 0.
    else:
      raise ValueError('Invalid value "%s" for config.scale_prior' %
                       config.scale_prior)

  def train_step(
      optimizer,
      state,
      batch,
      prng_key,
      opt_rng,
  ):
    step = optimizer.state.step
    lr = learning_rate_fn(step)
    temp = temperature_fn(step)
    step_size_factor = step_size_fn(step)

    grad_fn = jax.value_and_grad(
        functools.partial(loss_fn, state=state, batch=batch, prng_key=prng_key),
        has_aux=True)

    (loss, (new_state, logits, prior_penalty, model_penalty, weight_penalty,
            summaries)), grad = grad_fn(optimizer.target)
    grad = train_functions.pmean(grad, config, 'batch')

    metrics = train_functions.compute_metrics(config, logits, batch['label'])
    opt_kwargs = {}

    if config.optimizer in ['sym_euler']:
      opt_kwargs['temperature'] = temp
      opt_kwargs['step_size_factor'] = step_size_factor
      # NOTE: ignoring lr in lieu of step_size_factor.
      metrics['temperature'] = temp
      metrics['step_factor'] = step_size_factor
    else:
      opt_kwargs['learning_rate'] = lr
      metrics['learning_rate'] = lr

    grad = jax.experimental.optimizers.clip_grads(grad,
                                                  config.gradient_clipping)
    with flax.deprecated.nn.stochastic(opt_rng):
      new_optimizer = optimizer.apply_gradient(grad, **opt_kwargs)

    metrics['cum_loss'] = loss
    metrics['prior_penalty'] = prior_penalty
    metrics['model_penalty'] = model_penalty
    metrics['weight_penalty'] = weight_penalty
    metrics.update(summaries)
    return new_optimizer, new_state, metrics

  def update_grad_vars(optimizer, state, batch, prng_key, values):
    """Computes gradient variances for the preconditioner."""
    grad_fn = jax.value_and_grad(
        functools.partial(loss_fn, state=state, batch=batch, prng_key=prng_key),
        has_aux=True)

    _, grad = grad_fn(optimizer.target)
    grad = train_functions.pmean(grad, config, 'batch')

    values = jax.tree_multimap(lambda v, g: v + jnp.square(g), values, grad)
    return values

  return train_step, update_grad_vars


def eval_step(model, state, batch, config):
  """Evaluation step."""
  state = train_functions.pmean(state, config)

  with flax.deprecated.nn.stateful(state, mutable=False):
    logits, penalty, summaries = model(batch['image'], train=False)
  penalty = train_functions.pmean(penalty, config, 'batch')
  metrics = {'model_penalty': penalty}
  metrics.update(
      train_functions.compute_metrics(config, logits, batch['label']))
  metrics.update(summaries)
  return logits, batch['label'], metrics


def predict_step(model, state, batch, config):
  state = train_functions.pmean(state, config)
  with flax.deprecated.nn.stateful(state, mutable=False):
    logits, _ = model(batch['image'], train=False)

  return logits, batch['label']


def train(config, model_def, device_batch_size, eval_ds, num_steps,
          steps_per_epoch, steps_per_eval, train_ds, image_size, data_source,
          workdir):
  """Train model."""

  make_lr_fn = schedulers.get_make_lr_fn(config)
  make_temp_fn = schedulers.get_make_temp_fn(config)
  make_step_size_fn = schedulers.get_make_step_size_fn(config)
  if jax.host_count() > 1:
    raise ValueError('CIFAR10 example should not be run on '
                     'more than 1 host due to preconditioner updating.')

  initial_step = 0  # TODO(basv): load from checkpoint.
  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.host_id() > 0)

  # Write config to the summary files. This makes the hyperparameters available
  # in TensorBoard and makes comparison of runs in TensorBoard easier.
  # with writer.summary_writer.as_default():
  writer.write_hparams(dict(config))

  rng = random.PRNGKey(config.seed)
  rng, opt_rng, init_key, sampler_rng = jax.random.split(rng, 4)

  base_learning_rate = config.learning_rate

  # Create the model.
  model, state = create_model(rng, device_batch_size, image_size, model_def)
  parameter_overview.log_parameter_overview(model.params)
  state = jax_utils.replicate(state)

  train_size = data_source.TRAIN_IMAGES

  with flax.deprecated.nn.stochastic(init_key):
    optimizer = create_optimizer(config, model, base_learning_rate, train_size,
                                 sampler_rng)
  del model  # Don't keep a copy of the initial model.

  # Learning rate schedule
  learning_rate_fn = make_lr_fn(base_learning_rate, steps_per_epoch)
  temperature_fn = make_temp_fn(config.base_temp, steps_per_epoch)
  step_size_fn = make_step_size_fn(steps_per_epoch)

  p_eval_step, _, p_train_step, p_update_grad_vars = make_step_functions(
      config, config.l2_reg, learning_rate_fn, train_size, temperature_fn,
      step_size_fn)

  # Create dataset batch iterators.
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)

  # Gather metrics.
  train_metrics = []
  epoch = 0

  # Ensemble.
  ensemble = []
  ensemble_logits = []
  ensemble_labels = []
  ensemble_probs = []

  def ensemble_add_step(step):
    if config.lr_schedule == 'cosine':
      # Add if learning rate jumps up again in the next step.
      increase = step_size_fn(step) < step_size_fn(step + 1) - 1e-8
      _, temp_end = ast.literal_eval(config.temp_ramp)
      past_burn_in = step >= steps_per_epoch * temp_end
      return increase and past_burn_in

    elif config.lr_schedule == 'constant':
      if (step + 1) % steps_per_epoch == 0:
        return True
    return False

  logging.info('Starting training loop at step %d.', initial_step)

  for step in range(initial_step, num_steps):
    if config.optimizer in ['sym_euler'] and (step) % steps_per_epoch == 0:
      optimizer, rng = update_preconditioner(config, optimizer,
                                             p_update_grad_vars, rng, state,
                                             train_iter)
    # Generate a PRNG key that will be rolled into the batch
    step_key = jax.random.fold_in(rng, step)
    opt_step_rng = jax.random.fold_in(opt_rng, step)

    # Load and shard the TF batch
    batch = next(train_iter)
    batch = input_pipeline.load_and_shard_tf_batch(config, batch)
    if not config.debug_run:
      # Shard the step PRNG key
      # Don't shard the optimizer rng, as it should be equal among all machines.
      sharded_keys = common_utils.shard_prng_key(step_key)
    else:
      sharded_keys = step_key

    # Train step
    optimizer, state, metrics = p_train_step(optimizer, state, batch,
                                             sharded_keys, opt_step_rng)
    train_metrics.append(metrics)

    # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
    if step == initial_step:
      initial_train_metrics = get_metrics(config, train_metrics)
      train_summary = jax.tree_map(lambda x: x.mean(), initial_train_metrics)
      train_summary = {'train_' + k: v for k, v in train_summary.items()}
      logging.log(logging.INFO, 'initial metrics = %s',
                  str(train_summary.items()))

    if (step + 1) % steps_per_epoch == 0:
      # We've finished an epoch
      # Save model params/state.

      train_metrics = get_metrics(config, train_metrics)
      # Get training epoch summary for logging
      train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)

      train_summary = {'train_' + k: v for k, v in train_summary.items()}

      writer.write_scalars(epoch, train_summary)
      # Reset train metrics
      train_metrics = []

      # Evaluation
      if config.do_eval:
        eval_metrics = []
        eval_logits = []
        eval_labels = []
        for _ in range(steps_per_eval):
          eval_batch = next(eval_iter)
          # Load and shard the TF batch
          eval_batch = input_pipeline.load_and_shard_tf_batch(
              config, eval_batch)
          # Step
          logits, labels, metrics = p_eval_step(optimizer.target, state,
                                                eval_batch)
          eval_metrics.append(metrics)
          eval_logits.append(logits)
          eval_labels.append(labels)
        eval_metrics = get_metrics(config, eval_metrics)
        # Get eval epoch summary for logging
        eval_summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
        eval_summary = {'eval_' + k: v for k, v in eval_summary.items()}
        writer.write_scalars(epoch, eval_summary)

      if config.algorithm == 'sgmcmc' and ensemble_add_step(step):
        ensemble.append((serialization.to_state_dict(optimizer.target), state))

      if config.algorithm == 'sgmcmc' and ensemble_add_step(
          step) and len(ensemble) >= 1:
        # Gather predictions for this ensemble sample.
        eval_logits = jnp.concatenate(eval_logits, axis=0)
        eval_probs = jax.nn.softmax(eval_logits, axis=-1)
        eval_labels = jnp.concatenate(eval_labels, axis=0)
        # Ensure that labels are consistent between predict runs.
        if ensemble_labels:
          assert jnp.allclose(
              eval_labels,
              ensemble_labels[0]), 'Labels unordered between eval runs.'

        ensemble_logits.append(eval_logits)
        ensemble_probs.append(eval_probs)
        ensemble_labels.append(eval_labels)

        # Compute ensemble predictions over last config.ensemble_size samples.
        ensemble_last_probs = jnp.mean(
            jnp.array(ensemble_probs[-config.ensemble_size:]), axis=0)
        ensemble_metrics = train_functions.compute_metrics_probs(
            ensemble_last_probs, ensemble_labels[0])
        ensemble_summary = jax.tree_map(lambda x: x.mean(), ensemble_metrics)
        ensemble_summary = {'ens_' + k: v for k, v in ensemble_summary.items()}
        ensemble_summary['ensemble_size'] = min(config.ensemble_size,
                                                len(ensemble_probs))
        writer.write_scalars(epoch, ensemble_summary)

      epoch += 1

  return ensemble, optimizer


def update_preconditioner(config, optimizer, p_update_grad_vars, rng, state,
                          train_iter):
  """Computes preconditioner state using samples from dataloader."""
  # TODO(basv): support multiple hosts.
  values = jax.tree_map(jnp.zeros_like, optimizer.target)

  eps = config.precon_est_eps
  n_batches = config.precon_est_batches
  for _ in range(n_batches):
    rng, est_key = jax.random.split(rng)
    batch = next(train_iter)
    batch = input_pipeline.load_and_shard_tf_batch(config, batch)
    if not config.debug_run:
      # Shard the step PRNG key
      sharded_keys = common_utils.shard_prng_key(est_key)
    else:
      sharded_keys = est_key
    values = p_update_grad_vars(optimizer, state, batch, sharded_keys, values)
  stds = jax.tree_map(lambda v: jnp.sqrt(eps + (1 / n_batches) * jnp.mean(v)),
                      values)
  std_min = jnp.min(jnp.asarray(jax.tree_leaves(stds)))
  # TODO(basv): verify preconditioner estimate.
  new_precon = jax.tree_multimap(lambda s, x: jnp.ones_like(x) * (s / std_min),
                                 stds, optimizer.target)

  def convert_momentum(
      new_precon,
      state,
  ):
    """Converts momenta to new preconditioner."""
    if config.weight_norm == 'learned':
      state = state.direction_state
    old_precon = state.preconditioner
    momentum = state.momentum

    m_c = jnp.power(old_precon, -.5) * momentum
    m = jnp.power(new_precon, .5) * m_c
    return m

  # TODO(basv): verify momentum convert.
  new_momentum = jax.tree_multimap(convert_momentum, new_precon,
                                   optimizer.state.param_states)
  # TODO(basv): verify this is replaced correctly, check replicated.
  optimizer = replace_param_state(
      config, optimizer, preconditioner=new_precon, momentum=new_momentum)
  return optimizer, rng


def replace_param_state(config, optimizer, **replacements):
  """Return a new optimizer with param_states updated."""

  new_param_states = optimizer.state.param_states
  for key, val in replacements.items():
    val_flat, treedef = jax.tree_flatten(val)
    states_flat = treedef.flatten_up_to(new_param_states)
    if config.weight_norm == 'learned':
      # Accommodate nested state parameters.
      new_directions_state = [
          s.direction_state.replace(**{key: flat_val})
          for s, flat_val in zip(states_flat, val_flat)
      ]
      new_states_flat = [
          s.replace(direction_state=dir_state)
          for s, dir_state in zip(states_flat, new_directions_state)
      ]
    else:
      new_states_flat = [
          s.replace(**{key: flat_val})
          for s, flat_val in zip(states_flat, val_flat)
      ]
    new_param_states = jax.tree_unflatten(treedef, new_states_flat)
  new_state = optimizer.state.replace(param_states=new_param_states)

  optimizer = optimizer.replace(state=new_state)
  return optimizer


def get_metrics(config, metrics):
  if not config.debug_run:
    metrics = common_utils.get_metrics(metrics)
  else:
    metrics = common_utils.stack_forest(metrics)
    metrics = jax.device_get(metrics)
  return metrics


def make_step_functions(config, l2_reg, learning_rate_fn, train_size,
                        temperature_fn, step_size_fn):
  """Create pmap'ed versions of step functions."""
  train_step, update_grad_vars = make_training_steps(config, learning_rate_fn,
                                                     l2_reg, train_size,
                                                     temperature_fn,
                                                     step_size_fn)
  # pmap the train and eval functions
  if config.debug_run:

    def jitless(f):

      def jitless_f(*args, **kwargs):
        with jax.disable_jit():
          return f(*args, **kwargs)

      return jitless_f

    p_train_step = jitless(train_step)
    p_update_grad_vars = jitless(update_grad_vars)
    p_eval_step = functools.partial(eval_step, config=config)
    p_predict_step = functools.partial(predict_step, config=config)
  else:
    p_train_step = jax.pmap(
        train_step, axis_name='batch', in_axes=(0, 0, 0, 0, None))
    p_update_grad_vars = jax.pmap(update_grad_vars, axis_name='batch')
    p_eval_step = jax.pmap(
        functools.partial(eval_step, config=config), axis_name='batch')
    p_predict_step = jax.pmap(
        functools.partial(predict_step, config=config), axis_name='batch')
  return p_eval_step, p_predict_step, p_train_step, p_update_grad_vars


def get_dataset(config, batch_size, num_epochs):
  """Creates Dataset objects based on config flags."""
  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  device_batch_size = batch_size // jax.device_count()
  # Load dataset
  eval_batch_size = config.eval_batch_size
  data_source = input_pipeline.CIFAR10DataSource(
      train_batch_size=batch_size, eval_batch_size=eval_batch_size)
  train_ds = data_source.train_ds
  eval_ds = data_source.eval_ds
  # Compute steps per epoch and nb of eval steps
  steps_per_epoch = data_source.TRAIN_IMAGES // batch_size
  steps_per_eval = data_source.EVAL_IMAGES // eval_batch_size
  if config.debug_run:
    steps_per_epoch = 2
  num_steps = steps_per_epoch * num_epochs

  image_size = input_pipeline.WIDTH
  assert input_pipeline.WIDTH == input_pipeline.HEIGHT, ('Expecting square '
                                                         'input images.')
  return device_batch_size, eval_ds, num_steps, steps_per_epoch, steps_per_eval, train_ds, image_size, data_source


def train_and_evaluate(config, workdir):
  model_def = get_arch(config)

  if config.debug_run:
    logging.warning('Running a debug run on a single device, disabling JIT.')

  device_batch_size, eval_ds, num_steps, steps_per_epoch, steps_per_eval, train_ds, image_size, data_source = get_dataset(
      config, config.batch_size, config.num_epochs)
  train(config, model_def, device_batch_size, eval_ds, num_steps,
        steps_per_epoch, steps_per_eval, train_ds, image_size, data_source,
        workdir)


def get_arch(config):
  """Creates Model based on config flags."""
  if config.arch == 'wrn26_10':
    model_def = wideresnet.WideResnet.partial(
        blocks_per_group=4,
        channel_multiplier=10,
        num_outputs=10,
        activation_f=config.activation_f,
        normalization=config.normalization,
        dropout_rate=config.wrn_dropout_rate,
        std_penalty_mult=config.std_penalty_mult,
        use_residual=config.use_residual,
        bias_scale=config.bias_scale,
        weight_norm=config.weight_norm,
        softplus_scale=config.softplus_scale,
    )
  elif config.arch == 'wrn26_2':
    model_def = wideresnet.WideResnet.partial(
        blocks_per_group=4,
        channel_multiplier=2,
        num_outputs=10,
        activation_f=config.activation_f,
        normalization=config.normalization,
        dropout_rate=config.wrn_dropout_rate,
        std_penalty_mult=config.std_penalty_mult,
        use_residual=config.use_residual,
        bias_scale=config.bias_scale,
        weight_norm=config.weight_norm,
        softplus_scale=config.softplus_scale,
    )
  elif config.arch == 'wrn26_4':
    model_def = wideresnet.WideResnet.partial(
        blocks_per_group=4,
        channel_multiplier=4,
        num_outputs=10,
        activation_f=config.activation_f,
        normalization=config.normalization,
        dropout_rate=config.wrn_dropout_rate,
        std_penalty_mult=config.std_penalty_mult,
        use_residual=config.use_residual,
        bias_scale=config.bias_scale,
        weight_norm=config.weight_norm,
        softplus_scale=config.softplus_scale,
    )
  elif config.arch == 'rnv1_20':
    model_def = wideresnet.ResnetV1.partial(
        depth=20,
        num_outputs=10,
        activation_f=config.activation_f,
        normalization=config.normalization,
        dropout_rate=config.wrn_dropout_rate,
        std_penalty_mult=config.std_penalty_mult,
        use_residual=config.use_residual,
        bias_scale=config.bias_scale,
        weight_norm=config.weight_norm,
        report_metrics=config.report_metrics,
        softplus_scale=config.softplus_scale,
    )
  elif config.arch == 'rnv1_20_64':
    model_def = wideresnet.ResnetV1.partial(
        depth=20,
        num_outputs=10,
        activation_f=config.activation_f,
        normalization=config.normalization,
        dropout_rate=config.wrn_dropout_rate,
        std_penalty_mult=config.std_penalty_mult,
        use_residual=config.use_residual,
        bias_scale=config.bias_scale,
        weight_norm=config.weight_norm,
        report_metrics=config.report_metrics,
        filters=64,
        softplus_scale=config.softplus_scale,
    )
  elif config.arch == 'wrn14_2':
    model_def = wideresnet.WideResnet.partial(
        blocks_per_group=2,
        channel_multiplier=2,
        num_outputs=10,
        activation_f=config.activation_f,
        normalization=config.normalization,
        dropout_rate=config.wrn_dropout_rate,
        std_penalty_mult=config.std_penalty_mult,
        use_residual=config.use_residual,
        bias_scale=config.bias_scale,
        weight_norm=config.weight_norm,
        softplus_scale=config.softplus_scale,
    )
  elif config.arch == 'wrn8_1':
    model_def = wideresnet.WideResnet.partial(
        blocks_per_group=1,
        channel_multiplier=1,
        num_outputs=10,
        activation_f=config.activation_f,
        normalization=config.normalization,
        dropout_rate=config.wrn_dropout_rate,
        std_penalty_mult=config.std_penalty_mult,
        use_residual=config.use_residual,
        bias_scale=config.bias_scale,
        weight_norm=config.weight_norm,
        softplus_scale=config.softplus_scale,
    )
  else:
    raise ValueError('Unknown architecture {}'.format(config.arch))
  return model_def

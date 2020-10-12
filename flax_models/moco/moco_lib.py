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

"""Momentum Contrast for Unsupervised Visual Representation Learning."""

import functools
import time

from absl import logging

from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
import flax.nn
from flax.training import common_utils
from flax.training import lr_schedule

import jax
from jax import lax
import jax.nn
import jax.numpy as jnp
from flax_models.moco import imagenet_data_source


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def create_model(key, batch_size, image_size, module):
  input_shape = (batch_size, image_size, image_size, 3)
  with flax.nn.stateful() as init_state:
    with flax.nn.stochastic(jax.random.PRNGKey(0)):
      (_, _), initial_params = module.init_by_shape(
          key, [(input_shape, jnp.float32)])
      model = flax.nn.Model(module, initial_params)
  return model, init_state


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def create_linear_classifier(key, batch_size, feature_size, num_classes):
  input_shape = (batch_size, feature_size)
  module = flax.nn.Dense.partial(features=num_classes)
  with flax.nn.stateful():
    _, initial_params = module.init_by_shape(
        key, [(input_shape, jnp.float32)])
    model = flax.nn.Model(module, initial_params)
  return model


def cross_entropy_loss(logits, labels):
  log_softmax_logits = jax.nn.log_softmax(logits)
  num_classes = log_softmax_logits.shape[-1]
  one_hot_labels = common_utils.onehot(labels, num_classes)
  return -jnp.mean(jnp.sum(one_hot_labels * log_softmax_logits, axis=-1))


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  error_rate = jnp.mean(jnp.argmax(logits, -1) != labels)
  metrics = {
      'loss': loss,
      'error_rate': error_rate,
  }
  metrics = jax.lax.pmean(metrics, axis_name='batch')
  return metrics


def compute_train_moco_metrics(moco_loss_per_sample):
  metrics = {
      'moco_loss': moco_loss_per_sample.mean(),
  }
  metrics = jax.lax.pmean(metrics, axis_name='batch')
  return metrics


def normalize_embeddings(x, eps=1e-6):
  # A note for those who are interested:
  # For some reason I thought it would be beneficial to stop the gradient
  # on the L2 norm. Turns out this is a bad idea and makes the results
  # substantially worse. Don't do this.
  l2_norm = jnp.sqrt(jnp.square(x).sum(axis=1, keepdims=True))
  return x / (l2_norm + eps)


def moco_loss(emb_query, emb_key, moco_dictionary, temperature):
  """Compute MoCo loss.

  Args:
    emb_query: embedding predicted by query network
    emb_key: embedding predicted by key network
    moco_dictionary: dictionary of embeddings from prior epochs
    temperature: softmax temperature

  Returns:
    MoCo loss
  """
  # Positive logits
  # pos_logits.shape = (n_samples, 1)
  pos_logits = (emb_query * emb_key).sum(axis=1, keepdims=True) / temperature

  # Negative logits = (n_samples, n_codes)
  neg_logits = jnp.dot(emb_query, moco_dictionary.T) / temperature

  # We now want to:
  # - append pos_logits and neg_logits along axis 1
  # - compute negative log_softmax to get cross-entropy loss
  # - use the cross-entropy of the positive samples (position 0 in axis 1)
  logits = jnp.append(pos_logits, neg_logits, axis=1)
  moco_loss_per_sample = -jax.nn.log_softmax(logits)[:, 0]

  return moco_loss_per_sample


def moco_key_step(model_key, state_key, batch):
  """MoCo train step part 1; predict embeddings given key network.

  We separate our MoCo training step into two parts.
  This first part uses the key network to predict embeddings.
  The samples that are used have to be shuffled to prevent the network
  from cheating using differing batch stats between devices.
  (see https://arxiv.org/abs/1911.05722, sec 3.3)

  Args:
    model_key: key network
    state_key: batch stats and state for key network
    batch: batch of samples

  Returns:
    embeddings for samples in `batch`
  """
  # Average batch stats across devices/hosts
  state_key = jax.lax.pmean(state_key, axis_name='batch')

  # emb_key.shape = (n_samples, emb_size)
  x_key = batch['x_key']
  with flax.nn.stateful(state_key) as new_state_key:
    emb_key, _ = model_key(x_key, train=True)
  emb_key = jax.lax.stop_gradient(emb_key)
  emb_key = normalize_embeddings(emb_key)
  return emb_key, new_state_key


def moco_train_step(optimizer_query, state_query, model_key,
                    batch, moco_dictionary, n_devices,
                    moco_temperature, learning_rate_fn, l2_reg,
                    moco_momentum):
  """MoCo training step part 2.

  Given the keys generated in part 1, part 2
  uses the query network to predict embeddings for the same samples as in
  part 1.
  The MoCo loss encourages the query network to predict an
  embedding that is more similar to the corresponding key network
  embedding than to any of the embeddings in the MoCo dictionary
  (the paper uses the term dictionary).

  Args:
    optimizer_query: query network optimizer/model
    state_query: query network state / batch stats
    model_key: key network
    batch: data batch
    moco_dictionary: dictionary of embeddings from key network
    n_devices: number of devices in use
    moco_temperature: softmax temperature for computing MoCo loss
    learning_rate_fn: function fn(step) -> lr that defines learning rate
      schedule
    l2_reg: L2 regularization coefficient
    moco_momentum: MoCo key network momentum parameter

  Returns:
    (new_optimizer_query, new_state_query, metrics, model_key, emb_key_all)
      new_optimizer_query: query network optimizer and model after step
      new_state_query: query network state / batch stats after step
      metrics: MoCo training metrics
      model_key: key network model (used to update query network)
      emb_key_all: key network embeddings concatenated across devices
  """
  def loss_fn(model_query):
    """loss function used for training."""

    emb_key = batch['emb_key']
    x_query = batch['query_image']

    # Get predicted embeddings from query network
    with flax.nn.stateful(state_query) as new_state_query:
      emb_query, _ = model_query(x_query, train=True)
    emb_query = normalize_embeddings(emb_query)
    # emb_query.shape = (n_samples, emb_size)

    # Compute per-sample MoCo loss
    moco_loss_per_sample = moco_loss(emb_query, emb_key, moco_dictionary,
                                     moco_temperature)
    loss = moco_loss_per_sample.mean()

    # Apply L2 regularization
    if l2_reg > 0:
      weight_penalty_params = jax.tree_leaves(model_query.params)
      weight_l2 = sum([jnp.sum(x ** 2)
                       for x in weight_penalty_params
                       if x.ndim > 1])
      weight_penalty = l2_reg * 0.5 * weight_l2
      loss = loss + weight_penalty

    return loss, (new_state_query, moco_loss_per_sample, emb_key)

  step = optimizer_query.state.step
  lr = learning_rate_fn(step)
  new_optimizer_query, _, (new_state_query, moco_loss_per_sample,
                           emb_key) = \
      optimizer_query.optimize(loss_fn, learning_rate=lr)

  # Update key network - exponential moving average of query network
  model_key_params = jax.tree_multimap(
      lambda p_k, p_q: p_k * moco_momentum + p_q * (1.0 - moco_momentum),
      model_key.params, new_optimizer_query.target.params
  )
  model_key = model_key.replace(params=model_key_params)

  # Compute metrics
  metrics = compute_train_moco_metrics(moco_loss_per_sample)
  metrics['learning_rate'] = lr

  # In this step we use `lax.pswapaxes` to concatenate the embeddings
  # generated by the key network *across multiple hosts*
  emb_rep = [n_devices] + [1] * emb_key.ndim
  emb_key = emb_key[None, Ellipsis]
  emb_key = jnp.tile(emb_key, emb_rep)
  emb_key_all = lax.pswapaxes(emb_key, 'batch', 0)

  # Return the concatenated key embeddings
  return new_optimizer_query, new_state_query, metrics, model_key, emb_key_all


def classifier_train_step(clf_feat_optimizer, model_moco, state_moco,
                          batch, learning_rate_fn, l2_reg):
  """Linear classifier training step."""
  # Average batch stats across devices/hosts
  state_moco = jax.lax.pmean(state_moco, axis_name='batch')

  # Get data from batch
  sup_x = batch['image']

  # Predict features (ignore embeddings)
  with flax.nn.stateful(state_moco, mutable=False):
    _, features = model_moco(sup_x, train=False)
  features = jax.lax.stop_gradient(features)

  def features_loss_fn(model_clf):
    """loss function used for training."""
    logits = model_clf(features)
    loss = cross_entropy_loss(logits, batch['label'])

    if l2_reg > 0:
      weight_penalty_params = jax.tree_leaves(model_clf.params)
      weight_l2 = sum([jnp.sum(x ** 2)
                       for x in weight_penalty_params
                       if x.ndim > 1])
      weight_penalty = l2_reg * 0.5 * weight_l2
      loss = loss + weight_penalty
    return loss, (logits,)

  # Feature classifier
  feat_step = clf_feat_optimizer.state.step
  feat_lr = learning_rate_fn(feat_step)
  new_clf_feat_optimizer, _, (feat_logits,) = clf_feat_optimizer.optimize(
      features_loss_fn, learning_rate=feat_lr)

  feat_metrics = compute_metrics(feat_logits, batch['label'])
  feat_metrics['learning_rate'] = feat_lr

  return new_clf_feat_optimizer, feat_metrics


def eval_step(model_moco, state_moco, feat_clf_model, batch):
  """Linear classifier evaluation step."""
  # Average batch stats across devices/hosts
  state_moco = jax.lax.pmean(state_moco, axis_name='batch')
  # Use MoCo network to predict features
  with flax.nn.stateful(state_moco, mutable=False):
    _, features = model_moco(batch['image'], train=False)
  # Use linear model to predict class logits
  feat_logits = feat_clf_model(features)
  feat_metrics = compute_metrics(feat_logits, batch['label'])
  return feat_metrics


def train(module,
          model_dir,
          batch_size,
          eval_batch_size,
          num_moco_epochs,
          num_clf_epochs,
          moco_learning_rate,
          clf_learning_rate,
          sgd_momentum=0.9,
          sgd_nesterov=True,
          make_moco_lr_fun=None,
          make_clf_lr_fun=None,
          moco_l2_reg=0.0001,
          clf_l2_reg=0.0,
          feature_size=64 * 8 * 4,
          moco_momentum=0.999,
          emb_size=128,
          moco_temperature=0.07,
          dictionary_size=65536,
          run_seed=0,
          steps_per_epoch=None,
          steps_per_eval=None):
  """Train MoCo model."""
  if make_moco_lr_fun is None:
    def make_moco_lr_fun(base_lr, steps_per_epoch):  # pylint: disable=function-redefined
      return lr_schedule.create_stepped_learning_rate_schedule(
          base_lr, steps_per_epoch, [[120, 0.1], [160, 0.01]])

  if make_clf_lr_fun is None:
    def make_clf_lr_fun(base_lr, steps_per_epoch):  # pylint: disable=function-redefined
      return lr_schedule.create_stepped_learning_rate_schedule(
          base_lr, steps_per_epoch, [[60, 0.2], [75, 0.04], [90, 0.008]])

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(model_dir)
  else:
    summary_writer = None

  #
  #
  # If using more than 1 host, warn the user
  #
  #

  if jax.host_count() > 1:
    logging.info('WARNING: the all_to_all collective used by this program is '
                 'not yet supported in multi-host environments')

  train_rng = jax.random.PRNGKey(run_seed)
  (init_moco_rng, init_clf_rng, init_dictionary_rng,
   train_rng) = jax.random.split(train_rng, num=4)

  if batch_size % jax.device_count() > 0:
    raise ValueError('Train batch size must be divisible by the number '
                     'of devices')
  if eval_batch_size % jax.device_count() > 0:
    raise ValueError('Eval batch size must be divisible by the number '
                     'of devices')
  local_batch_size = batch_size // jax.host_count()
  local_eval_batch_size = eval_batch_size // jax.host_count()
  n_devices = jax.device_count()
  n_local_devices = jax.local_device_count()

  device_batch_size = batch_size // n_devices

  image_size = 224
  data_source = imagenet_data_source.load_imagenet(
      train_batch_size=local_batch_size,
      eval_batch_size=local_eval_batch_size,
      greyscale_prob=0.1)

  n_train = data_source.n_train
  train_moco_ds = data_source.train_moco_ds
  train_clf_ds = data_source.train_clf_ds
  eval_ds = data_source.test_ds
  n_eval = data_source.n_test

  logging.info('DATA: |train|=%d, |eval|=%d', data_source.n_train, n_eval)

  if steps_per_epoch is None:
    steps_per_epoch = n_train // batch_size
  if steps_per_eval is None:
    steps_per_eval = n_eval // eval_batch_size
  num_moco_steps = steps_per_epoch * num_moco_epochs
  num_clf_steps = steps_per_epoch * num_clf_epochs

  logging.info('Loaded dataset')

  #
  # Create query model
  #
  model_query, state_query = create_model(
      init_moco_rng, device_batch_size, image_size, module)
  state_query = jax_utils.replicate(state_query)

  # Create linear classifier
  feat_model_clf = create_linear_classifier(
      init_clf_rng, device_batch_size, feature_size, data_source.n_classes)

  # Randomly initialise dictionary
  moco_dictionary = jax.random.normal(
      init_dictionary_rng, (dictionary_size, emb_size), dtype=jnp.float32)
  moco_dictionary = normalize_embeddings(moco_dictionary)
  logging.info('Built model')

  #
  # Create optimizer
  #

  optimizer_def = optim.Momentum(learning_rate=moco_learning_rate,
                                 beta=sgd_momentum, nesterov=sgd_nesterov)
  optimizer_query = optimizer_def.create(model_query)
  optimizer_query = optimizer_query.replicate()
  del model_query  # don't keep a copy of the initial model

  feat_clf_optimizer_def = optim.Momentum(
      learning_rate=clf_learning_rate, beta=sgd_momentum,
      nesterov=sgd_nesterov)
  feat_clf_optimizer = feat_clf_optimizer_def.create(feat_model_clf)
  feat_clf_optimizer = feat_clf_optimizer.replicate()
  logging.info('Built optimizer')

  #
  # Learning rate schedule
  #

  base_moco_learning_rate = moco_learning_rate * batch_size / 256.
  base_clf_learning_rate = clf_learning_rate * batch_size / 256.
  moco_learning_rate_fn = make_moco_lr_fun(
      base_moco_learning_rate, steps_per_epoch)
  clf_learning_rate_fn = make_clf_lr_fun(
      base_clf_learning_rate, steps_per_epoch)

  # The key model is a replica of the query model. Since Flax models are
  # immutable, we can start with the query model
  model_key = optimizer_query.target
  # Replicate batch stats
  state_key = jax.tree_map(lambda x: x, state_query)

  # Set up epoch and step counter
  # Load existing checkpoint if available
  moco_epoch = 1
  clf_epoch = 1
  moco_step = 0
  clf_step = 0

  #
  # Training and eval functions
  #
  p_moco_key_step = jax.pmap(
      functools.partial(moco_key_step),
      axis_name='batch')
  p_moco_train_step = jax.pmap(
      functools.partial(moco_train_step, n_devices=n_devices,
                        moco_temperature=moco_temperature,
                        learning_rate_fn=moco_learning_rate_fn,
                        l2_reg=moco_l2_reg,
                        moco_momentum=moco_momentum),
      axis_name='batch')
  p_classifier_train_step = jax.pmap(
      functools.partial(classifier_train_step,
                        learning_rate_fn=clf_learning_rate_fn,
                        l2_reg=clf_l2_reg),
      axis_name='batch')
  p_eval_step = jax.pmap(
      functools.partial(eval_step),
      axis_name='batch')

  # Create MoCo dataset batch iterator
  train_moco_it = iter(train_moco_ds)

  #
  # Training loop
  #

  logging.info('Training MoCo...')

  epoch_metrics_moco = []
  t1 = time.time()
  while moco_step < num_moco_steps:
    (train_rng, shuffle_rng) = jax.random.split(train_rng, num=2)

    batch = next(train_moco_it)
    # TF to NumPy
    batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access

    # Compute key embeddings
    # We have to shuffle the batch to prevent the network from cheating using
    # batch stats
    shuffle_forward = jax.random.shuffle(
        shuffle_rng, jnp.arange(local_batch_size))
    shuffle_backward = jnp.zeros((local_batch_size,), dtype=int)
    shuffle_backward = jax.ops.index_update(
        shuffle_backward, shuffle_forward, jnp.arange(local_batch_size))

    key_batch = dict(x_key=batch['key_image'][shuffle_forward, Ellipsis])
    key_batch_sharded = common_utils.shard(key_batch)
    emb_key, state_key = p_moco_key_step(
        model_key, state_key, key_batch_sharded)
    emb_key = emb_key.reshape((-1, emb_size))
    emb_key = emb_key[shuffle_backward, Ellipsis]

    #
    # Main MoCo training step
    #
    moco_batch = batch.copy()
    moco_batch['emb_key'] = emb_key
    sharded_moco_batch = common_utils.shard(moco_batch)

    # Repeat the MoCo dictionary across shards
    sharded_dict = jnp.repeat(moco_dictionary[None, Ellipsis], n_local_devices,
                              axis=0)

    # The main train step function is applied slightly differently in
    # multi-host environments
    optimizer_query, state_query, metrics_moco, model_key, code_batch = \
        p_moco_train_step(optimizer_query, state_query, model_key,
                          sharded_moco_batch, sharded_dict)
    code_batch = code_batch[0].reshape((-1, emb_size))

    moco_dictionary = jnp.append(
        code_batch, moco_dictionary, axis=0)[:dictionary_size]

    epoch_metrics_moco.append(metrics_moco)
    if (moco_step + 1) % steps_per_epoch == 0:
      epoch_metrics_moco = common_utils.get_metrics(epoch_metrics_moco)
      train_epoch_metrics = jax.tree_map(lambda x: x.mean(),
                                         epoch_metrics_moco)
      if summary_writer is not None:
        for key, vals in epoch_metrics_moco.items():
          tag = 'train_%s' % key
          for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, moco_step - len(vals) + i + 1)

      epoch_metrics_moco = []

      t2 = time.time()

      logging.info(
          'MoCo EPOCH %d: (took %.3fs): MoCo loss=%.6f',
          moco_epoch, t2 - t1, train_epoch_metrics['moco_loss'])

      t1 = t2

      if summary_writer is not None:
        summary_writer.flush()

      moco_epoch += 1

    moco_step += 1

  del train_moco_it

  #
  #
  # Unsupervised MoCo training complete
  # Train classifier
  #
  #

  logging.info('Training Linear Classifier...')

  train_clf_it = iter(train_clf_ds)
  eval_iter = iter(eval_ds)

  epoch_feat_metrics = []
  t1 = time.time()
  while clf_step < num_clf_steps:
    batch = next(train_clf_it)
    # TF to NumPy
    batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access
    batch = common_utils.shard(batch)

    feat_clf_optimizer, feat_metrics = p_classifier_train_step(
        feat_clf_optimizer, model_key, state_key, batch)

    epoch_feat_metrics.append(feat_metrics)
    if (clf_step + 1) % steps_per_epoch == 0:
      epoch_feat_metrics = common_utils.get_metrics(epoch_feat_metrics)
      train_epoch_feat_metrics = jax.tree_map(lambda x: x.mean(),
                                              epoch_feat_metrics)
      if summary_writer is not None:
        for key, vals in epoch_feat_metrics.items():
          tag = 'train_feat_%s' % key
          for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, clf_step - len(vals) + i + 1)

      epoch_feat_metrics = []
      eval_feat_metrics = []
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        # TF to NumPy
        eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
        # Shard across local devices
        eval_batch = common_utils.shard(eval_batch)
        feat_metrics = p_eval_step(
            model_key, state_key, feat_clf_optimizer.target, eval_batch)
        eval_feat_metrics.append(feat_metrics)
      eval_feat_metrics = common_utils.get_metrics(eval_feat_metrics)
      eval_epoch_feat_metrics = jax.tree_map(lambda x: x.mean(),
                                             eval_feat_metrics)

      t2 = time.time()

      logging.info(
          'Linear classifier EPOCH %d: (took %.3fs): TRAIN FEAT loss=%.6f, '
          'err=%.3f; EVAL FEAT loss=%.6f, err=%.3f',
          clf_epoch, t2 - t1, train_epoch_feat_metrics['loss'],
          train_epoch_feat_metrics['error_rate'] * 100.0,
          eval_epoch_feat_metrics['loss'],
          eval_epoch_feat_metrics['error_rate'] * 100.0,
      )

      t1 = t2

      if summary_writer is not None:
        summary_writer.scalar('eval_feat_loss',
                              eval_epoch_feat_metrics['loss'], clf_epoch)
        summary_writer.scalar('eval_feat_error_rate',
                              eval_epoch_feat_metrics['error_rate'], clf_epoch)
        summary_writer.flush()

      clf_epoch += 1

    clf_step += 1

  return eval_epoch_feat_metrics

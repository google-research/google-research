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

"""Standard data selection NMT runner.

This script trains a Transformer on a WMT dataset. It supports
a number of alternative training options used in various
experimental setups.
"""

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error

import functools
import os

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from data_selection.wmt import common
from data_selection.wmt import decode
from data_selection.wmt import input_pipeline
from data_selection.wmt import models
from data_selection.wmt import train_util
from data_selection.wmt.gradient_utils import tree_diff
from data_selection.wmt.gradient_utils import tree_div
from data_selection.wmt.gradient_utils import tree_dot
from data_selection.wmt.gradient_utils import tree_mult
from data_selection.wmt.gradient_utils import tree_norm

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'sample_size', default=-1,
    help='Number of examples from train dataset to keep.')

flags.DEFINE_float(
    'newscomment_sample_ratio', default=1.0,
    help='Upweight newscommentary.')

flags.adopt_module_key_flags(train_util)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  if FLAGS.jax_backend_target:
    jax.config.update('jax_xla_backend', 'tpu_driver')
    jax.config.update('jax_backend_target', FLAGS.jax_backend_target)

  # Number of local devices for this host.
  n_devices = jax.local_device_count()

  if jax.process_index() == 0:
    tf.io.gfile.makedirs(FLAGS.model_dir)

  if FLAGS.batch_size % n_devices:
    raise ValueError('Batch size must be divisible by the number of devices')

  vocab_path = FLAGS.vocab_path
  if vocab_path is None:
    vocab_path = os.path.join(FLAGS.model_dir, 'sentencepiece_model')
  tf.io.gfile.makedirs(os.path.split(vocab_path)[0])

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  if FLAGS.dynamic:
    train_ds_mgr, eval_ds, predict_ds, encoder = input_pipeline.get_dynamic_datasets(
        dataset_name=FLAGS.dataset_name,
        eval_dataset_name=FLAGS.eval_dataset_name,
        shard_idx=jax.process_index(),
        shard_count=jax.process_count(),
        data_dir=FLAGS.data_dir,
        vocab_path=FLAGS.vocab_path,
        target_vocab_size=FLAGS.vocab_size,
        batch_size=FLAGS.batch_size,
        max_length=FLAGS.max_target_length,
        max_eval_length=FLAGS.max_eval_target_length,
        paracrawl_size=FLAGS.paracrawl_size,
        is_scores_path=FLAGS.is_scores_path,
        num_buckets=FLAGS.num_data_buckets)
    if FLAGS.static:
      weights = np.array([float(w) for w in FLAGS.static.split(',')])
      assert len(weights) == FLAGS.num_data_buckets
      train_ds = train_ds_mgr.sampled_dataset(weights)
      FLAGS.dynamic = False
    else:
      init_dist = np.zeros(FLAGS.num_data_buckets)
      if FLAGS.data_selection_size < FLAGS.num_data_buckets:
        init_dist[range(FLAGS.data_selection_size)] = 1.0
        train_ds = train_ds_mgr.sampled_dataset(init_dist)
      else:
        train_ds = build_split(train_ds_mgr, 1.0)

  else:
    train_ds, eval_ds, predict_ds, encoder = input_pipeline.get_wmt_datasets(
        dataset_name=FLAGS.dataset_name,
        eval_dataset_name=FLAGS.eval_dataset_name,
        shard_idx=jax.process_index(),
        shard_count=jax.process_count(),
        data_dir=FLAGS.data_dir,
        vocab_path=vocab_path,
        target_vocab_size=FLAGS.vocab_size,
        batch_size=FLAGS.batch_size,
        max_length=FLAGS.max_target_length,
        max_eval_length=FLAGS.max_eval_target_length,
        paracrawl_size=FLAGS.paracrawl_size,
        is_scores_path=FLAGS.is_scores_path,
        num_to_keep=FLAGS.data_selection_size,
        pseudo_path=FLAGS.pseudo_path,
        repeat_count=FLAGS.repeat_count,
        newscommentary_size=FLAGS.newscommentary_size,
        split_tokenizer=FLAGS.split_tokenizer,
        sample_size=FLAGS.sample_size,
        newscomment_sample_ratio=FLAGS.newscomment_sample_ratio)

  if FLAGS.aux_eval_dataset:
    aux_datasets = []
    aux_names = FLAGS.aux_eval_dataset.split(',')
    for name in aux_names:
      _, aux_eval_ds, _, _ = input_pipeline.get_wmt_datasets(
          dataset_name=name,
          eval_dataset_name=None,
          shard_idx=jax.process_index(),
          shard_count=jax.process_count(),
          data_dir=FLAGS.data_dir,
          vocab_path=vocab_path,
          target_vocab_size=FLAGS.vocab_size,
          batch_size=FLAGS.batch_size,
          max_length=FLAGS.max_target_length,
          max_eval_length=FLAGS.max_eval_target_length,
          paracrawl_size=FLAGS.paracrawl_size,
          is_scores_path=FLAGS.is_scores_path,
          num_to_keep=FLAGS.data_selection_size,
          pseudo_path=FLAGS.pseudo_path,
          repeat_count=FLAGS.repeat_count,
          newscommentary_size=FLAGS.newscommentary_size)
      aux_datasets.append(aux_eval_ds)

  train_iter = iter(train_ds)
  vocab_size = int(encoder.vocab_size())
  eos_id = decode.EOS_ID  # Default Sentencepiece EOS token.

  def decode_tokens(toks):
    valid_toks = toks[:np.argmax(toks == eos_id) + 1].astype(np.int32)
    return encoder.detokenize(valid_toks).numpy().decode('utf-8')

  logging.info('Initializing model, optimizer, and step functions.')

  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  train_config = models.TransformerConfig(
      vocab_size=vocab_size,
      output_vocab_size=vocab_size,
      share_embeddings=FLAGS.share_embeddings,
      logits_via_embedding=FLAGS.logits_via_embedding,
      dtype=jnp.bfloat16 if FLAGS.use_bfloat16 else jnp.float32,
      emb_dim=FLAGS.emb_dim,
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      qkv_dim=FLAGS.qkv_dim,
      mlp_dim=FLAGS.mlp_dim,
      max_len=max(FLAGS.max_target_length, FLAGS.max_eval_target_length),
      dropout_rate=FLAGS.dropout_rate,
      attention_dropout_rate=FLAGS.attention_dropout_rate,
      deterministic=False,
      decode=False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6))
  eval_config = train_config.replace(deterministic=True)
  predict_config = train_config.replace(deterministic=True, decode=True)

  start_step = 0
  rng = jax.random.PRNGKey(FLAGS.random_seed)
  rng, init_rng = jax.random.split(rng)
  # It's possible that is supposed to be per device batch size
  input_shape = (FLAGS.batch_size, FLAGS.max_target_length)
  target_shape = (FLAGS.batch_size, FLAGS.max_target_length)

  m = models.Transformer(eval_config)
  initial_variables = jax.jit(m.init)(init_rng,
                                      jnp.ones(input_shape, jnp.float32),
                                      jnp.ones(target_shape, jnp.float32))

  # apply an optimizer to this tree
  optimizer_def = optim.Adam(
      FLAGS.learning_rate,
      beta1=0.9,
      beta2=0.98,
      eps=1e-9,
      weight_decay=FLAGS.weight_decay)
  optimizer = optimizer_def.create(initial_variables['params'])

  # We access model params only from optimizer below via optimizer.target.
  del initial_variables

  if FLAGS.restore_checkpoints:
    logging.info('Restoring checkpoint.')
    # If we have a pretrained model, use that. Else, just continue where leftoff
    model_path = FLAGS.pretrained_model_dir if FLAGS.pretrained_model_dir else FLAGS.model_dir
    optimizer = checkpoints.restore_checkpoint(model_path, optimizer)
    # Grab last step.
    start_step = int(optimizer.state.step)

  if FLAGS.adapter != train_util.NONE:
    adapter = optim.ModelParamTraversal(lambda path, _: FLAGS.adapter in path)
    optimizer = optimizer_def.create(optimizer.target, focus=adapter)

  writer = metric_writers.create_default_writer(
      FLAGS.model_dir, just_logging=jax.process_index() > 0)

  flag_key = [k for k in FLAGS.flags_by_module_dict().keys() if 'wmt.par' in k
             ]
  if flag_key:
    flag_key = flag_key[0]
    local_flags = {
        f.name: f.value for f in FLAGS.flags_by_module_dict()[flag_key]
    }
    writer.write_hparams(local_flags)

  # Replicate optimizer.
  optimizer = jax_utils.replicate(optimizer)

  if FLAGS.adapter != train_util.NONE:
    learning_rate_fn = common.create_learning_rate_scheduler(
        factors='constant',
        base_learning_rate=FLAGS.learning_rate,
        warmup_steps=FLAGS.warmup_steps)
  else:
    learning_rate_fn = common.create_learning_rate_scheduler(
        base_learning_rate=FLAGS.learning_rate, warmup_steps=FLAGS.warmup_steps,
        steps_per_cycle=FLAGS.steps_per_cycle, init_step=start_step,
        finetune_lr=FLAGS.finetune_lr)

  # compile multidevice versions of train/eval/predict step and cache init fn.
  p_train_step = jax.pmap(
      functools.partial(
          train_util.train_step,
          config=train_config,
          learning_rate_fn=learning_rate_fn,
          label_smoothing=FLAGS.label_smoothing),
      axis_name='batch',
      donate_argnums=(0,))  # pytype: disable=wrong-arg-types
  p_eval_step = jax.pmap(
      functools.partial(train_util.eval_step, config=eval_config),
      axis_name='batch')
  p_init_cache = jax.pmap(
      functools.partial(
          train_util.initialize_cache,
          max_decode_len=FLAGS.max_predict_length,
          config=predict_config),
      axis_name='batch')
  p_pred_step = jax.pmap(
      functools.partial(
          train_util.predict_step,
          config=predict_config,
          beam_size=FLAGS.beam_size),
      axis_name='batch',
      static_broadcasted_argnums=(3, 4))  # eos token, max_length are constant

  p_get_diag_grads = jax.pmap(
      functools.partial(
          train_util.get_diag_grads,
          config=eval_config),
      axis_name='batch')

  p_get_bucket_score = jax.pmap(
      functools.partial(
          get_diag_score,
          strategy=FLAGS.strategy),
      axis_name='batch')

  # Main Train Loop
  # ---------------------------------------------------------------------------

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap"d training update for performance.
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  del rng

  logging.info('Starting training loop.')
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=FLAGS.num_train_steps, writer=writer)
  if jax.process_index() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(logdir=FLAGS.model_dir, num_profile_steps=5)
    ]
  train_metrics = []
  total_steps = start_step + FLAGS.num_train_steps
  if FLAGS.eval_only:
    total_steps = start_step + 1
  best_eval_loss = 1000
  curr_eval_loss = 1000
  with metric_writers.ensure_flushes(writer):
    for step in range(start_step, total_steps):
      is_last_step = step == total_steps - 1

      if FLAGS.dynamic and ((step - start_step) % FLAGS.resample_freq == 0):
        # Dynamic macro: use gradient alignment to score different ratios
        # of top k vs bottom N-k bins
        if FLAGS.macro:
          train_iter = get_macro_distribution(p_get_diag_grads,
                                              p_get_bucket_score, aux_eval_ds,
                                              train_ds_mgr, optimizer, eval_ds)
        else:
          # Use gradient alignment to score bins
          # take the top k bins and sample uniformly from them.
          raw_distribution = get_new_distribution(p_get_diag_grads,
                                                  p_get_bucket_score,
                                                  aux_eval_ds, train_ds_mgr,
                                                  optimizer,
                                                  eval_ds)
          logging.info(raw_distribution)
          selected = np.argsort(
              raw_distribution)[::-1][:FLAGS.data_selection_size]
          new_distribution = np.zeros(100)
          new_distribution[selected] = 1.0
          logging.info(new_distribution)
          train_ds = train_ds_mgr.sampled_dataset(new_distribution)
          train_iter = iter(train_ds)

      # Shard data to devices and do a training step.
      if not FLAGS.eval_only:
        logging.info('Doing Training.')
        with jax.profiler.StepTraceAnnotation('train', step_num=step):
          try:
            batch = common_utils.shard(
                jax.tree.map(np.asarray, next(train_iter)))
            optimizer, metrics = p_train_step(
                optimizer, batch, dropout_rng=dropout_rngs)
            train_metrics.append(metrics)
          except StopIteration:
            is_last_step = True

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
      for h in hooks:
        h(step)

      # Periodic metric handling.
      if (step - start_step) % FLAGS.eval_frequency == 0 or is_last_step:
        if not FLAGS.eval_only:
          with report_progress.timed('training_metrics'):
            logging.info('Gathering training metrics.')
            train_metrics = common_utils.get_metrics(train_metrics)
            lr = train_metrics.pop('learning_rate').mean()
            metrics_sums = jax.tree.map(jnp.sum, train_metrics)
            denominator = metrics_sums.pop('denominator')
            summary = jax.tree.map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
            summary['learning_rate'] = lr
            summary = {'train_' + k: v for k, v in summary.items()}
            writer.write_scalars(step, summary)
            train_metrics = []

        if FLAGS.eval_only:
          p_eval_per_pos_step = jax.pmap(
              functools.partial(
                  train_util.eval_per_pos_step, config=eval_config),
              axis_name='batch')
          # Get per example loss
          loss_filename = FLAGS.model_dir + '/test_losses.csv'
          train_util.write_per_example_losses(
              p_eval_step=p_eval_per_pos_step,
              target=optimizer.target,
              eval_ds=eval_ds,
              num_eval_steps=FLAGS.num_eval_steps,
              loss_filename=loss_filename)
        else:
          with report_progress.timed('eval'):
            eval_results = train_util.evaluate(
                p_eval_step=p_eval_step,
                target=optimizer.target,
                eval_ds=eval_ds,
                num_eval_steps=FLAGS.num_eval_steps)
            curr_eval_loss = eval_results['loss']
            writer.write_scalars(
                step, {'eval_' + k: v for k, v in eval_results.items()})

        if FLAGS.aux_eval_dataset:
          for aux_i, aux_eval_ds in enumerate(aux_datasets):
            with report_progress.timed('aux_eval'):
              eval_results = train_util.evaluate(
                  p_eval_step=p_eval_step,
                  target=optimizer.target,
                  eval_ds=aux_eval_ds,
                  num_eval_steps=FLAGS.num_eval_steps)
              writer.write_scalars(
                  step, {
                      'aux' + str(aux_i) + '_eval_' + k: v
                      for k, v in eval_results.items()
                  })

        if FLAGS.compute_bleu:
          with report_progress.timed('translate_and_bleu'):
            decode_file = FLAGS.model_dir + '/decodes.csv'
            exemplars, bleu_score = train_util.translate_and_calculate_bleu(
                p_pred_step=p_pred_step,
                p_init_cache=p_init_cache,
                target=optimizer.target,
                predict_ds=predict_ds,
                decode_tokens=decode_tokens,
                max_predict_length=FLAGS.max_predict_length,
                num_eval_steps=FLAGS.num_eval_steps,
                decode_file=decode_file if FLAGS.eval_only else '')
            writer.write_scalars(step, {'bleu': bleu_score})
            writer.write_texts(step, {'samples': exemplars})

      # Save a checkpoint on one host after every checkpoint_freq steps.
      save_checkpoint = ((step - start_step) % FLAGS.checkpoint_freq == 0 or
                         is_last_step)
      if FLAGS.save_checkpoints and save_checkpoint and jax.process_index(
      ) == 0:
        if curr_eval_loss < best_eval_loss:  # only save better checkpoints
          best_eval_loss = curr_eval_loss
          with report_progress.timed('checkpoint'):
            checkpoints.save_checkpoint(
                FLAGS.model_dir, jax_utils.unreplicate(optimizer),
                step, keep=FLAGS.chkpts_to_keep, overwrite=True)

      if is_last_step:
        break


def get_new_distribution(p_get_diag_grads, p_get_bucket_score, eval_ds,
                         train_ds_mgr, optimizer, ft_ds):
  """Compute the new training distribution."""
  new_distribution = [1./FLAGS.num_data_buckets]*FLAGS.num_data_buckets
  val_grad, curr_grad, ft_grad = None, None, None
  for bucket_id in range(-2, FLAGS.num_data_buckets):
    print('Running bucket', bucket_id)
    if curr_grad is not None:
      del curr_grad
      curr_grad = None

    if bucket_id == -2:
      diag_iter = iter(eval_ds)
    elif bucket_id == -1:
      diag_iter = iter(ft_ds)
    else:
      diag_iter = train_ds_mgr.get_bucket(bucket_id)

    diag_batch = next(diag_iter)
    diag_batch = common_utils.shard(jax.tree.map(
        lambda x: x._numpy(), diag_batch))  # pylint: disable=protected-access

    if bucket_id == -2:
      val_grad = p_get_diag_grads(optimizer, diag_batch)
    elif bucket_id == -1:
      ft_grad = p_get_diag_grads(optimizer, diag_batch)
    else:  # get diag grad
      curr_grad = p_get_diag_grads(optimizer, diag_batch)

    # compute bucket score
    if bucket_id == -2:
      print('Val grad mean')
      val_grad = jax.pmap(
          functools.partial(tree_div, val_y=FLAGS.num_train_steps),
          axis_name='batch')(val_grad)
    if bucket_id == -1:
      print('Val grad mean')
      ft_grad = jax.pmap(
          functools.partial(tree_div, val_y=FLAGS.num_train_steps),
          axis_name='batch')(ft_grad)
    if bucket_id >= 0:
      print('cur grad mean')
      curr_grad = jax.pmap(
          functools.partial(tree_div, val_y=FLAGS.num_train_steps),
          axis_name='batch')(curr_grad)

      print('compute bucket scores')
      score = p_get_bucket_score(val_grad, ft_grad, curr_grad)
      device_score = jax.tree.map(lambda x: x[0], score)
      score_np = jax.device_get(device_score)
      new_distribution[bucket_id] = score_np
  logging.info(new_distribution)
  new_distribution = np.array(new_distribution).ravel()

  return new_distribution


def build_split(train_ds_mgr, split):
  bucket_size = 4500000. / FLAGS.num_data_buckets
  in_domain = int(np.round(FLAGS.data_selection_size / bucket_size))
  deadbins = 20
  new_distribution = np.zeros(FLAGS.num_data_buckets)
  new_distribution[:in_domain] = split / in_domain
  out_of_domain = FLAGS.num_data_buckets - in_domain - deadbins
  new_distribution[in_domain:out_of_domain] = (1-split) / out_of_domain
  train_ds = train_ds_mgr.sampled_dataset(new_distribution)
  return iter(train_ds)


def get_macro_distribution(p_get_diag_grads, p_get_bucket_score, eval_ds,
                           train_ds_mgr, optimizer, ft_ds):
  """Compute the new training distribution."""
  options = [0] * 10
  val_grad, curr_grad, ft_grad = None, None, None
  splits = [1.0, .95, .9, .85, .8, .75, .6, .5, .4, .25]
  for bucket_id in range(-2, 10):
    print('Running bucket', bucket_id)
    if curr_grad is not None:
      del curr_grad
      curr_grad = None

    if bucket_id == -2:
      diag_iter = iter(eval_ds)
    elif bucket_id == -1:
      diag_iter = iter(ft_ds)
    else:
      diag_iter = build_split(train_ds_mgr, splits[bucket_id])

    diag_batch = next(diag_iter)
    diag_batch = common_utils.shard(jax.tree.map(
        lambda x: x._numpy(), diag_batch))  # pylint: disable=protected-access

    if bucket_id == -2:
      val_grad = p_get_diag_grads(optimizer, diag_batch)
    elif bucket_id == -1:
      ft_grad = p_get_diag_grads(optimizer, diag_batch)
    else:  # get diag grad
      curr_grad = p_get_diag_grads(optimizer, diag_batch)

    # compute bucket score
    if bucket_id == -2:
      print('Val grad mean')
      val_grad = jax.pmap(
          functools.partial(tree_div, val_y=FLAGS.num_train_steps),
          axis_name='batch')(val_grad)
    if bucket_id == -1:
      print('Val grad mean')
      ft_grad = jax.pmap(
          functools.partial(tree_div, val_y=FLAGS.num_train_steps),
          axis_name='batch')(ft_grad)
    if bucket_id >= 0:
      print('cur grad mean')
      curr_grad = jax.pmap(
          functools.partial(tree_div, val_y=FLAGS.num_train_steps),
          axis_name='batch')(curr_grad)

      print('compute bucket scores')
      score = p_get_bucket_score(val_grad, ft_grad, curr_grad)
      device_score = jax.tree.map(lambda x: x[0], score)
      score_np = jax.device_get(device_score)
      options[bucket_id] = score_np
  logging.info(options)
  logging.info(splits[np.argmax(options)])
  return build_split(train_ds_mgr, splits[np.argmax(options)])


def get_diag_score(g_val, g_ft, g_curr, strategy=train_util.ORTH):
  """Compute diagnostic - cosine similarity between gradients."""
  if strategy == train_util.ORTH:
    g_curr_proj_on_train = tree_div(
        tree_mult(g_val, tree_dot(g_curr, g_val)),
        tree_dot(g_val, g_val))
    g_curr_orth = tree_diff(g_curr, g_curr_proj_on_train)
    orth_dot_valid = tree_dot(g_curr_orth, g_ft)
    orth_score = orth_dot_valid / tree_norm(g_curr_orth) * tree_norm(g_ft)
    return orth_score

  else:
    obw_dot_valid = tree_dot(g_curr, g_val)
    greedy_score = obw_dot_valid / tree_norm(g_curr) * tree_norm(g_val)
    return greedy_score


if __name__ == '__main__':
  app.run(main)

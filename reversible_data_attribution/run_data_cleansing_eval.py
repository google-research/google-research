# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

r"""Binary CLI for running Section 7.2 Data Cleansing evaluation experiments.

Usage:
  python -m reversible_data_attribution.run_data_cleansing_eval \
    --dataset=mnist --num_epochs=20 --k_list=1,10,100 --output_dir=/tmp/cleansing_results
"""

from collections.abc import Sequence
import csv
import json
import os
import random
from typing import Any

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torch import nn
from torch.utils import data as torch_data

from reversible_data_attribution import base_model as base_model_lib
from reversible_data_attribution import data_cleansing
from reversible_data_attribution import models
from reversible_data_attribution import pipeline
from reversible_data_attribution import reversible
from reversible_data_attribution import utils

FLAGS = flags.FLAGS

if 'dataset' not in flags.FLAGS:
  flags.DEFINE_enum(
      'dataset',
      'mnist',
      ['mnist', 'cifar', 'cifar10', 'cifar100', 'fashion_mnist', 'imdb'],
      'Dataset to use for evaluation.',
  )
if 'data_path' not in flags.FLAGS:
  flags.DEFINE_string(
      'data_path',
      'third_party/google_research/google_research/reversible_data_attribution/mnist/mnist_local.npz',
      'Path to dataset npz file.',
  )
if 'num_train' not in flags.FLAGS:
  flags.DEFINE_integer('num_train', 1000, 'Number of training samples.')
if 'num_val' not in flags.FLAGS:
  flags.DEFINE_integer('num_val', 200, 'Number of validation samples.')
if 'num_test' not in flags.FLAGS:
  flags.DEFINE_integer('num_test', 200, 'Number of test samples.')
if 'num_epochs' not in flags.FLAGS:
  flags.DEFINE_integer('num_epochs', 10, 'Number of training epochs.')
if 'batch_size' not in flags.FLAGS:
  flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')
if 'learning_rate' not in flags.FLAGS:
  flags.DEFINE_float('learning_rate', 0.05, 'Learning rate for optimizers.')
if 'momentum' not in flags.FLAGS:
  flags.DEFINE_float('momentum', 0.0, 'Momentum for optimizers.')
if 'k_list' not in flags.FLAGS:
  flags.DEFINE_list(
      'k_list',
      ['100', '1000'],
      'List of k values (number of samples to remove during cleansing).',
  )
if 'methods' not in flags.FLAGS:
  flags.DEFINE_list(
      'methods',
      [
          'adam_recursive',
          'adam_recursive_nodv',
          'adam_exact',
          'tracin_adam',
          'tracin_sgd',
          'sgd_all',
          'sgd_last',
          'icml',
          'random',
          'ae',
          'iso',
      ],
      'List of methods to compare.',
  )
if 'optimizer_type' not in flags.FLAGS:
  flags.DEFINE_enum(
      'optimizer_type',
      'adam',
      ['adam', 'sgd'],
      'Optimizer type for model training and retraining.',
  )
if 'beta_1' not in flags.FLAGS:
  flags.DEFINE_float('beta_1', 0.9, 'Beta 1 parameter for Adam optimizer.')
if 'beta_2' not in flags.FLAGS:
  flags.DEFINE_float('beta_2', 0.999, 'Beta 2 parameter for Adam optimizer.')
if 'eps' not in flags.FLAGS:
  flags.DEFINE_float('eps', 1e-8, 'Epsilon parameter for Adam optimizer.')

if 'device' not in flags.FLAGS:
  flags.DEFINE_string('device', 'cpu', 'Device for computation (cpu or cuda).')
if 'output_dir' not in flags.FLAGS:
  flags.DEFINE_string(
      'output_dir',
      '/tmp/data_cleansing_results',
      'Directory to save output evaluation CSV results.',
  )
if 'adam_mode' not in flags.FLAGS:
  flags.DEFINE_enum(
      'adam_mode',
      'recursive',
      ['recursive', 'recursive_nodv', 'forward', 'tracin'],
      'Adam influence mode: "recursive", "recursive_nodv", "forward", or'
      ' "tracin".',
  )
if 'adam_remove_dv' not in flags.FLAGS:
  flags.DEFINE_boolean(
      'adam_remove_dv',
      False,
      'Whether to remove the dv term for Adam forward update.',
  )
if 'adam_dv_option' not in flags.FLAGS:
  flags.DEFINE_enum(
      'adam_dv_option',
      'first_order',
      ['first_order', 'exact'],
      'Option for variance change in Adam forward update.',
  )
if 'adam_theta_option' not in flags.FLAGS:
  flags.DEFINE_enum(
      'adam_theta_option',
      'first_order',
      ['first_order', 'second_order', 'exact'],
      'Option for theta change in Adam forward update.',
  )
if 'noise_type' not in flags.FLAGS:
  flags.DEFINE_enum(
      'noise_type',
      'gaussian',
      [
          'none',
          'label_flip',
          'gaussian',
          'both',
          'lowpass',
          'highpass',
          'low_pass',
          'high_pass',
      ],
      'Type of noise to corrupt the training dataset with.',
  )
if 'noise_rate' not in flags.FLAGS:
  flags.DEFINE_float(
      'noise_rate',
      0.1,
      'Fraction of training dataset to corrupt with noise.',
  )
if 'gaussian_std' not in flags.FLAGS:
  flags.DEFINE_float(
      'gaussian_std',
      0.1,
      'Standard deviation of Gaussian noise added to inputs.',
  )

# Flags for lowpass / highpass filtering.
if 'cutoff_freq' not in flags.FLAGS:
  flags.DEFINE_float(
      'cutoff_freq',
      0.5,
      'Cutoff frequency for lowpass / highpass filtering.',
  )
if 'filter_order' not in flags.FLAGS:
  flags.DEFINE_integer(
      'filter_order',
      2,
      'Filter order for lowpass / highpass filtering.',
  )
if 'sampling_rate' not in flags.FLAGS:
  flags.DEFINE_float(
      'sampling_rate',
      60.0,
      'Sampling rate for lowpass / highpass filtering.',
  )
if 'noise_seed' not in flags.FLAGS:
  flags.DEFINE_integer(
      'noise_seed',
      42,
      'Random seed for dataset noise generation.',
  )
if 'auto_cleansing' not in flags.FLAGS:
  flags.DEFINE_boolean(
      'auto_cleansing',
      True,
      'Whether to automatically select and remove counterfactuals that decrease'
      ' loss.',
  )
if 'eval_counterfactuals' not in flags.FLAGS:
  flags.DEFINE_boolean(
      'eval_counterfactuals',
      False,
      'Whether to run full ground truth leave-one-out counterfactual evaluation'
      ' and compare estimation vs cleansing abilities.',
  )
if 'random_masking' not in flags.FLAGS:
  flags.DEFINE_float(
      'random_masking',
      None,
      'Proportion of gradient coordinates to keep track via Bernoulli(p) random'
      ' masking for forward Adam update.',
  )
if 'retrain_seeds' not in flags.FLAGS:
  flags.DEFINE_list(
      'retrain_seeds',
      None,
      'List of random seeds for randomized counterfactual retraining (in'
      ' addition to preserving batch structure).',
  )


if 'model_type' not in flags.FLAGS:
  flags.DEFINE_enum(
      'model_type',
      'cnn',
      ['small_dnn', 'dnn', 'huge_dnn', 'cnn', 'vit', 'linear', 'resnet', 'resnet18'],
      'Model architecture choice for evaluation.',
  )
if 'max_workers' not in flags.FLAGS:
  flags.DEFINE_integer(
      'max_workers',
      1,
      'Number of parallel worker threads for forward update and LOO'
      ' counterfactual evaluation.',
  )
if 'shard_id' not in flags.FLAGS:
  flags.DEFINE_integer(
      'shard_id',
      None,
      'Shard ID (0-indexed) for distributed Map phase score evaluation.',
  )
if 'num_shards' not in flags.FLAGS:
  flags.DEFINE_integer(
      'num_shards',
      1,
      'Total number of shards for distributed Map phase score evaluation.',
  )
if 'save_scores' not in flags.FLAGS:
  flags.DEFINE_boolean(
      'save_scores',
      True,
      'Whether to save computed scores to output_dir or save_scores_dir.',
  )
if 'save_scores_dir' not in flags.FLAGS:
  flags.DEFINE_string(
      'save_scores_dir',
      None,
      'Directory to save score artifacts.',
  )
if 'precomputed_scores_dir' not in flags.FLAGS:
  flags.DEFINE_string(
      'precomputed_scores_dir',
      None,
      'Directory or file path containing precomputed/sharded scores to load.',
  )
if 'mode' not in flags.FLAGS:
  flags.DEFINE_enum(
      'mode',
      'all',
      ['all', 'compute_scores', 'cleansing_from_scores'],
      'Execution mode: "all" (full pipeline), "compute_scores" (only evaluate'
      ' scores for shard), or "cleansing_from_scores" (load precomputed scores'
      ' and run cleansing retraining).',
  )
if 'use_reversible_transform' not in flags.FLAGS:
  flags.DEFINE_bool(
      'use_reversible_transform',
      False,
      'Whether to use the reversible transform for backward Adam update.',
  )
if 'use_reversible' not in flags.FLAGS:
  flags.DEFINE_bool(
      'use_reversible',
      False,
      'Whether to use reversible Adam with quantization for backward update.',
  )
if 'quantization_scale' not in flags.FLAGS:
  flags.DEFINE_integer(
      'quantization_scale',
      1000000000000,
      'Scale factor for fixed-point quantization in reversible Adam.',
  )
if 'ignore_first' not in flags.FLAGS:
  flags.DEFINE_boolean(
      'ignore_first',
      False,
      'Whether to ignore gradients contributed by a datapoint on its first pass'
      ' (sets start_step to number of batches).',
  )
if 'start_step' not in flags.FLAGS:
  flags.DEFINE_integer(
      'start_step',
      0,
      'Start step to ignore any gradient effect before start_step in recursive'
      ' updates.',
  )
if 'checkpoint_freq' not in flags.FLAGS:
  flags.DEFINE_integer(
      'checkpoint_freq',
      None,
      'Step interval for periodic checkpointing (e.g. 100). If None, stores'
      ' only 0th and last checkpoints when reversible is enabled.',
  )
if 'checkpoint_frequency' not in flags.FLAGS:
  flags.DEFINE_integer(
      'checkpoint_frequency',
      None,
      'Alias for checkpoint_freq.',
  )
if 'norm_type' not in flags.FLAGS:
  flags.DEFINE_enum(
      'norm_type',
      'group_norm',
      ['batch_norm', 'batch_norm_no_stats', 'group_norm', 'layer_norm', 'none'],
      'Normalization layer type for ResNet architecture.',
  )


def load_dataset(
    dataset_name,
    data_path,
    num_train,
    num_val,
    num_test,
    noise_type = 'none',
    noise_rate = 0.0,
    gaussian_std = 0.1,
    cutoff_freq = 0.5,
    filter_order = 2,
    sampling_rate = 60.0,
    seed = 42,
):
  """Loads and splits dataset into train, val, test datasets and applies noise to train set."""
  dataset_name = dataset_name.lower()
  if dataset_name == 'mnist':
    data_file = (
        data_path
        or 'third_party/google_research/google_research/reversible_data_attribution/mnist/mnist_local.npz'
    )
    x_tr, y_tr, x_te, y_te = pipeline.load_mnist_data(
        data_file,
        num_train=num_train + num_val,
        num_test=num_test,
    )
    num_classes = 10
  elif dataset_name in ('cifar', 'cifar10', 'cifar100'):
    data_file = (
        data_path
        or 'third_party/google_research/google_research/reversible_data_attribution/cifar/cifar_local.npz'
    )
    num_classes = 100 if dataset_name == 'cifar100' else 10
    x_tr, y_tr, x_te, y_te = pipeline.load_cifar_data(
        data_file,
        num_train=num_train + num_val,
        num_test=num_test,
        flatten=True,
        num_classes=num_classes,
    )
  elif dataset_name == 'fashion_mnist':
    data_file = (
        data_path
        or 'third_party/google_research/google_research/reversible_data_attribution/fashion_mnist/fashion_mnist.npz'
    )
    num_classes = 10
    x_tr, y_tr, x_te, y_te = pipeline.load_fashion_mnist_data(
        data_file,
        num_train=num_train + num_val,
        num_test=num_test,
    )
  elif dataset_name == 'imdb':
    data_file = (
        data_path
        or 'third_party/google_research/google_research/reversible_data_attribution/imdb/imdb_local.npz'
    )
    num_classes = 2
    x_tr, y_tr, x_te, y_te = pipeline.load_imdb_data(
        data_file,
        num_train=num_train + num_val,
        num_test=num_test,
    )
  else:
    raise NotImplementedError(f'Dataset {dataset_name} not implemented.')

  x_train_raw = x_tr[:num_train]
  x_val_raw = x_tr[num_train : num_train + num_val]
  y_train_raw = y_tr[:num_train]
  y_val_raw = y_tr[num_train : num_train + num_val]

  corrupted_indices = []
  if noise_type.lower() != 'none' and noise_rate > 0.0:
    x_train_raw, y_train_raw, corrupted_indices = utils.apply_dataset_noise(
        x_train=x_train_raw,
        y_train=y_train_raw,
        noise_type=noise_type,
        noise_rate=noise_rate,
        num_classes=num_classes,
        gaussian_std=gaussian_std,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        sampling_rate=sampling_rate,
        seed=seed,
    )

  input_dim = int(x_train_raw.shape[1])
  corrupted_seq: Sequence[int] = (
      list(corrupted_indices)
      if isinstance(corrupted_indices, (list, tuple, np.ndarray))
      else []
  )

  train_ds = torch_data.TensorDataset(
      torch.from_numpy(x_train_raw), torch.from_numpy(y_train_raw)
  )
  val_ds = torch_data.TensorDataset(
      torch.from_numpy(x_val_raw), torch.from_numpy(y_val_raw)
  )
  test_ds = torch_data.TensorDataset(
      torch.from_numpy(x_te), torch.from_numpy(y_te)
  )

  return train_ds, val_ds, test_ds, input_dim, corrupted_seq


def save_config(output_dir):
  """Saves experiment configuration and hyperparameters to config.json."""
  if not output_dir:
    return
  config_data = {}
  try:
    for flag_name, flag_obj in FLAGS.flags_by_name().items():
      is_custom_module = (
          flag_obj.module_name
          and 'data_attribution' in str(flag_obj.module_name)
      )
      is_explicitly_set = getattr(flag_obj, 'present', 0) > 0
      if is_custom_module or is_explicitly_set:
        val = flag_obj.value
        try:
          json.dumps(val)
          config_data[flag_name] = val
        except (TypeError, OverflowError):
          config_data[flag_name] = str(val)
  except Exception:  # pylint: disable=broad-except
    for k in [
        'dataset',
        'data_path',
        'num_train',
        'num_val',
        'num_test',
        'num_epochs',
        'batch_size',
        'learning_rate',
        'momentum',
        'k_list',
        'methods',
        'optimizer_type',
        'beta_1',
        'beta_2',
        'eps',
        'device',
        'output_dir',
        'adam_mode',
        'adam_remove_dv',
        'adam_dv_option',
        'adam_theta_option',
        'noise_type',
        'noise_rate',
        'gaussian_std',
        'cutoff_freq',
        'filter_order',
        'sampling_rate',
        'noise_seed',
        'model_type',
        'eval_counterfactuals',
        'retrain_seeds',
        'mode',
        'num_shards',
        'max_workers',
        'ignore_first',
        'start_step',
        'checkpoint_freq',
        'checkpoint_frequency',
        'use_reversible_transform',
        'use_reversible',
        'quantization_scale',
    ]:
      if hasattr(FLAGS, k):
        config_data[k] = getattr(FLAGS, k)

  config_path = os.path.join(output_dir, 'config.json')
  try:
    with pipeline._open_file(config_path, 'w') as f:
      json.dump(config_data, f, indent=2)
    logging.info('Successfully saved configuration to %s', config_path)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('Failed to write config.json to %s: %s', config_path, e)


def log_quantization_and_buffer_stats(
    base_model_obj,
    quantized_checkpoints,
    momentum_buffer,
    variance_buffer,
    output_dir = None,
):
  """Computes and logs numerical statistics for quantized checkpoints and buffers.

  Args:
    base_model_obj: The unquantized base model object.
    quantized_checkpoints: Dictionary of quantized checkpoints from
      retrain_with_quantize.
    momentum_buffer: The GPUReversibleTransform momentum buffer.
    variance_buffer: The GPUReversibleTransform variance buffer.
    output_dir: Optional directory to save quantization_stats.json.

  Returns:
    A dictionary summarizing the numerical statistics.
  """
  stats = {}
  unquantized_ckpts = (
      base_model_obj.checkpoints()
      if hasattr(base_model_obj, 'checkpoints')
      else base_model_obj.get_checkpoint_models()
  )

  if quantized_checkpoints:
    logging.info('=== QUANTIZATION NUMERICAL STATS ===')
    ckpt_diffs = {}
    overall_max_diff = 0.0

    for step_key, q_val in sorted(
        quantized_checkpoints.items(),
        key=lambda item: (item[0] if item[0] >= 0 else float('inf')),
    ):
      if step_key not in unquantized_ckpts:
        continue
      unq_val = unquantized_ckpts[step_key]
      if isinstance(unq_val, base_model_lib.CheckpointState):
        unq_model = unq_val.model
      elif isinstance(unq_val, nn.Module):
        unq_model = unq_val
      elif hasattr(unq_val, 'model'):
        unq_model = getattr(unq_val, 'model')
      else:
        continue

      if isinstance(q_val, base_model_lib.CheckpointState):
        q_model = q_val.model
      elif isinstance(q_val, nn.Module):
        q_model = q_val
      elif hasattr(q_val, 'model'):
        q_model = getattr(q_val, 'model')
      else:
        continue

      step_max_diff = 0.0
      for p_unq, p_q in zip(unq_model.parameters(), q_model.parameters()):
        diff = (p_unq.cpu().detach() - p_q.cpu().detach()).abs().max().item()
        step_max_diff = max(step_max_diff, diff)

      ckpt_diffs[str(step_key)] = step_max_diff
      overall_max_diff = max(overall_max_diff, step_max_diff)
      step_name = f'Step {step_key}' if step_key >= 0 else 'Final Step (-1)'
      logging.info(
          '  [Checkpoint Diff] %s: max absolute difference = %.8e',
          step_name,
          step_max_diff,
      )

    stats['checkpoint_max_diffs'] = ckpt_diffs
    stats['overall_checkpoint_max_diff'] = overall_max_diff
    logging.info(
        '  [Checkpoint Diff] Overall max absolute difference across checkpoints ='
        ' %.8e',
        overall_max_diff,
    )

  def _compute_buffer_stats(
      name, buf
  ):
    if buf is None:
      return None

    # Handle GPUReversibleTransform
    if hasattr(buf, 'active_buffer'):
      active_buf = buf.active_buffer.detach().cpu()
      tape = buf.tape.detach().cpu()
      limb_idx = getattr(buf, 'limb_idx', 0)
      max_limbs = getattr(buf, 'max_limbs', 0)
      num_params = getattr(buf, 'num_params', active_buf.numel())

      buf_stats = {
          'num_elements': num_params,
          'tape_length_limbs': limb_idx,
          'max_limbs_allocated': max_limbs,
          'active_buffer_max': int(active_buf.abs().max().item()),
          'active_buffer_min': int(active_buf.abs().min().item()),
          'active_buffer_mean': float(
              active_buf.abs().to(torch.float64).mean().item()
          ),
          'tape_memory_kb': float(
              (
                  tape.element_size() * tape.nelement()
                  + active_buf.element_size() * active_buf.nelement()
              )
              / 1024.0
          ),
      }
      logging.info('=== %s GPU BUFFER STATS ===', name.upper())
      logging.info('  Elements stored: %d', buf_stats['num_elements'])
      logging.info(
          '  Tape spilled limbs (length): %d / %d', limb_idx, max_limbs
      )
      logging.info(
          '  Active buffer abs: max=%d, min=%d, mean=%.2f',
          buf_stats['active_buffer_max'],
          buf_stats['active_buffer_min'],
          buf_stats['active_buffer_mean'],
      )
      logging.info(
          '  Total GPU buffer memory: %.2f KB', buf_stats['tape_memory_kb']
      )
      return buf_stats

    return None

  if momentum_buffer is not None:
    stats['momentum_buffer_stats'] = _compute_buffer_stats(
        'momentum', momentum_buffer
    )
  if variance_buffer is not None:
    stats['variance_buffer_stats'] = _compute_buffer_stats(
        'variance', variance_buffer
    )

  if output_dir:
    stats_file = os.path.join(output_dir, 'quantization_stats.json')
    try:
      with pipeline._open_file(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
      logging.info('Saved quantization stats to %s', stats_file)
    except Exception as e:  # pylint: disable=broad-except
      logging.warning('Could not save quantization_stats.json: %s', e)

  return stats


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  device = torch.device(FLAGS.device)
  pipeline._make_dirs(FLAGS.output_dir)
  save_config(FLAGS.output_dir)

  # Set deterministic random seeds so initial model weights (timestamp 0)
  # and training trajectories are identical across all parallel shards and methods.
  torch.manual_seed(FLAGS.noise_seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(FLAGS.noise_seed)
  np.random.seed(FLAGS.noise_seed)
  random.seed(FLAGS.noise_seed)

  logging.info('Loading dataset %s...', FLAGS.dataset)
  train_ds, val_ds, test_ds, input_dim, corrupted_indices = load_dataset(
      FLAGS.dataset,
      FLAGS.data_path,
      FLAGS.num_train,
      FLAGS.num_val,
      FLAGS.num_test,
      noise_type=FLAGS.noise_type,
      noise_rate=FLAGS.noise_rate,
      gaussian_std=FLAGS.gaussian_std,
      cutoff_freq=FLAGS.cutoff_freq,
      filter_order=FLAGS.filter_order,
      sampling_rate=FLAGS.sampling_rate,
      seed=FLAGS.noise_seed,
  )

  train_loader = torch_data.DataLoader(
      train_ds, batch_size=FLAGS.batch_size, shuffle=False
  )
  val_loader = torch_data.DataLoader(
      val_ds, batch_size=FLAGS.batch_size, shuffle=False
  )
  test_loader = torch_data.DataLoader(
      test_ds, batch_size=FLAGS.batch_size, shuffle=False
  )

  model_fn = lambda: models.get_model(
      FLAGS.model_type,
      input_dim=input_dim,
      output_dim=10,
      norm_type=FLAGS.norm_type,
  )
  criterion = nn.CrossEntropyLoss()

  use_reversible = bool(
      FLAGS.use_reversible or FLAGS.use_reversible_transform
  )
  checkpoint_freq = (
      FLAGS.checkpoint_frequency
      if FLAGS.checkpoint_frequency is not None
      else FLAGS.checkpoint_freq
  )
  quantization_scale = FLAGS.quantization_scale

  # Pass 1: Full Model Training using AttributionBaseModel
  logging.info(
      'Training full model (%s) for %d epochs (checkpoint_freq=%s)...',
      FLAGS.optimizer_type,
      FLAGS.num_epochs,
      checkpoint_freq,
  )
  torch.manual_seed(FLAGS.noise_seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(FLAGS.noise_seed)
  raw_base_model = model_fn().to(device)

  base_model_obj = base_model_lib.AttributionBaseModel(
      model=raw_base_model,
      loss_fn=criterion,
      lr=FLAGS.learning_rate,
      beta_1=FLAGS.beta_1,
      beta_2=FLAGS.beta_2,
      eps=FLAGS.eps,
      seed=FLAGS.noise_seed,
  )
  save_all_ckpts = (
      checkpoint_freq is None and not use_reversible and (FLAGS.mode != 'cleansing_from_scores')
  )
  base_model_obj.train_model(
      train_loader=train_loader,
      num_epochs=FLAGS.num_epochs,
      checkpoint_freq=checkpoint_freq,
      device=device,
      save_all_ckpts=save_all_ckpts,
  )
  checkpoints = base_model_obj.get_checkpoint_models()
  momentum_lst = base_model_obj.get_momentum_dict()
  variance_lst = base_model_obj.get_variance_dict()
  num_steps = base_model_obj.total_steps()
  utils.log_cuda_memory(
      'After full model training loop (run_data_cleansing_eval)', device=device
  )

  # Pass 2: Quantized Model Training if use_reversible is active
  quantized_model = None
  quantized_checkpoints = None
  momentum_buffer = None
  variance_buffer = None
  if use_reversible:
    logging.info(
        'Retraining quantized model for reversible Adam'
        ' (quantization_scale=%d, checkpoint_freq=%s)...',
        quantization_scale,
        checkpoint_freq,
    )
    (
        quantized_model,
        quantized_checkpoints,
        momentum_buffer,
        variance_buffer,
    ) = base_model_obj.retrain_with_quantize(
        train_loader=train_loader,
        num_epochs=FLAGS.num_epochs,
        checkpoint_freq=checkpoint_freq,
        device=device,
        quantization_scale=quantization_scale,
    )
    utils.log_cuda_memory(
        'After quantized model retraining (run_data_cleansing_eval)',
        device=device,
    )
    log_quantization_and_buffer_stats(
        base_model_obj=base_model_obj,
        quantized_checkpoints=quantized_checkpoints,
        momentum_buffer=momentum_buffer,
        variance_buffer=variance_buffer,
        output_dir=FLAGS.output_dir,
    )

  random_mask = None
  if FLAGS.random_masking is not None and FLAGS.random_masking > 0:
    random_mask = utils.generate_random_mask(
        checkpoints[-1],
        prob=FLAGS.random_masking,
        seed=FLAGS.noise_seed,
        device=device,
    )

  raw_seeds = FLAGS.retrain_seeds
  retrain_seeds: list[int] | None = None
  if raw_seeds:
    if isinstance(raw_seeds, (list, tuple)):
      retrain_seeds = [int(s) for s in raw_seeds if str(s).strip()]
    elif isinstance(raw_seeds, str):
      retrain_seeds = [
          int(s.strip()) for s in raw_seeds.split(',') if s.strip()
      ]
    elif isinstance(raw_seeds, int):
      retrain_seeds = [raw_seeds]

  if retrain_seeds:
    logging.info(
        'Randomized counterfactual retraining enabled for seeds: %s',
        retrain_seeds,
    )

  raw_methods = FLAGS.methods
  if isinstance(raw_methods, str):
    raw_methods = [m.strip() for m in raw_methods.split(',') if m.strip()]
  elif isinstance(raw_methods, (list, tuple)):
    flat_methods = []
    for item in raw_methods:
      if isinstance(item, str) and ',' in item:
        flat_methods.extend([m.strip() for m in item.split(',') if m.strip()])
      else:
        flat_methods.append(item)
    raw_methods = flat_methods

  adam_method_name = data_cleansing.get_adam_method_name(FLAGS.random_masking)
  methods = []
  for m in raw_methods:
    if m in (
        'adam_forward',
        'adam_exact',
        'adam_forward_masked',
    ) or m.startswith('adam_masked'):
      methods.append(adam_method_name)
    else:
      methods.append(m)

  # Determine shard indices if sharded evaluation is requested
  num_train = len(train_ds)
  shard_indices = None
  if FLAGS.shard_id is not None:
    chunk_size = (num_train + FLAGS.num_shards - 1) // FLAGS.num_shards
    start_idx = FLAGS.shard_id * chunk_size
    end_idx = min(start_idx + chunk_size, num_train)
    shard_indices = list(range(start_idx, end_idx))
    logging.info(
        'Sharded evaluation active: shard %d/%d covering indices [%d, %d) (%d'
        ' samples)',
        FLAGS.shard_id,
        FLAGS.num_shards,
        start_idx,
        end_idx,
        len(shard_indices),
    )

  # Handle precomputed scores or standalone score computation mode
  scores_dict = {}
  if FLAGS.precomputed_scores_dir:
    logging.info(
        'Loading precomputed / sharded scores from %s',
        FLAGS.precomputed_scores_dir,
    )
    scores_dict = data_cleansing.merge_sharded_scores(
        FLAGS.precomputed_scores_dir, num_train=num_train
    )
    logging.info('Loaded precomputed methods: %s', list(scores_dict.keys()))

  num_batches = len(utils.prepare_batches(train_loader))
  start_step = num_batches if FLAGS.ignore_first else FLAGS.start_step
  if start_step > 0:
    logging.info(
        'Ignoring gradient effects before start_step=%d (ignore_first=%s) for'
        ' backwards recursion',
        start_step,
        FLAGS.ignore_first,
    )

  if FLAGS.mode == 'compute_scores':
    logging.info(
        'Running standalone score computation (Map phase) for methods: %s'
        ' (max_workers=%d)',
        methods,
        FLAGS.max_workers,
    )
    computed_scores = data_cleansing.compute_scores_for_methods(
        methods=methods,
        base_model_obj=base_model_obj,
        val_loader=val_loader,
        optimizer_type=FLAGS.optimizer_type,
        random_mask=random_mask,
        candidate_indices=shard_indices,
        max_workers=FLAGS.max_workers,
        num_steps=num_steps,
        device=device,
        use_reversible_transform=use_reversible,
        start_step=start_step,
        quantized_checkpoints=quantized_checkpoints,
        momentum_buffer=momentum_buffer,
        variance_buffer=variance_buffer,
        quantization_scale=quantization_scale,
    )
    scores_dict.update(computed_scores)

    if FLAGS.eval_counterfactuals:
      logging.info(
          'Computing ground-truth leave-one-out counterfactuals for shard'
          ' (max_workers=%d)...',
          FLAGS.max_workers,
      )
      loo_res = data_cleansing.compute_leave_one_out_counterfactuals(
          model_fn=model_fn,
          train_dataset=train_ds,
          train_loader=train_loader,
          val_loader=val_loader,
          test_loader=test_loader,
          sample_indices=shard_indices,
          num_epochs=FLAGS.num_epochs,
          batch_size=FLAGS.batch_size,
          lr=FLAGS.learning_rate,
          optimizer_type=FLAGS.optimizer_type,
          momentum=FLAGS.momentum,
          beta_1=FLAGS.beta_1,
          beta_2=FLAGS.beta_2,
          eps=FLAGS.eps,
          criterion=criterion,
          seed=FLAGS.noise_seed,
          retrain_seeds=retrain_seeds,
          max_workers=FLAGS.max_workers,
          device=device,
      )
      scores_dict['actual_cf'] = loo_res['val_loss_changes']

    save_dir = FLAGS.save_scores_dir or FLAGS.output_dir
    pipeline._make_dirs(save_dir)
    shard_str = f'_{FLAGS.shard_id:04d}' if FLAGS.shard_id is not None else ''
    shard_out_file = os.path.join(save_dir, f'scores_shard{shard_str}.npz')
    logging.info('Saving computed shard scores to %s', shard_out_file)
    data_cleansing.save_scores(scores_dict, shard_out_file)
    if corrupted_indices:
      corrupted_file = os.path.join(save_dir, 'corrupted_indices.json')
      try:
        with pipeline._open_file(corrupted_file, 'w') as f:
          json.dump([int(i) for i in corrupted_indices], f)
        logging.info(
            'Saved corrupted indices (%d samples) to %s',
            len(corrupted_indices),
            corrupted_file,
        )
      except Exception as e:
        logging.warning('Could not save corrupted_indices.json: %s', e)

    config_file = os.path.join(save_dir, 'config.json')
    try:
      config_data = {
          'dataset': FLAGS.dataset,
          'model_type': FLAGS.model_type,
          'optimizer_type': FLAGS.optimizer_type,
          'learning_rate': FLAGS.learning_rate,
          'batch_size': FLAGS.batch_size,
          'num_epochs': FLAGS.num_epochs,
          'noise_type': FLAGS.noise_type,
          'noise_rate': FLAGS.noise_rate,
          'gaussian_std': FLAGS.gaussian_std,
          'cutoff_freq': FLAGS.cutoff_freq,
          'filter_order': FLAGS.filter_order,
          'sampling_rate': FLAGS.sampling_rate,
          'noise_seed': FLAGS.noise_seed,
          'num_train': FLAGS.num_train,
          'num_val': FLAGS.num_val,
          'num_test': FLAGS.num_test,
          'num_shards': FLAGS.num_shards,
          'ignore_first': FLAGS.ignore_first,
          'start_step': start_step,
      }
      with pipeline._open_file(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
      logging.info('Saved hyperparameter configuration to %s', config_file)
    except Exception as e:
      logging.warning('Could not save config.json: %s', e)
    logging.info('Score computation completed successfully for shard.')
    return

  if not scores_dict and FLAGS.mode != 'cleansing_from_scores':
    # Precompute scores once for all methods
    logging.info(
        'Precomputing scores for methods: %s (max_workers=%d)',
        methods,
        FLAGS.max_workers,
    )
    scores_dict = data_cleansing.compute_scores_for_methods(
        methods=methods,
        base_model_obj=base_model_obj,
        val_loader=val_loader,
        optimizer_type=FLAGS.optimizer_type,
        random_mask=random_mask,
        candidate_indices=shard_indices,
        max_workers=FLAGS.max_workers,
        num_steps=num_steps,
        device=device,
        use_reversible_transform=use_reversible,
        start_step=start_step,
        quantized_checkpoints=quantized_checkpoints,
        momentum_buffer=momentum_buffer,
        variance_buffer=variance_buffer,
        quantization_scale=quantization_scale,
    )
    if FLAGS.save_scores:
      save_dir = FLAGS.save_scores_dir or FLAGS.output_dir
      pipeline._make_dirs(save_dir)
      score_save_path = os.path.join(save_dir, 'scores.npz')
      logging.info('Saving computed scores to %s', score_save_path)
      data_cleansing.save_scores(scores_dict, score_save_path)
      if corrupted_indices:
        corrupted_file = os.path.join(save_dir, 'corrupted_indices.json')
        try:
          with pipeline._open_file(corrupted_file, 'w') as f:
            json.dump([int(i) for i in corrupted_indices], f)
          logging.info(
              'Saved corrupted indices (%d samples) to %s',
              len(corrupted_indices),
              corrupted_file,
          )
        except Exception as e:
          logging.warning('Could not save corrupted_indices.json: %s', e)

  k_list = [int(k) for k in FLAGS.k_list]
  if corrupted_indices:
    # To make comparisons more meaningful, add the exact number of
    # corrupted samples to the k_list, as well as as some smaller k values.
    num_corrupted = len(corrupted_indices)
    add_list = [
        int(0.1 * num_corrupted),
        int(0.25 * num_corrupted),
        int(0.5 * num_corrupted),
        num_corrupted,
        int(1.5 * num_corrupted),
    ]
    for k in add_list:
      if k not in k_list:
        logging.info(
            'Adding k = %d to k_list',
            k,
        )
        k_list.append(k)
      k_list = sorted(k_list)

  logging.info(
      'Running unified data cleansing evaluation across methods: %s'
      ' (k_list: %s)',
      methods,
      k_list,
  )
  cleansing_details = data_cleansing.evaluate_data_cleansing(
      model_fn=model_fn,
      checkpoints=checkpoints,
      train_dataset=train_ds,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      scores_dict=scores_dict,
      corrupted_indices=corrupted_indices,
      momentum_lst=momentum_lst,
      variance_lst=variance_lst,
      criterion=criterion,
      lr=FLAGS.learning_rate,
      momentum=FLAGS.momentum,
      beta_1=FLAGS.beta_1,
      beta_2=FLAGS.beta_2,
      eps=FLAGS.eps,
      optimizer_type=FLAGS.optimizer_type,
      num_epochs=FLAGS.num_epochs,
      batch_size=FLAGS.batch_size,
      k_list=k_list,
      methods=methods,
      retrain_seeds=retrain_seeds,
      random_mask=random_mask,
      max_workers=FLAGS.max_workers,
      num_steps=num_steps,
      device=device,
      seed=FLAGS.noise_seed,
      eval_auto=FLAGS.auto_cleansing,
      eval_oracle=True,
      return_details=True,
      use_reversible_transform=use_reversible,
      start_step=start_step,
      ignore_first=FLAGS.ignore_first,
      quantized_checkpoints=quantized_checkpoints,
      momentum_buffer=momentum_buffer,
      variance_buffer=variance_buffer,
      quantization_scale=quantization_scale,
  )

  results = cleansing_details['results']

  if FLAGS.auto_cleansing:
    logging.info('=== AUTOMATED COUNTERFACTUAL CLEANSING SUMMARY ===')
    primary_method = methods[0] if methods else 'adam_recursive'
    auto_res = cleansing_details['auto_cleansing'].get(primary_method)
    if auto_res:
      logging.info(
          'Baseline Test Loss / Acc: %.4f / %.4f',
          auto_res['baseline']['test_loss'],
          auto_res['baseline']['test_acc'],
      )
      logging.info(
          'Cleansed Test Loss / Acc: %.4f / %.4f',
          auto_res['cleansed']['test_loss'],
          auto_res['cleansed']['test_acc'],
      )
      logging.info(
          'Test Loss Change: %+.4f, Test Acc Change: %+.4f',
          auto_res['delta']['test_loss_change'],
          auto_res['delta']['test_acc_change'],
      )
      if 'oracle' in auto_res:
        logging.info(
            'Corrupted-Removed Test Loss / Acc: %.4f / %.4f',
            auto_res['oracle']['test_loss'],
            auto_res['oracle']['test_acc'],
        )
        logging.info(
            'Corrupted-Removed Loss Change: %+.4f, Acc Change: %+.4f',
            auto_res['oracle_delta']['test_loss_change'],
            auto_res['oracle_delta']['test_acc_change'],
        )
      if 'randomized_cleansed' in auto_res:
        for s, r_cleansed in auto_res['randomized_cleansed'].items():
          logging.info(
              'Randomized Cleansed (seed=%s) Test Loss / Acc: %.4f / %.4f'
              ' (Change: %+.4f / %+.4f)',
              s,
              r_cleansed['test_loss'],
              r_cleansed['test_acc'],
              auto_res['randomized_delta'][s]['test_loss_change'],
              auto_res['randomized_delta'][s]['test_acc_change'],
          )
      det = auto_res['detection_metrics']
      logging.info(
          'Corrupted Detection Precision: %.2f, Recall: %.2f, F1: %.2f'
          ' (%d/%d ground truth detected)',
          det['precision'],
          det['recall'],
          det['f1_score'],
          det['true_positives'],
          det['num_corrupted'],
      )

      summary_csv = os.path.join(
          FLAGS.output_dir, 'counterfactual_cleansing_summary.csv'
      )
      with pipeline._open_file(summary_csv, 'w') as f:
        writer = csv.writer(f)
        has_oracle = 'oracle' in auto_res
        header = ['metric', 'baseline', 'cleansed']
        if has_oracle:
          header.append('remove_corrupted')
        header.append('delta')
        if has_oracle:
          header.append('remove_corrupted_delta')
        if retrain_seeds and 'randomized_cleansed' in auto_res:
          for s in retrain_seeds:
            header.extend([f'cleansed_seed_{s}', f'delta_seed_{s}'])
            if (
                has_oracle
                and 'randomized_oracle' in auto_res
                and s in auto_res['randomized_oracle']
            ):
              header.extend([
                  f'remove_corrupted_seed_{s}',
                  f'remove_corrupted_delta_seed_{s}',
              ])
        writer.writerow(header)

        for metric_key in ['train_loss', 'val_loss', 'test_loss', 'test_acc']:
          row = [
              metric_key,
              auto_res['baseline'][metric_key],
              auto_res['cleansed'][metric_key],
          ]
          if has_oracle:
            row.append(auto_res['oracle'][metric_key])
          delta_key = (
              f'{metric_key}_change'
              if f'{metric_key}_change' in auto_res['delta']
              else metric_key
          )
          row.append(
              auto_res['delta'].get(
                  delta_key,
                  auto_res['cleansed'][metric_key]
                  - auto_res['baseline'][metric_key],
              )
          )
          if has_oracle:
            row.append(
                auto_res['oracle_delta'].get(
                    delta_key,
                    auto_res['oracle'][metric_key]
                    - auto_res['baseline'][metric_key],
                )
            )
          if retrain_seeds and 'randomized_cleansed' in auto_res:
            for s in retrain_seeds:
              if s in auto_res['randomized_cleansed']:
                r_m = auto_res['randomized_cleansed'][s]
                r_d = auto_res['randomized_delta'][s]
                row.extend([
                    r_m[metric_key],
                    r_d.get(
                        delta_key,
                        r_m[metric_key] - auto_res['baseline'][metric_key],
                    ),
                ])
                if (
                    has_oracle
                    and 'randomized_oracle' in auto_res
                    and s in auto_res['randomized_oracle']
                ):
                  r_o = auto_res['randomized_oracle'][s]
                  r_od = auto_res['randomized_oracle_delta'][s]
                  row.extend([
                      r_o[metric_key],
                      r_od.get(
                          delta_key,
                          r_o[metric_key] - auto_res['baseline'][metric_key],
                      ),
                  ])
          writer.writerow(row)

        det_padding = [''] * (len(header) - 2)
        writer.writerow(['detection_precision', det['precision']] + det_padding)
        writer.writerow(['detection_recall', det['recall']] + det_padding)
        writer.writerow(['detection_f1', det['f1_score']] + det_padding)

  gt_cf_res = None
  if FLAGS.eval_counterfactuals:
    logging.info(
        'Evaluating ground truth leave-one-out counterfactuals & cleansing'
        ' (max_workers=%d)...',
        FLAGS.max_workers,
    )
    gt_cf_res = (
        data_cleansing.evaluate_ground_truth_counterfactuals_and_cleansing(
            model_fn=model_fn,
            train_dataset=train_ds,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            checkpoints=checkpoints,
            scores_dict=scores_dict,
            corrupted_indices=corrupted_indices,
            momentum_lst=momentum_lst,
            variance_lst=variance_lst,
            optimizer_type=FLAGS.optimizer_type,
            methods=methods,
            sample_indices=shard_indices,
            num_epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size,
            lr=FLAGS.learning_rate,
            momentum=FLAGS.momentum,
            beta_1=FLAGS.beta_1,
            beta_2=FLAGS.beta_2,
            eps=FLAGS.eps,
            criterion=criterion,
            k_list=k_list,
            retrain_seeds=retrain_seeds,
            random_mask=random_mask,
            max_workers=FLAGS.max_workers,
            device=device,
            output_dir=FLAGS.output_dir,
            use_reversible_transform=use_reversible,
            start_step=start_step,
            ignore_first=FLAGS.ignore_first,
            quantized_checkpoints=quantized_checkpoints,
            momentum_buffer=momentum_buffer,
            variance_buffer=variance_buffer,
            quantization_scale=quantization_scale,
        )
    )

  if FLAGS.eval_counterfactuals and gt_cf_res is not None:
    results.update(gt_cf_res['topk_actual_results'])
  elif FLAGS.output_dir:
    summary_csv = os.path.join(
        FLAGS.output_dir, 'counterfactual_estimation_summary.csv'
    )
    summary_rows = []
    fieldnames = [
        'method',
        'val_loss_mae',
        'test_loss_mae',
        'val_loss_corr',
        'test_loss_corr',
        'detection_precision',
        'detection_recall',
        'detection_f1',
        'mean_val_loss_change_clean',
        'mean_val_loss_change_corrupted',
        'roc_auc',
    ]
    corrupted_set = (
        set(corrupted_indices) if corrupted_indices is not None else set()
    )
    if corrupted_set:
      summary_rows.append({
          'method': 'remove_corrupted',
          'val_loss_mae': '',
          'test_loss_mae': '',
          'val_loss_corr': '',
          'test_loss_corr': '',
          'detection_precision': 1.0,
          'detection_recall': 1.0,
          'detection_f1': 1.0,
          'mean_val_loss_change_clean': '',
          'mean_val_loss_change_corrupted': '',
          'roc_auc': 1.0,
      })

    det_metrics_map = cleansing_details.get('detection_metrics', {})
    for m_name in methods:
      if m_name in det_metrics_map:
        det = det_metrics_map[m_name]
        summary_rows.append({
            'method': m_name,
            'val_loss_mae': '',
            'test_loss_mae': '',
            'val_loss_corr': '',
            'test_loss_corr': '',
            'detection_precision': det['precision'],
            'detection_recall': det['recall'],
            'detection_f1': det['f1_score'],
            'mean_val_loss_change_clean': det.get('mean_score_clean', 0.0),
            'mean_val_loss_change_corrupted': det.get(
                'mean_score_corrupted', 0.0
            ),
            'roc_auc': det.get('roc_auc', 0.5),
        })

    if summary_rows:
      with pipeline._open_file(summary_csv, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
      logging.info(
          'Wrote multi-method counterfactual estimation summary to %s',
          summary_csv,
      )

  csv_path = os.path.join(FLAGS.output_dir, 'data_cleansing_results.csv')
  logging.info('Writing results to %s', csv_path)
  with pipeline._open_file(csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow([
        'method',
        'k',
        'train_loss',
        'val_loss',
        'test_loss',
        'train_acc',
        'val_acc',
        'test_acc',
    ])
    for key, metrics in results.items():
      if key == 'baseline':
        method, k = 'baseline', 0
      else:
        method, k = key
      writer.writerow([method, k] + list(metrics))

  logging.info('Section 7.2 Data Cleansing evaluation completed successfully.')


if __name__ == '__main__':
  app.run(main)

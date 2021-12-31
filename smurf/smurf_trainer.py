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

"""Trains and evaluates smurf on various datasets."""

from absl import flags
from absl import logging

# pylint:disable=g-bad-import-order
import sys
import time
import tensorflow as tf

# pylint:disable=unused-import
from smurf import smurf_augmentation
from smurf import smurf_data
from smurf import smurf_flags
from smurf import smurf_plotting
from smurf.smurf_net import SMURFNet

FLAGS = flags.FLAGS


def set_virtual_gpus_to_at_least(num_virtual_gpus):
  """Creates virtual GPU devices if they haven't yet been created.

  This is useful testing tf.distribute strategies locally.

  Args:
    num_virtual_gpus: int, the number of virtual gpus to run with

  Returns: None

  Raises:
    RuntimeError: if no GPUs are found or already have configured virtual
      GPUs.
    ValueError: if num_virtual_gpus is < 1
  """
  if num_virtual_gpus < 1:
    raise ValueError('`num_virtual_gpus` must be at least 1 not %r' %
                     (num_virtual_gpus,))
  config = tf.config
  physical_devices = config.list_physical_devices('GPU')
  if not physical_devices:
    raise RuntimeError('No GPUs found')
  configs = config.get_logical_device_configuration(physical_devices[0])
  if configs is None:
    logical_devices = [
        tf.python.eager.context.LogicalDeviceConfiguration(memory_limit=4000)
        for _ in range(num_virtual_gpus)
    ]
    config.set_logical_device_configuration(physical_devices[0],
                                            logical_devices)
  else:
    if len(configs) < num_virtual_gpus:
      raise RuntimeError('Already configured with %d < %d virtual GPUs' %
                         (len(configs), num_virtual_gpus))


def learning_rate_fn():
  """Controls the learning rate based on the global step."""
  start_learning_rate = FLAGS.start_learning_rate
  step = tf.cast(tf.compat.v1.train.get_or_create_global_step(), 'float32')
  effective_step = tf.maximum(step - FLAGS.lr_decay_after_num_steps + 1, 0)
  lr_step_ratio = tf.cast(effective_step, 'float32') / float(
      FLAGS.lr_decay_steps)
  warm_up_factor = tf.cast(tf.minimum(step / float(FLAGS.warm_up_steps), 1.),
                           'float32')
  final_learning_rate = FLAGS.gpu_learning_rate
  # Ease in to final learning rate.
  lr = ((1. - warm_up_factor) * start_learning_rate) + (
      warm_up_factor * final_learning_rate)
  lr = tf.cast(lr, 'float32')
  if FLAGS.lr_decay_type == 'none' or FLAGS.lr_decay_steps <= 0:
    return lr
  elif FLAGS.lr_decay_type == 'exponential':
    return lr * 0.5**lr_step_ratio
  else:
    raise ValueError('Unknown lr_decay_type', FLAGS.lr_decay_type)


def create_smurf():
  """Build the smurf model."""

  selfsup_transform = smurf_augmentation.build_selfsup_transformations(
      crop_height=FLAGS.selfsup_crop_height,
      crop_width=FLAGS.selfsup_crop_width,
      resize=FLAGS.resize_selfsup)

  smurf = SMURFNet(
      checkpoint_dir=FLAGS.checkpoint_dir,
      optimizer=FLAGS.optimizer,
      learning_rate=learning_rate_fn,
      only_forward=FLAGS.only_forward,
      dropout_rate=FLAGS.dropout_rate,
      selfsup_transform=selfsup_transform,
      fb_sigma_teacher=FLAGS.fb_sigma_teacher,
      fb_sigma_student=FLAGS.fb_sigma_student,
      train_mode=FLAGS.train_mode,
      smoothness_edge_weighting=FLAGS.smoothness_edge_weighting,
      smoothness_edge_constant=FLAGS.smoothness_edge_constant,
      teacher_image_version=FLAGS.teacher_image_version,
      stop_gradient_mask=FLAGS.stop_gradient_mask,
      selfsup_mask=FLAGS.selfsup_mask,
      feature_architecture=FLAGS.feature_architecture,
      flow_architecture=FLAGS.flow_architecture,
      size=(FLAGS.global_gpu_batch_size, FLAGS.height, FLAGS.width),
      occlusion_estimation=FLAGS.occlusion_estimation,
      smoothness_at_level=FLAGS.smoothness_at_level,
      use_float16=True,
  )
  return smurf


def clip_grad_norm(grads, max_norm):
  """Performs pytorch like gradient clipping.

  Args:
    grads: list of gradient tf.Tensor
    max_norm: float, gradients above this norm will be clipped.
  Returns:
    clipped_grads: list of gradients with clipping applied.
  """
  grads_without_none = []
  orig_indices = []
  # Remove None gradients.
  for i, g in enumerate(grads):
    if g is not None:
      orig_indices.append(i)
      grads_without_none.append(g)
  total_norm = tf.norm(
      tf.stack([tf.norm(g, ord=2) for g in grads_without_none]), ord=2)
  clip_coef = max_norm / (total_norm + 1e-6)
  new_grads = []
  for g in grads_without_none:
    new_g = tf.cond(clip_coef < 1,
                    lambda: tf.math.scalar_mul(clip_coef, g),
                    lambda: g)
    new_grads.append(new_g)

  # Add None gradients back in.
  grads = [None for _ in grads]
  for i, grad in zip(orig_indices, new_grads):
    grads[i] = grad
  return grads


def train_eval():
  """Main train and evaluation loop."""
  logging.info('Setting strategy to mirrored strategy...')
  # Synchronous SGD
  strategy = tf.distribute.MirroredStrategy()
  # Make directories if they do not exist yet.
  if FLAGS.checkpoint_dir and not tf.io.gfile.exists(FLAGS.checkpoint_dir):
    logging.info('Making new checkpoint directory: %s', FLAGS.checkpoint_dir)
    tf.io.gfile.makedirs(FLAGS.checkpoint_dir)
  if FLAGS.plot_dir and not tf.io.gfile.exists(FLAGS.plot_dir):
    logging.info('Making new plot directory: %s', FLAGS.plot_dir)
    tf.io.gfile.makedirs(FLAGS.plot_dir)

  with strategy.scope():
    logging.info('Getting train step...')
    step = tf.compat.v1.train.get_or_create_global_step()
    smurf = create_smurf()
    if not FLAGS.from_scratch:
      # First restore from init_checkpoint_dir, which is only restored from but
      # not saved to, and then restore from checkpoint_dir if there is already
      # a model there (e.g. if the run was stopped and restarted).
      if FLAGS.init_checkpoint_dir:
        logging.info('Initializing model from checkpoint %s.',
                     FLAGS.init_checkpoint_dir)
        logging.info('Restoring smurf...')
        smurf.update_checkpoint_dir(FLAGS.init_checkpoint_dir)
        smurf.restore(
            reset_optimizer=FLAGS.reset_optimizer,
            reset_global_step=FLAGS.reset_global_step)
        smurf.update_checkpoint_dir(FLAGS.checkpoint_dir)

      if FLAGS.checkpoint_dir:
        logging.info('Restoring model from checkpoint %s.',
                     FLAGS.checkpoint_dir)
        smurf.restore()
    else:
      logging.info('Starting from scratch.')

  logging.info('Making eval datasets and eval functions.')

  if FLAGS.eval_on:
    logging.info('Making eval function...')
    evaluate, _ = smurf_data.make_eval_function(
        FLAGS.eval_on,
        FLAGS.height,
        FLAGS.width,
        progress_bar=True,
        plot_dir=FLAGS.plot_dir,
        num_plots=50)

  if FLAGS.train_on:
    # Build training iterator.
    logging.info('Making training iterator.')
    train_dataset = smurf_data.make_train_dataset(
        FLAGS.train_on,
        FLAGS.height,
        FLAGS.width,
        FLAGS.shuffle_buffer_size,
        FLAGS.global_gpu_batch_size,
        FLAGS.seq_len,
        crop_instead_of_resize=FLAGS.crop_instead_of_resize,
        apply_augmentation=True,
        include_ground_truth=('unsupervised' not in FLAGS.train_mode),
        resize_gt_flow=FLAGS.resize_gt_flow_supervision,
        return_full_scale=FLAGS.full_size_warp,
    )
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    train_it = iter(train_dataset)

    if FLAGS.check_data and FLAGS.plot_dir:
      smurf_plotting.plot_data(train_it, FLAGS.plot_dir, num_plots=100)
    if FLAGS.train_mode in ('supervised', 'supervised-sequence'):
      # Since this is the only loss in this setting, and the Adam optimizer
      # is scale invariant, the actual weight here does not matter for now.
      weights = {'supervision': FLAGS.weight_supervision}
    else:
      # Note that self-supervision loss is added during training.
      weights = {
          'supervision': FLAGS.weight_supervision,
          'census': FLAGS.weight_census,
      }

      # Switch off loss-terms that have weights < 1e-7.
      weights = {
          k: v for (k, v) in weights.items() if v > 1e-7
      }

    def weight_selfsup_fn():
      step = tf.compat.v1.train.get_or_create_global_step()
      # Start self-supervision only after a certain number of steps.
      # Linearly increase self-supervision weight for a number of steps.
      ramp_up_factor = tf.clip_by_value(
          float(step - (FLAGS.selfsup_after_num_steps - 1)) /
          float(max(FLAGS.selfsup_ramp_up_steps, 1)), 0., 1.)
      return FLAGS.weight_selfsup * ramp_up_factor

    logging.info('Starting training loop.')
    epoch = 0

    while True:
      current_step = tf.compat.v1.train.get_or_create_global_step().numpy()

      # Set which occlusion estimation methods could be active at this point.
      # (They will only be used if occlusion_estimation is set accordingly.)
      occ_active = {
          'brox':
              current_step > FLAGS.occ_after_num_steps_brox,
          'wang':
              current_step > FLAGS.occ_after_num_steps_wang,
      }

      current_weights = {k: v for k, v in weights.items()}

      # Prepare self-supervision if it will be used in the next epoch.
      if (FLAGS.weight_selfsup > 1e-7 and current_step +
          FLAGS.epoch_length > FLAGS.selfsup_after_num_steps):

        # Add selfsup weight with a ramp-up schedule. This will cause a
        # recompilation of the training graph defined in smurf.train(...).
        current_weights['selfsup'] = weight_selfsup_fn

      if current_step > FLAGS.smoothness_after_num_steps:
        current_weights['smooth1'] = FLAGS.weight_smooth1
        current_weights['smooth2'] = FLAGS.weight_smooth2

      def train_step(inputs):
        weights = {
            k: v() if callable(v) else v for k, v in current_weights.items()
        }
        losses, gradients, variables = smurf.loss_and_grad(
            inputs,
            weights,
            occ_active=occ_active)
        if FLAGS.gradient_clipping:
          gradients = clip_grad_norm(gradients,
                                     FLAGS.gradient_clipping_max_value)
        smurf.optimizer.apply_gradients(
            zip(gradients, variables))
        return losses

      @tf.function
      def distributed_train_step(dist_inputs):
        per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
        output = {}
        for k, v in per_replica_losses.items():
          output[k] = strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None)
          output[k] /= FLAGS.num_gpus
          if FLAGS.log_per_replica_values:
            if hasattr(v, 'values'):
              for i, value in enumerate(v.values):
                output[k + str(i)] = value
        return output

      # Train for an epoch and save the results.
      log = {}
      log['learning-rate'] = [learning_rate_fn()]
      for step in range(FLAGS.epoch_length):
        sys.stdout.write(f'{step},')
        sys.stdout.flush()
        start_time_data = time.time()
        distributed_inputs = train_it.next()
        stop_time_data = time.time()
        global_step = tf.compat.v1.train.get_or_create_global_step()
        global_step.assign(global_step + 1)
        logging.info('Step is %d', global_step.numpy())
        start_time_train = time.time()
        log_update = distributed_train_step(distributed_inputs)
        stop_time_train = time.time()
        log_update['data-time'] = (stop_time_data - start_time_data) * 1000
        log_update['train-time'] = (stop_time_train - start_time_train) * 1000

        for key in log_update:
          if key in log:
            log[key].append(log_update[key])
          else:
            log[key] = [log_update[key]]

      if FLAGS.checkpoint_dir and not FLAGS.no_checkpointing:
        smurf.save()

      smurf_plotting.print_log(log, epoch)
      if FLAGS.eval_on and FLAGS.evaluate_during_train:
        eval_results = evaluate(smurf)
        smurf_plotting.print_eval(eval_results)

      if current_step >= FLAGS.num_train_steps:
        break

      epoch += 1

  else:
    logging.info(
        'Specify flag train_on to enable training to <format>:<path>;... .')
    logging.info('Just doing evaluation now.')
    eval_results = evaluate(smurf)
    if eval_results:
      logging.info(smurf_plotting.print_eval(eval_results))
    logging.info('Evaluation complete.')

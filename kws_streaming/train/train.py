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

"""Train utility functions, based on tensorflow/examples/speech_commands.

  It consists of several steps:
  1. Creates model.
  2. Reads data
  3. Trains model
  4. Select the best model and evaluates it
"""

import os.path
import pprint
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
import kws_streaming.data.input_data as input_data
from kws_streaming.models import models
from kws_streaming.models import utils


def train(flags):
  """Model training."""

  flags.training = True

  # Set the verbosity based on flags (default is INFO, so we see all messages)
  logging.set_verbosity(flags.verbosity)

  # Start a new TensorFlow session.
  tf.reset_default_graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)

  audio_processor = input_data.AudioProcessor(flags)

  time_shift_samples = int((flags.time_shift_ms * flags.sample_rate) / 1000)

  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, flags.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, flags.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))
  logging.info(flags)
  model = models.MODELS[flags.model_name](flags)
  logging.info(model.summary())

  # fix for InvalidArgumentError:
  # Node 'Adam/gradients/gradients/gru/cell/while_grad/gru/cell/while_grad':
  # Connecting to invalid output 51 of source node
  # gru/cell/while which has 51 outputs.
  if flags.model_name in ['crnn', 'gru', 'lstm']:
    tf.compat.v1.experimental.output_all_intermediates(True)

  # save model summary
  utils.save_model_summary(model, flags.train_dir)

  # save model and data flags
  with open(os.path.join(flags.train_dir, 'flags.txt'), 'wt') as f:
    pprint.pprint(flags, stream=f)

  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.Adam(epsilon=flags.optimizer_epsilon)

  if flags.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(epsilon=flags.optimizer_epsilon)
  elif flags.optimizer == 'momentum':
    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
  elif flags.optimizer == 'novograd':
    optimizer = tfa.optimizers.NovoGrad(
        lr=0.05,
        beta_1=flags.novograd_beta_1,
        beta_2=flags.novograd_beta_2,
        weight_decay=flags.novograd_weight_decay,
        grad_averaging=bool(flags.novograd_grad_averaging))
  else:
    raise ValueError('Unsupported optimizer:%s' % flags.optimizer)

  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  train_writer = tf.summary.FileWriter(
      os.path.join(flags.summaries_dir, 'train'), sess.graph)
  validation_writer = tf.summary.FileWriter(
      os.path.join(flags.summaries_dir, 'validation'))

  start_step = 1

  logging.info('Training from step: %d ', start_step)

  # Save graph.pbtxt.
  tf.train.write_graph(sess.graph_def, flags.train_dir, 'graph.pbtxt')

  # Save list of words.
  with tf.io.gfile.GFile(os.path.join(flags.train_dir, 'labels.txt'), 'w') as f:
    f.write('\n'.join(audio_processor.words_list))

  best_accuracy = 0.0

  # prepare parameters for exp learning rate decay
  training_steps_max = np.sum(training_steps_list)
  lr_init = learning_rates_list[0]
  exp_rate = -np.log(learning_rates_list[-1] / lr_init)/training_steps_max

  # configure checkpointer
  checkpoint_directory = os.path.join(flags.train_dir, 'restore/')
  checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

  sess.run(tf.global_variables_initializer())
  status.initialize_or_restore(sess)

  # Training loop.
  for training_step in range(start_step, training_steps_max + 1):
    offset = (training_step -
              1) * flags.batch_size if flags.pick_deterministically else 0
    # Pull the audio samples we'll use for training.
    train_fingerprints, train_ground_truth = audio_processor.get_data(
        flags.batch_size, offset, flags, flags.background_frequency,
        flags.background_volume, time_shift_samples, 'training',
        flags.resample, flags.volume_resample, sess)

    if flags.lr_schedule == 'exp':
      learning_rate_value = lr_init * np.exp(-exp_rate * training_step)
    elif flags.lr_schedule == 'linear':
      # Figure out what the current learning rate is.
      training_steps_sum = 0
      for i in range(len(training_steps_list)):
        training_steps_sum += training_steps_list[i]
        if training_step <= training_steps_sum:
          learning_rate_value = learning_rates_list[i]
          break
    else:
      raise ValueError('Wrong lr_schedule: %s' % flags.lr_schedule)

    tf.keras.backend.set_value(model.optimizer.lr, learning_rate_value)
    result = model.train_on_batch(train_fingerprints, train_ground_truth)

    summary = tf.Summary(value=[
        tf.Summary.Value(tag='accuracy', simple_value=result[1]),
    ])
    train_writer.add_summary(summary, training_step)

    logging.info(
        'Step #%d: rate %f, accuracy %.2f%%, cross entropy %f',
        *(training_step, learning_rate_value, result[1] * 100, result[0]))

    is_last_step = (training_step == training_steps_max)
    if (training_step % flags.eval_step_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('validation')
      set_size = int(set_size / flags.batch_size) * flags.batch_size
      total_accuracy = 0.0
      count = 0.0
      for i in range(0, set_size, flags.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(flags.batch_size, i, flags, 0.0, 0.0, 0,
                                     'validation', 0.0, 0.0, sess))

        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        result = model.test_on_batch(validation_fingerprints,
                                     validation_ground_truth)

        summary = tf.Summary(value=[
            tf.Summary.Value(tag='accuracy', simple_value=result[1]),])

        validation_writer.add_summary(summary, training_step)

        total_accuracy += result[1]
        count = count + 1.0

      total_accuracy = total_accuracy / count
      logging.info('Step %d: Validation accuracy = %.2f%% (N=%d)',
                   *(training_step, total_accuracy * 100, set_size))

      model.save_weights(
          os.path.join(
              flags.train_dir, 'train/',
              str(int(best_accuracy * 10000)) + 'weights_' +
              str(training_step)))

      # Save the model checkpoint when validation accuracy improves
      if total_accuracy >= best_accuracy:
        best_accuracy = total_accuracy
        # overwrite the best model weights
        model.save_weights(os.path.join(flags.train_dir, 'best_weights'))

        # save checkpoint
        checkpoint.save(file_prefix=checkpoint_prefix, session=sess)

      logging.info('So far the best validation accuracy is %.2f%%',
                   (best_accuracy * 100))

  tf.keras.backend.set_learning_phase(0)
  set_size = audio_processor.set_size('testing')
  set_size = int(set_size / flags.batch_size) * flags.batch_size
  logging.info('set_size=%d', set_size)
  total_accuracy = 0.0
  count = 0.0

  for i in range(0, set_size, flags.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        flags.batch_size, i, flags, 0.0, 0.0, 0, 'testing', 0.0, 0.0, sess)

    result = model.test_on_batch(test_fingerprints, test_ground_truth)

    total_accuracy += result[1]
    count = count + 1.0
  total_accuracy = total_accuracy / count

  logging.info('Final test accuracy = %.2f%% (N=%d)',
               *(total_accuracy * 100, set_size))
  with open(os.path.join(flags.train_dir, 'accuracy_last.txt'), 'wt') as fd:
    fd.write(str(total_accuracy * 100))
  model.save_weights(os.path.join(flags.train_dir, 'last_weights'))

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

"""Train the Contrack model."""

import logging
import os

from absl import app
from absl import flags

import tensorflow as tf

from contrack import data
from contrack import encoding
from contrack import env
from contrack import model

flags.DEFINE_string('model_path', '',
                    'Base output directory where the model is stored.')
flags.DEFINE_string('config_path', '', 'File path of config json file.')
flags.DEFINE_string(
    'config_json', '',
    'The contents of a json config file if --config_file was not provided.')
flags.DEFINE_string(
    'mode', '',
    'How to train the model, either "only_new_entities", "only_tracking", "full" or "two_steps".'
)
flags.DEFINE_string(
    'train_data_glob', '',
    'A TF glob pattern specifying the location of the training data files.')
flags.DEFINE_string(
    'eval_data_glob', '',
    'A TF glob pattern specifying the location of the validation data files.')
FLAGS = flags.FLAGS


def train(argv):
  """Train a contrack model."""
  del argv  # Unused.

  mode = FLAGS.mode
  if FLAGS.config_path:
    config = env.ContrackConfig.load_from_path(FLAGS.config_path)
  elif FLAGS.config_json:
    config = env.ContrackConfig.load_from_json(FLAGS.config_json)
  else:
    raise ValueError('Must provide --config_path or --config_json')

  logging.info('Training with config:\n%s', config)
  encodings = encoding.Encodings()
  env.Env.init(config, encodings)
  environment = env.Env.get()

  logging.info('Reading training data from %s', FLAGS.train_data_glob)
  train_data = data.read_training_data(FLAGS.train_data_glob, config, encodings)

  if FLAGS.eval_data_glob:
    logging.info('Reading validation data from %s', FLAGS.eval_data_glob)
    eval_data = data.read_eval_data(FLAGS.eval_data_glob, config, encodings)
  else:
    eval_data = None

  tensorboard_dir = os.path.join(FLAGS.model_path, 'tensorboard')
  checkpoint_dir = os.path.join(FLAGS.model_path, 'checkpoints')
  callbacks = [
      tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir),
      tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir),
      tf.keras.callbacks.TerminateOnNaN()
  ]


  # Compile model
  if mode == 'only_new_entities' or mode == 'full' or mode == 'only_tracking':
    contrack_model = model.ContrackModel(mode)
    loss = model.ContrackLoss(mode)
    metrics = model.build_metrics(mode)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    contrack_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Do the actual training
    contrack_model.fit(
        x=train_data,
        epochs=int(config.max_steps / config.steps_per_epoch),
        callbacks=callbacks,
        steps_per_epoch=config.steps_per_epoch,
        validation_data=eval_data)
  elif mode == 'two_steps':
    logging.info('Training new entity model...')
    new_id_model = model.ContrackModel('only_new_entities')
    loss = model.ContrackLoss('only_new_entities')
    metrics = model.build_metrics('only_new_entities')
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate, clipnorm=1.0)
    new_id_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    new_id_model.fit(
        x=train_data,
        epochs=int(config.max_steps / config.steps_per_epoch),
        callbacks=callbacks,
        steps_per_epoch=config.steps_per_epoch,
        validation_data=eval_data)

    logging.info('Training tracking model...')
    contrack_model = model.ContrackModel('only_tracking')
    loss = model.ContrackLoss('only_tracking')
    metrics = model.build_metrics('only_tracking')
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    contrack_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    contrack_model.init_weights_from_new_entity_model(new_id_model)
    contrack_model.fit(
        x=train_data,
        epochs=int(config.max_steps / config.steps_per_epoch),
        callbacks=callbacks,
        steps_per_epoch=config.steps_per_epoch,
        validation_data=eval_data)
  else:
    raise ValueError('Unknown mode "%s"' % mode)

  # Save it
  filepath = FLAGS.model_path
  with tf.keras.utils.custom_object_scope(model.get_custom_objects()):
    tf.keras.models.save_model(contrack_model, filepath)
    environment.config.save(filepath)
    environment.encodings.save(filepath)


if __name__ == '__main__':
  app.run(train)

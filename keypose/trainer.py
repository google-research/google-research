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

"""Keypose estimator training / eval functions.

Adapted from 'Discovery of Latent 3D Keypoints via End-to-end
Geometric Reasoning' keypoint network.

Given a 2D image and viewpoint, predict a set of 3D keypoints that
match the target examples.

Can be instance or class specific, depending on training set.

Typical invocation:
$ python3 -m keypose.trainer.py configs/bottle_0_t5 /tmp/model
"""

import os
import sys

from tensorflow import estimator as tf_estimator
from tensorflow import keras

from keypose import estimator as est
from keypose import inputs as inp
from keypose import utils


def train_and_eval(params,
                   model_fn,
                   input_fn,
                   keep_checkpoint_every_n_hours=0.5,
                   save_checkpoints_secs=100,
                   eval_steps=0,
                   eval_start_delay_secs=10,
                   eval_throttle_secs=100,
                   save_summary_steps=50):
  """Trains and evaluates our model.

  Supports local and distributed training.

  Args:
    params: ConfigParams class with model training and network parameters.
    model_fn: A func with prototype model_fn(features, labels, mode, hparams).
    input_fn: A input function for the tf.estimator.Estimator.
    keep_checkpoint_every_n_hours: Number of hours between each checkpoint to be
      saved.
    save_checkpoints_secs: Save checkpoints every this many seconds.
    eval_steps: Number of steps to evaluate model; 0 for one epoch.
    eval_start_delay_secs: Start evaluating after waiting for this many seconds.
    eval_throttle_secs: Do not re-evaluate unless the last evaluation was
      started at least this many seconds ago
    save_summary_steps: Save summaries every this many steps.
  """

  mparams = params.model_params

  run_config = tf_estimator.RunConfig(
      keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
      save_checkpoints_secs=save_checkpoints_secs,
      save_summary_steps=save_summary_steps)

  if run_config.model_dir:
    params.model_dir = run_config.model_dir
  print('\nCreating estimator with model dir %s' % params.model_dir)
  estimator = tf_estimator.Estimator(
      model_fn=model_fn,
      model_dir=params.model_dir,
      config=run_config,
      params=params)

  print('\nCreating train_spec')
  train_spec = tf_estimator.TrainSpec(
      input_fn=input_fn(params, split='train'), max_steps=params.steps)

  print('\nCreating eval_spec')

  def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders.

    Returns:
      A tf.estimator.export.ServingInputReceiver.
    """
    modelx = mparams.modelx
    modely = mparams.modely
    offsets = keras.Input(shape=(3,), name='offsets', dtype='float32')
    hom = keras.Input(shape=(3, 3), name='hom', dtype='float32')
    to_world = keras.Input(shape=(4, 4), name='to_world_L', dtype='float32')
    img_l = keras.Input(
        shape=(modely, modelx, 3), name='img_L', dtype='float32')
    img_r = keras.Input(
        shape=(modely, modelx, 3), name='img_R', dtype='float32')
    features = {
        'img_L': img_l,
        'img_R': img_r,
        'to_world_L': to_world,
        'offsets': offsets,
        'hom': hom
    }
    return tf_estimator.export.build_raw_serving_input_receiver_fn(features)

  class SaveModel(tf_estimator.SessionRunHook):
    """Saves a model in SavedModel format."""

    def __init__(self, estimator, output_dir):
      self.output_dir = output_dir
      self.estimator = estimator
      self.save_num = 0

    def begin(self):
      ckpt = self.estimator.latest_checkpoint()
      print('Latest checkpoint in hook:', ckpt)
      ckpt_num_str = ckpt.split('.ckpt-')[1]
      if (int(ckpt_num_str) - self.save_num) > 4000:
        fname = os.path.join(self.output_dir, 'saved_model-' + ckpt_num_str)
        print('**** Saving model in train hook: %s' % fname)
        self.estimator.export_saved_model(fname, serving_input_receiver_fn())
        self.save_num = int(ckpt_num_str)

  saver_hook = SaveModel(estimator, params.model_dir)

  if eval_steps == 0:
    eval_steps = None
  eval_spec = tf_estimator.EvalSpec(
      input_fn=input_fn(params, split='val'),
      steps=eval_steps,
      hooks=[saver_hook],
      start_delay_secs=eval_start_delay_secs,
      throttle_secs=eval_throttle_secs)

  if run_config.is_chief:
    outdir = params.model_dir
    if outdir is not None:
      print('Writing params to %s' % outdir)
      os.makedirs(outdir, exist_ok=True)
      params.write_yaml(os.path.join(outdir, 'params.yaml'))

  print('\nRunning estimator')
  tf_estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  print('\nSaving last model')
  ckpt = estimator.latest_checkpoint()
  print('Last checkpoint:', ckpt)
  ckpt_num_str = ckpt.split('.ckpt-')[1]
  fname = os.path.join(params.model_dir, 'saved_model-' + ckpt_num_str)
  print('**** Saving last model: %s' % fname)
  estimator.export_saved_model(fname, serving_input_receiver_fn())


def main(argv):
  if not len(argv) >= 2:
    print('Usage: ./trainer.py <config_file, e.g., configs/bottle_0_t5> '
          '[model_dir (/tmp/model)]')
    exit(0)

  config_file = argv[1]
  if len(argv) > 2:
    model_dir = argv[2]
  else:
    model_dir = '/tmp/model'

  fname = os.path.join(utils.KEYPOSE_PATH, config_file + '.yaml')
  with open(fname, 'r') as f:
    params, _, _ = utils.get_params(param_file=f)
  dset_dir = os.path.join(utils.KEYPOSE_PATH, params.dset_dir)
  # Configuration has the dset directory, now get more info from there.
  with open(fname, 'r') as f:
    params, _, _ = utils.get_params(
        param_file=f,
        cam_file=os.path.join(os.path.join(dset_dir, 'data_params.pbtxt')))
  params.model_dir = model_dir
  params.dset_dir = dset_dir

  print('Parameters to train and eval:\n', params.make_dict())

  train_and_eval(
      params,
      model_fn=est.est_model_fn,
      input_fn=inp.create_input_fn,
      save_checkpoints_secs=600,
      eval_throttle_secs=600,
      eval_steps=1000,
  )


if __name__ == '__main__':
  main(sys.argv)

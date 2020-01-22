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

# Lint as: python3
r"""Testing script for data iterator and training for implementing ROAR.

"""
from absl import flags
from absl.testing import absltest
import tensorflow.compat.v1 as tf
from interpretability_benchmark import data_input
from interpretability_benchmark.train_resnet import imagenet_params
from interpretability_benchmark.train_resnet import resnet_model_fn

flags.DEFINE_string('dest_dir', '/tmp/saliency/',
                    'Pathway to directory where output is saved.')

# model params
FLAGS = flags.FLAGS


class TrainSaliencyTest(absltest.TestCase):

  def testEndToEnd(self):

    params = imagenet_params

    params['output_dir'] = '/tmp/'
    params['batch_size'] = 2
    params['num_train_steps'] = 1
    params['eval_steps'] = 1
    params['threshold'] = 80.
    params['data_format'] = 'channels_last'
    mean_stats = [0.485, 0.456, 0.406]
    std_stats = [0.229, 0.224, 0.225]
    update_params = {
        'mean_rgb': mean_stats,
        'stddev_rgb': std_stats,
        'lr_schedule': [  # (multiplier, epoch to start) tuples
            (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
        ],
        'momentum': 0.9,
        'data_format': 'channels_last'
    }
    params.update(update_params)

    dataset_ = data_input.DataIterator(
        mode=FLAGS.mode,
        data_directory='',
        saliency_method='ig_smooth_2',
        transformation='modified_image',
        threshold=params['threshold'],
        keep_information=False,
        use_squared_value=True,
        mean_stats=mean_stats,
        std_stats=std_stats,
        test_small_sample=True,
        num_cores=FLAGS.num_cores)

    images, labels = dataset_.input_fn(params)
    self.assertEqual(images.shape.as_list(), [2, 224, 224, 3])
    self.assertEqual(labels.shape.as_list(), [
        2,
    ])

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.dest_dir,
        save_checkpoints_steps=FLAGS.steps_per_checkpoint)

    classifier = tf.estimator.Estimator(
        model_fn=resnet_model_fn,
        model_dir=FLAGS.dest_dir,
        params=params,
        config=run_config)
    classifier.train(input_fn=dataset_.input_fn, max_steps=1)
    tf.logging.info('finished training.')


if __name__ == '__main__':
  absltest.main()

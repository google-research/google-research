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

# Lint as: python3
"""Training and evaluation binary that is implemented as a custom loop."""

from absl import app
from absl import flags

from deep_representation_one_class import train_and_eval_lib

flags.DEFINE_string(
    name='model_dir',
    default=None,
    help='Path to output model directory where event and checkpoint files will '
    'be written.')
flags.DEFINE_string(name='root', default=None, help='Path to root data path.')
flags.DEFINE_string(name='method', default='UnsupEmbed', help='dataset')

# data
flags.DEFINE_string(name='dataset', default='cifar10ood', help='dataset')
flags.DEFINE_string(name='category', default='', help='category')
flags.DEFINE_boolean(name='is_validation', default=False, help='validation')
flags.DEFINE_string(
    name='aug_list',
    default='hflip+jitter,hflip+jitter+cutout0.3',
    help='input augmentation list')
flags.DEFINE_string(
    name='aug_list_for_test', default=None, help='input augmentation list')
flags.DEFINE_string(
    name='input_shape', default='32,32,3', help='data input shape')
flags.DEFINE_string(
    name='distaug_type',
    default='1',
    help='number of distribution augmentation')

# network architecture
flags.DEFINE_string(
    name='net_type',
    default='ResNet18',
    help='network type (see model/__init__.py)')
flags.DEFINE_float(
    name='net_width', default=1, help='network width (# channnels)')
flags.DEFINE_string(name='head_dims', default=None, help='MLP architecture')
flags.DEFINE_integer(
    name='latent_dim', default=2, help='number of hidden units for FC layer')

# optimization
flags.DEFINE_integer(name='seed', default=0, help='random seed')
flags.DEFINE_boolean(
    name='force_init', default=False, help='force training from scratch')
flags.DEFINE_string(
    name='optim_type', default='sgd', help='stochastic optimizer')
flags.DEFINE_string(
    name='sched_type', default='cos', help='learning rate scheduler')
flags.DEFINE_string(
    name='sched_freq',
    default='epoch',
    help='update frequency. `step` or `epoch`')
flags.DEFINE_integer(
    name='sched_step_size', default=1, help='step size for step LR')
flags.DEFINE_float(name='sched_gamma', default=0.995, help='gamma for step LR')
flags.DEFINE_float(
    name='sched_min_rate', default=0.0, help='minimum rate for cosine LR')
flags.DEFINE_integer(
    name='sched_level', default=7, help='level for half-cosine cycle')
flags.DEFINE_float(name='learning_rate', default=0.3, help='learning rate')
flags.DEFINE_float(name='weight_decay', default=0.00001, help='weight decay')
flags.DEFINE_boolean(
    name='regularize_bn', default=False, help='regularize BN parameters')
flags.DEFINE_float(name='momentum', default=0.9, help='momentum')
flags.DEFINE_boolean(name='nesterov', default=False, help='nesterov')
flags.DEFINE_integer(
    name='num_epoch', default=2048, help='number of training epochs')
flags.DEFINE_integer(
    name='num_batch', default=0, help='number of batches per epoch')
flags.DEFINE_integer(name='batch_size', default=64, help='batch size')

# monitoring and checkpoint
flags.DEFINE_string(name='ckpt_prefix', default='', help='checkpoint prefix')

flags.DEFINE_integer(
    name='ckpt_epoch', default=32, help='frequency to save checkpoints')
flags.DEFINE_string(name='file_path', default=None, help='file path')
flags.DEFINE_float(name='temperature', default=0.1, help='Temperature')

flags.mark_flag_as_required('model_dir')

FLAGS = flags.FLAGS


class HParams(dict):

  def __init__(self, *args, **kwargs):
    super(HParams, self).__init__(*args, **kwargs)
    self.__dict__ = self


def main(unused_argv):
  hparams = HParams({
      flag.name: flag.value for flag in FLAGS.get_flags_for_module('__main__')
  })
  # start training
  trainer = train_and_eval_lib.get_trainer(hparams)
  with trainer.strategy.scope():
    trainer.config()
    trainer.train()


if __name__ == '__main__':
  app.run(main)

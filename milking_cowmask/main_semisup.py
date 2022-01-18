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

"""Launch 'Milking CowMask for semi-supervised image classification'."""

import ast

from absl import app
from absl import flags
from absl import logging


import train_semisup


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir', default='checkpoint',
    help=('Directory to store model data'))

flags.DEFINE_string(
    'imagenet_subset_dir', default=None,
    help=('Directory to store model data'))

flags.DEFINE_string(
    'dataset', default='cifar10',
    help=('Dataset to use (cifar10|cifar100|svhn|imagenet)'))

flags.DEFINE_integer(
    'batch_size', default=256,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'eval_batch_size', default=1000,
    help=('Batch size for evaluation.'))

flags.DEFINE_integer(
    'num_epochs', default=300,
    help=('Number of training epochs.'))

flags.DEFINE_float(
    'learning_rate', default=0.05,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_bool(
    'aug_imagenet_apply_colour_jitter', default=False,
    help=('ImageNet augmentation: apply colour jitter.'))

flags.DEFINE_float(
    'aug_imagenet_greyscale_prob', default=0.0,
    help=('ImageNet augmentation: probability to convert image to greyscale.'))

flags.DEFINE_float(
    'sgd_momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_bool(
    'sgd_nesterov', default=True,
    help=('Use Nesterov momentum.'))

flags.DEFINE_string(
    'lr_schedule', default='stepped',
    help=('Learning rate schedule type; (constant|stepped|cosine)'))

flags.DEFINE_string(
    'lr_sched_steps', default='[[120, 0.2], [240, 0.04]]',
    help=('Learning rate schedule steps as a Python list; '
          '[[step1_epoch, step1_lr_scale], '
          '[step2_epoch, step2_lr_scale], ...]'))

flags.DEFINE_integer(
    'lr_sched_halfcoslength', default=300,
    help=('Length of cosine learning rate annealing half-cycle'))

flags.DEFINE_float(
    'lr_sched_warmup', default=0.0,
    help=('Learning rate schedule warmup length in epochs.'))

flags.DEFINE_float(
    'l2_reg', default=0.0005,
    help=('The amount of L2-regularization to apply.'))

flags.DEFINE_float(
    'weight_decay', default=0.0,
    help=('The amount of weight decay to apply.'))

flags.DEFINE_string(
    'architecture', default='wrn26_6_shakeshake',
    help=('Network architecture (wrn20_10|wrn26_10|wrn26_2|wrn20_6_shakeshake'
          '|wrn26_6_shakeshake|wrn26_2_shakeshake|pyramid|resnet50|resnet101'
          '|resnet152|resnet50x2|resnet101x2|resnet152x2|resnet50x4'
          '|resnet101x4|resnet152x4|resnext50_32x4d|resnext101_32x8d'
          '|resnext152_32x4d).'))

flags.DEFINE_integer(
    'n_val', default=0,
    help=('Number of samples to split off the training set for validation.'))

flags.DEFINE_integer(
    'n_sup', default=1000,
    help=('Number of samples to be used for supervised loss (-1 for all).'))

flags.DEFINE_float(
    'teacher_alpha', default=0.97,
    help=('Teacher EMA alpha.'))

flags.DEFINE_bool(
    'anneal_teacher_alpha', default=False,
    help=('Anneal 1-teacher_alpha using the learning rate schedule '
          '(no warmup).'))

flags.DEFINE_string(
    'unsup_reg', default='none',
    help=('Unsupervised/perturbation regularizer '
          '(none|mt|aug|cutout|aug_cutout|cowout|aug_cowout).'))

flags.DEFINE_float(
    'cons_weight', default=1.0,
    help=('Consistency (perturbation) loss weight.'))

flags.DEFINE_float(
    'conf_thresh', default=0.97,
    help=('Consistency (perturbation) confidence threshold.'))

flags.DEFINE_bool(
    'conf_avg', default=False,
    help=('Consistency (perturbation) confidence mask averaging.'))

flags.DEFINE_float(
    'cut_backg_noise', default=1.0,
    help=('Consistency (perturbation) cut background noise (e.g. 1.0 for '
          'RandErase).'))

flags.DEFINE_float(
    'cut_prob', default=1.0,
    help=('Consistency (perturbation) cut probability.'))

flags.DEFINE_string(
    'box_reg_scale_mode', default='random_size',
    help=('Consistency (perturbation), unsup_reg is cutout/aug_cutout: box '
          'mask scaling (fixed|random_area|random_size).'))

flags.DEFINE_float(
    'box_reg_scale', default=0.25,
    help=('Consistency (perturbation), unsup_reg is cutout/aug_cutout: '
          'fraction of the image to mask out when box scale mode is fixed.'))

flags.DEFINE_bool(
    'box_reg_random_aspect_ratio', default=True,
    help=('Consistency (perturbation), unsup_reg is cutout/aug_cutout: vary '
          'the aspect ratio of the box'))

flags.DEFINE_string(
    'cow_sigma_range', default='4.0:16.0',
    help=('Consistency (perturbation), unsup_reg is cowout/aug_coowout: the '
          'range of the Gaussian smoothing sigma that controls the scale of '
          'CowMask'))

flags.DEFINE_string(
    'cow_prop_range', default='0.25:1.0',
    help=('Consistency (perturbation), unsup_reg is cowout/aug_coowout: the '
          'range of proportion of the image to be masked out by CowMask'))

flags.DEFINE_string(
    'mix_reg', default='cowmix',
    help=('Mix regularizer '
          '(none|ict|cutmix|cowmix).'))

flags.DEFINE_bool(
    'mix_aug_separately', default=False,
    help=('Mix regularization, use different augmentations for teacher '
          '(unmixed) and student (mixed) paths'))

flags.DEFINE_bool(
    'mix_logits', default=False,
    help=('Mix regularization, mix pre-softmax logits rather than '
          'post-softmax probabilities'))

flags.DEFINE_float(
    'mix_weight', default=30.0,
    help=('Mix regularization, mix consistency loss weight.'))

flags.DEFINE_float(
    'mix_conf_thresh', default=0.6,
    help=('Mix regularization, confidence threshold.'))

flags.DEFINE_bool(
    'mix_conf_avg', default=True,
    help=('Mix regularization, average confidence threshold masks'))

flags.DEFINE_string(
    'mix_conf_mode', default='mix_conf',
    help=('Mix either confidence or probabilities for confidence '
          'thresholding (prob|conf).'))

flags.DEFINE_float(
    'ict_alpha', default=0.1,
    help=('Mix regularization, mix_reg=ict: ICT Beta distribution alpha '
          'parameter.'))

flags.DEFINE_string(
    'mix_box_reg_scale_mode', default='random_area',
    help=('Mix regularization, mix_reg=cutmix: box '
          'mask scaling (fixed|random_area|random_size).'))

flags.DEFINE_float(
    'mix_box_reg_scale', default=0.25,
    help=('Mix regularization, mix_reg=cutmixt: '
          'fraction of the image to mask out when box scale mode is fixed.'))

flags.DEFINE_bool(
    'mix_box_reg_random_aspect_ratio', default=True,
    help=('Mix regularization, mix_reg=cutmix: vary '
          'the aspect ratio of the box'))

flags.DEFINE_string(
    'mix_cow_sigma_range', default='4.0:16.0',
    help=('Mix regularization, mix_reg=cowmix: the '
          'range of the Gaussian smoothing sigma that controls the scale of '
          'CowMask'))

flags.DEFINE_string(
    'mix_cow_prop_range', default='0.2:0.8',
    help=('Mix regularization, mix_reg=cowmix: the '
          'range of proportion of the image to be masked out by CowMask'))

flags.DEFINE_integer(
    'subset_seed', default=12345,
    help=('Random seed used to choose supervised samples (n_sup != -1).'))

flags.DEFINE_integer(
    'val_seed', default=131,
    help=('Random seed used to choose validation samples (when n_val > 0).'))

flags.DEFINE_integer(
    'run_seed', default=None,
    help=('Random seed used for network initialisation and training. If '
          'run_seed = None then one will be generated using n_val '
          'and subset_seed.'))

flags.DEFINE_string(
    'checkpoints', default='on',
    help=('Checkpointing after each epoch (none|on|retain); '
          'disabled/enabled/retain'))


def _range_str_to_tuple(s):
  xs = [x.strip() for x in s.split(':')]
  return tuple([float(x) for x in xs])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_semisup.experiment(
      model_dir=FLAGS.model_dir, imagenet_subset_dir=FLAGS.imagenet_subset_dir,
      dataset=FLAGS.dataset, batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.eval_batch_size, num_epochs=FLAGS.num_epochs,
      learning_rate=FLAGS.learning_rate,
      aug_imagenet_apply_colour_jitter=FLAGS.aug_imagenet_apply_colour_jitter,
      aug_imagenet_greyscale_prob=FLAGS.aug_imagenet_greyscale_prob,
      sgd_momentum=FLAGS.sgd_momentum, sgd_nesterov=FLAGS.sgd_nesterov,
      lr_schedule=FLAGS.lr_schedule,
      lr_sched_steps=ast.literal_eval(FLAGS.lr_sched_steps),
      lr_sched_halfcoslength=FLAGS.lr_sched_halfcoslength,
      lr_sched_warmup=FLAGS.lr_sched_warmup, l2_reg=FLAGS.l2_reg,
      weight_decay=FLAGS.weight_decay,
      architecture=FLAGS.architecture, n_val=FLAGS.n_val, n_sup=FLAGS.n_sup,
      teacher_alpha=FLAGS.teacher_alpha,
      anneal_teacher_alpha=FLAGS.anneal_teacher_alpha,
      unsupervised_regularizer=FLAGS.unsup_reg,
      cons_weight=FLAGS.cons_weight, conf_thresh=FLAGS.conf_thresh,
      conf_avg=FLAGS.conf_avg,
      cut_backg_noise=FLAGS.cut_backg_noise, cut_prob=FLAGS.cut_prob,
      box_reg_scale_mode=FLAGS.box_reg_scale_mode,
      box_reg_scale=FLAGS.box_reg_scale,
      box_reg_random_aspect_ratio=FLAGS.box_reg_random_aspect_ratio,
      cow_sigma_range=_range_str_to_tuple(FLAGS.cow_sigma_range),
      cow_prop_range=_range_str_to_tuple(FLAGS.cow_prop_range),
      mix_regularizer=FLAGS.mix_reg,
      mix_aug_separately=FLAGS.mix_aug_separately, mix_logits=FLAGS.mix_logits,
      mix_weight=FLAGS.mix_weight, mix_conf_thresh=FLAGS.mix_conf_thresh,
      mix_conf_avg=FLAGS.mix_conf_avg,
      mix_conf_mode=FLAGS.mix_conf_mode,
      ict_alpha=FLAGS.ict_alpha,
      mix_box_reg_scale_mode=FLAGS.mix_box_reg_scale_mode,
      mix_box_reg_scale=FLAGS.mix_box_reg_scale,
      mix_box_reg_random_aspect_ratio=FLAGS.mix_box_reg_random_aspect_ratio,
      mix_cow_sigma_range=_range_str_to_tuple(FLAGS.mix_cow_sigma_range),
      mix_cow_prop_range=_range_str_to_tuple(FLAGS.mix_cow_prop_range),
      subset_seed=FLAGS.subset_seed, val_seed=FLAGS.val_seed,
      run_seed=FLAGS.run_seed,
      log_fn=logging.info, checkpoints=FLAGS.checkpoints)


if __name__ == '__main__':
  app.run(main)

# coding=utf-8
"""Define FLAGS of the model."""

from __future__ import absolute_import
from __future__ import division

from absl import flags

EVAL_FREQ = 500


def define_basic_flags():
  """Defines basic flags."""

  flags.DEFINE_integer('max_iteration', 200000, 'Number of iteration')
  flags.DEFINE_integer(
      'max_epoch', None, 'Number of epochs. If max_epoch is set, '
      'max_iteration will be automatically re-adjusted based on data size.')
  flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
  flags.DEFINE_integer('batch_size', 100, 'Batch size')
  flags.DEFINE_integer('val_batch_size', 100, 'Validation data batch size.')
  flags.DEFINE_integer('restore_step', 0, ('Checkpoint id.'
                                           '0: load the latest step ckpt.'
                                           '>0: load the assigned step ckpt.'))
  flags.DEFINE_enum(
      'network_name', 'wrn28-10',
      ['resnet29', 'wrn28-10', 'resnet50', 'resnet32', 'resnet18'],
      'Network architecture name')
  flags.DEFINE_string('dataset', 'cifar100_uniform_0.8',
                      'Dataset schema: <dataset>_<noise-type>_<ratio>')
  flags.DEFINE_integer('seed', 12345, 'Seed for selecting validation set')
  flags.DEFINE_enum('method', 'ieg', ['ieg', 'l2r', 'supervised', 'fsr'],
                    'Method to deploy.')
  flags.DEFINE_float('momentum', 0.9,
                     'Use momentum optimizer and the same for meta update')
  flags.DEFINE_string('decay_steps', '500',
                      'Decay steps, format (integer[,<integer>,<integer>]')
  flags.DEFINE_string(
      'decay_epochs', '1',
      'Decay epoch with format integer[,<integer>,<integer>]. '
      'Used only as max_epoch is set.')
  flags.DEFINE_float('decay_rate', 0.1, 'Decay steps')
  flags.DEFINE_integer('eval_freq', EVAL_FREQ,
                       'How many steps evaluate and save model')
  flags.DEFINE_string('checkpoint_path', '/tmp/ieg',
                      'Checkpoint saving root folder')
  flags.DEFINE_integer('warmup_epochs', 0, 'Warmup with standard training')
  flags.DEFINE_enum(
      'lr_schedule', 'cosine',
      ['cosine', 'custom_step', 'cosine_warmup', 'exponential', 'cosine_one'],
      'Learning rate schedule.')
  flags.DEFINE_float('cos_t_mul', 1.5, 't_mul of cosine learning rate')
  flags.DEFINE_float('cos_m_mul', 0.9, 'm_mul of cosine learning rate')
  flags.DEFINE_bool('use_ema', True, 'Use EMA')

  # Method related arguments
  flags.DEFINE_float('meta_momentum', 0.9, 'Meta momentum.')
  flags.DEFINE_float('meta_stepsize', 0.1, 'Meta learning step size.')
  flags.DEFINE_float('ce_factor', 5,
                     'Weight of cross_entropy loss (p, see paper).')
  flags.DEFINE_float('consistency_factor', 20,
                     'Weight of KL loss (k, see paper)')
  flags.DEFINE_float(
      'probe_dataset_hold_ratio', 0.02,
      'Probe set holdout ratio from the training set (0.02 indicates 1000 '
      'images for CIFAR datasets).')
  flags.DEFINE_float('grad_eps_init', 0.9, 'eps for meta learning init value')
  flags.DEFINE_enum(
      'aug_type', 'autoaug', ['autoaug', 'randaug', 'default'],
      'Fake autoaugmentation type. See dataset_utils/ for more details')
  flags.DEFINE_bool('post_batch_mode_autoaug', True,
                    'If true, apply batch augmentation.')
  flags.DEFINE_enum('mode', 'train', ['train', 'evaluation'],
                    'Train or evaluation mode.')
  flags.DEFINE_bool(
      'use_imagenet_as_eval', False,
      'Use imagenet as eval when training on webvision while use '
      'webvision eval when False')
  flags.DEFINE_bool('ds_include_metadata', False, 'Returns image ids')
  flags.DEFINE_string('dataset_dir', './ieg/data', 'Dataset dictionary.')

  flags.DEFINE_integer('xm_exp_id', None,
                       'Experiment id for specifying folders for xcloud.')
  flags.DEFINE_string(
      'exp_path_pattern', None,
      'Experiment subfolder pattern splitted by +, e.g. dataset+network_name')
  flags.DEFINE_string('distribution_strategy', 'mirrored',
                      'Distribution strategy.')
  flags.DEFINE_string('tpu', None,
                      'TPU address if distribution_strategy is tpu.')
  flags.DEFINE_string('pretrained_ckpt', None, 'Pretrained checkpoint path.')
  flags.DEFINE_float('label_smoothing', 0.0, 'Label smoothing.')
  flags.DEFINE_float(
      'dropout_rate', 0.0,
      'Dropout rate of the classifier layer, only effective for certain models.'
  )
  flags.DEFINE_bool('summary_eval_to_train', True,
                    'Export evaluation metrics to train events.')
  flags.DEFINE_float('l2_weight_decay', 0.0004, 'L2 weight decay rate.')
  flags.DEFINE_bool(
      'fix_network_features', False,
      'Fix backbone features, only appliable to certain architectures.')
  flags.DEFINE_string(
      'pretrained_ckpt_mode', 'imagenet',
      'Pretrain checkpoint mode. imagenet means loading imagenet weights. '
      'It is applicable for certain architectures.')

  # Used for the FSR method. Default parameters for noise labels.
  flags.DEFINE_float('queue_beta', 0.9,
                     'Beta of queue metric global averaging param.')
  flags.DEFINE_bool('use_mixup', True, 'Use mixup for training data.')
  flags.DEFINE_integer('queue_capacity', 2000, 'Capacity of queue.')
  flags.DEFINE_integer('queue_bs', 200, 'Batch size of queue.')
  flags.DEFINE_string('queue_metric', 'margin', 'Queue data selection metric.')
  flags.DEFINE_integer('meta_start_epoch', 20, 'Start epoch of meta step')
  flags.DEFINE_string(
      'use_pseudo_loss', 'all_1_0.1_2',
      'Type of using pseudo labeling, format: type_temp_gamma_weight, '
      'where type is loss type, tempature is softmax loss tempature and '
      'gamma is moving averaging weight of pseudo logits, and weight is '
      'loss weight.')
  flags.DEFINE_float(
      'meta_partial_level', 0,
      'Meta variable level to compute second-order gradients.'
      'O means only including the dense layer. -1 means all layers.')
  flags.DEFINE_bool(
      'clip_meta_weight', True,
      'Clip meta weights smaller than zero if True, often used for noisy labels. '
      'Shift and normalize otherwise.')
  flags.DEFINE_float(
      'moving_weight_gamma', 0,
      'Using moving average of meta weight, instead of online version.')
  flags.DEFINE_integer('meta_moving_end_epoch', 0,
                       'End epoch of meta moving weight averaging.')
  flags.DEFINE_bool('verbose_finer_log', True,
                    'Whether show finer loggings in tensorboard.')


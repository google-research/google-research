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
"""Base model trainer."""

import json
import os
import random
import shutil
import time

from absl import logging
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.svm import OneClassSVM
import tensorflow as tf
from tqdm import trange

from deep_representation_one_class.data.celeba import CelebA
from deep_representation_one_class.data.cifar import CIFAROOD
from deep_representation_one_class.data.dogvscat import DogVsCatOOD
from deep_representation_one_class.data.fmnist import FashionMNISTOOD
from deep_representation_one_class.model import resnet as model
import deep_representation_one_class.util.metric as util_metric
from deep_representation_one_class.util.scheduler import CustomLearningRateSchedule as CustomSchedule

_SUPPORTED_DATASET = frozenset([
    'cifar10ood', 'cifar20ood', 'cifar100ood', 'fashion_mnistood', 'fmnistood',
    'dogvscatood', 'dvcood', 'celeba'
])


def setup_tf():
  logging.set_verbosity(logging.ERROR)
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  if not physical_devices:
    logging.info('No GPUs are detected')
  for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)
  return tf.distribute.MirroredStrategy()


class BaseTrain(object):
  """Base model trainer.

  Model constructor:
    Parameters
    Data loader
    Model architecture
    Optimizer
  Model trainer:
    Custom train loop
    Evaluation loop
  """

  def __init__(self, hparams):
    self.strategy = setup_tf()
    self.hparams = hparams
    # data
    self.is_validation = hparams.is_validation
    self.root = hparams.root
    self.dataset = hparams.dataset
    self.category = hparams.category
    self.aug_list = hparams.aug_list.split(',')
    self.aug_list_for_test = hparams.aug_list_for_test.split(
        ',') if hparams.aug_list_for_test is not None else None
    self.input_shape = tuple(
        [int(float(s)) for s in hparams.input_shape.split(',')])
    try:
      self.distaug_type = int(hparams.distaug_type)
    except ValueError:
      self.distaug_type = hparams.distaug_type
    # network architecture
    self.net_type = hparams.net_type
    self.net_width = hparams.net_width
    self.head_dims = tuple([int(d) for d in hparams.head_dims.split(',') if d
                           ]) if hparams.head_dims not in [None, ''] else None
    self.latent_dim = hparams.latent_dim

    # optimizer
    self.seed = hparams.seed
    self.force_init = hparams.force_init
    self.optim_type = hparams.optim_type
    self.sched_type = hparams.sched_type
    self.sched_freq = hparams.sched_freq
    self.sched_step_size = hparams.sched_step_size
    self.sched_gamma = hparams.sched_gamma
    self.sched_min_rate = hparams.sched_min_rate
    self.sched_level = hparams.sched_level
    self.learning_rate = hparams.learning_rate
    self.weight_decay = hparams.weight_decay
    self.regularize_bn = hparams.regularize_bn
    self.weight_decay_constraint = []
    if self.regularize_bn:
      self.weight_decay_constraint.append('bn')
    self.momentum = hparams.momentum
    self.nesterov = hparams.nesterov
    self.num_epoch = hparams.num_epoch
    self.num_batch = hparams.num_batch
    self.batch_size = hparams.batch_size
    # monitoring and checkpoint
    self.ckpt_prefix = os.path.join(hparams.model_dir, hparams.ckpt_prefix)
    self.ckpt_epoch = hparams.ckpt_epoch
    self.file_path = hparams.file_path
    # additional hparams
    self.set_hparams(hparams=hparams)
    self.set_metrics()

  def set_random_seed(self):
    seed = self.seed
    if seed > 0:
      random.seed(seed)
      np.random.seed(seed)
      tf.random.set_seed(seed)

  def set_hparams(self, hparams):
    pass

  def config(self):
    """Config."""
    self.set_random_seed()
    # Data loader.
    self.get_dataloader()
    # Model architecture.
    self.model = self.get_model(
        arch=self.net_type,
        width=self.net_width,
        head_dims=self.head_dims,
        input_shape=self.input_shape,
        num_class=self.latent_dim)
    # Scheduler.
    self.scheduler, self.sched_name = self.get_scheduler(
        sched_type=self.sched_type,
        step_per_epoch=1 if self.sched_freq == 'step' else self.num_batch,
        max_step=self.num_epoch * self.num_batch,
        learning_rate=self.learning_rate,
        **{
            'step_size': self.sched_step_size,
            'gamma': self.sched_gamma,
            'min_rate': self.sched_min_rate,
            'level': self.sched_level
        })
    # Optimizer.
    self.optimizer, self.optim_name = self.get_optimizer(
        scheduler=self.scheduler,
        optim_type=self.optim_type,
        learning_rate=self.learning_rate,
        **{
            'momentum': self.momentum,
            'nesterov': self.nesterov
        })
    # Set file path.
    self.get_file_path()

  def get_dataloader(self):
    """Gets the data loader."""
    dl = self.get_dataset(self.root, self.dataset.lower(), self.category,
                          self.input_shape)

    datasets = dl.load_dataset(
        is_validation=self.is_validation,
        aug_list=self.aug_list,
        aug_list_for_test=self.aug_list_for_test,
        batch_size=self.batch_size,
        num_batch_per_epoch=self.num_batch,
        distaug_type=self.distaug_type)

    # train_loader: train data for representation learning (augmentation)
    # cls_loader: train data for classifier learning (no augmentation)
    # test_loader: test data
    self.train_loader = datasets[0]
    if isinstance(self.train_loader, (list, tuple)):
      self.num_batch = self.train_loader[1]
      self.train_loader = self.train_loader[0]
    self.cls_loader = datasets[1]
    self.test_loader = datasets[2]
    self.db_name = dl.fname

    if self.strategy:
      self.train_loader = self.strategy.experimental_distribute_dataset(
          self.train_loader)
      self.cls_loader[0] = self.strategy.experimental_distribute_dataset(
          self.cls_loader[0])
      self.test_loader[0] = self.strategy.experimental_distribute_dataset(
          self.test_loader[0])

  @staticmethod
  def get_dataset(root, dataset, category, input_shape):
    """Gets the dataset."""
    if dataset not in _SUPPORTED_DATASET:
      msg = (f'Unsupported dataset {dataset} is provided. Only '
             f'{_SUPPORTED_DATASET} are available.')
      raise ValueError(msg)

    if dataset in ['cifar10ood', 'cifar20ood', 'cifar100ood']:
      dl = CIFAROOD(
          root=root,
          dataset=dataset,
          category=category,
          input_shape=input_shape or (32, 32, 3))
    elif dataset in ['fashion_mnistood', 'fmnistood']:
      dl = FashionMNISTOOD(
          root=root,
          dataset=dataset,
          category=category,
          input_shape=input_shape or (32, 32, 3))
    elif dataset in ['dogvscatood', 'dvcood']:
      dl = DogVsCatOOD(
          root=root,
          dataset=dataset,
          category=category,
          input_shape=input_shape or (64, 64, 3))
    elif dataset == 'celeba':
      dl = CelebA(
          root=root,
          dataset=dataset,
          category=category,
          input_shape=input_shape or (64, 64, 3))
    return dl

  @staticmethod
  def get_model(arch='ResNet18',
                width=1.0,
                head_dims=None,
                input_shape=(256, 256, 3),
                num_class=2):
    """Gets the ResNet model."""
    net = model.__dict__[arch](
        width=width,
        head_dims=head_dims,
        input_shape=input_shape,
        num_class=num_class)
    net.summary()
    return net

  @staticmethod
  def get_optimizer(scheduler, optim_type='sgd', learning_rate=0.03, **kwargs):
    """Gets the optimizer."""
    if optim_type == 'sgd':
      momentum = kwargs['momentum'] if 'momentum' in kwargs else 0.9
      nesterov = kwargs['nesterov'] if 'nesterov' in kwargs else False
      optimizer = tf.keras.optimizers.SGD(
          learning_rate=scheduler, momentum=momentum, nesterov=nesterov)
      name = 'sgd_lr{:g}_mom{:g}'.format(learning_rate, momentum)
      if nesterov:
        name += '_nesterov'
    elif optim_type == 'adam':
      optimizer = tf.keras.optimizers.Adam(
          learning_rate=scheduler, amsgrad=True)
      name = 'adam_lr{:g}'.format(learning_rate)
    else:
      raise NotImplementedError
    return optimizer, name

  @staticmethod
  def get_scheduler(sched_type='cosine',
                    step_per_epoch=1,
                    max_step=256,
                    learning_rate=0.1,
                    **kwargs):
    """Gets the scheduler."""
    scheduler = CustomSchedule(
        step_per_epoch=step_per_epoch,
        base_lr=learning_rate,
        max_step=max_step,
        mode=sched_type,
        **kwargs)
    return scheduler, scheduler.name

  def get_file_path(self):
    """Gets the file path for saving."""
    if self.file_path:
      self.file_path = os.path.join(self.ckpt_prefix, self.file_path)
    else:
      self.file_path = os.path.join(
          self.ckpt_prefix, '{}_seed{}'.format(self.db_name, self.seed),
          self.model.name, '{}_{}_{}_wd{:g}_{}_epoch{}_nb{}_bs{}'.format(
              self.__class__.__name__, self.optim_name, self.sched_name,
              self.weight_decay, '_'.join(self.weight_decay_constraint),
              self.num_epoch, self.num_batch, self.batch_size))
      if self.file_suffix:
        self.file_path = '{}_{}'.format(self.file_path, self.file_suffix)
    self.file_path = self.file_path.replace('__', '_')
    self.json_path = os.path.join(self.file_path, 'stats')

  def get_current_train_epoch(self):
    """Returns current training epoch."""
    return tf.math.floordiv(self.optimizer.iterations, self.num_batch).numpy()

  def get_current_train_step(self):
    """Returns current training step."""
    return self.optimizer.iterations

  def get_checkpoint(self):
    """Restores from the checkpoint and returns start epoch."""
    self.checkpoint.restore(self.manager.latest_checkpoint)
    self.epoch = start_epoch = self.get_current_train_epoch()
    self.step = self.get_current_train_step()
    return start_epoch

  def train(self):
    """Called for model training."""
    start_epoch = self.train_begin()
    if self.num_epoch == 0:
      self.train_epoch_begin()
    else:
      for _ in range(start_epoch, self.num_epoch):
        self.train_epoch_begin()
        self.train_epoch()
        self.train_epoch_end(
            is_eval=False, is_save=(self.epoch % self.ckpt_epoch == 0))
    self.train_epoch_end(is_eval=True, is_save=True)
    self.train_end()

  def train_begin(self):
    """Initializes metrics, checkpoint, summary at the beginning of training."""
    self.metrics = {}
    self.metrics.update({
        key: tf.keras.metrics.Mean()
        for key in self.list_of_metrics
        if key.startswith(('loss'))
    })
    self.metrics.update({
        key: tf.keras.metrics.Accuracy()
        for key in self.list_of_metrics
        if key.startswith('acc')
    })
    self.monitor = {
        'learning_rate': 0,
        'step_per_second': 0,
    }
    self.eval_metrics = {}
    self.eval_metrics.update({key: None for key in self.list_of_eval_metrics})
    if self.force_init:
      shutil.rmtree(self.file_path, ignore_errors=True)
    # Generate file paths
    if not tf.io.gfile.isdir(self.file_path):
      tf.io.gfile.makedirs(self.file_path)
    if not tf.io.gfile.isdir(self.json_path):
      tf.io.gfile.makedirs(self.json_path)
    # Checkpoint
    self.checkpoint = tf.train.Checkpoint(
        optimizer=self.optimizer, model=self.model)
    self.manager = tf.train.CheckpointManager(
        checkpoint=self.checkpoint,
        directory=os.path.join(self.file_path, 'raw'),
        max_to_keep=1)
    self.summary_writer = tf.summary.create_file_writer(
        logdir=os.path.join(self.file_path, 'tb'))
    # Initiate train iterator once
    # Note that creating iterator every epoch slows down
    # the training since it clears the data buffer
    self.train_iterator = iter(self.train_loader)
    self.cls_iterator = (iter(self.cls_loader[0]), self.cls_loader[1])
    self.test_iterator = (iter(self.test_loader[0]), self.test_loader[1])
    return self.get_checkpoint()

  def train_end(self, verbose=False):
    """Saves and prints summary statistics."""
    self.manager.save()
    self.summary_writer.close()

    if verbose:
      # pylint: disable=protected-access
      logdir = self.summary_writer._init_op_fn.keywords['logdir'].numpy(
      ).decode()
      event_files = [
          event for event in tf.io.gfile.glob(os.path.join(logdir, '*'))
      ]
      event_files.sort(key=os.path.getmtime)
      event_dict = {
          key: []
          for key in self.metrics.keys()
          if not key.startswith('monitor')
      }
      event_dict.update({key: [] for key in self.eval_metrics.keys()})
      for event_file in event_files:
        for event in tf.compat.v1.train.summary_iterator(event_file):
          for v in event.summary.value:
            if v.tag.replace('/', '.') in event_dict:
              event_dict[v.tag.replace('/', '.')].append(
                  tf.make_ndarray(v.tensor).tolist())
      # Print stats of last 20 epochs in json format
      num_epoch_to_save = 20
      event_dict = {
          key: event_dict[key][-num_epoch_to_save:] for key in event_dict
      }
      if not os.path.isdir(self.json_path):
        os.makedirs(self.json_path)
      summary_dict = {}
      for key in event_dict:
        dict_to_write = {
            'median (last%02d)' % x: np.median(event_dict[key][-x:])
            for x in [1, 5, 10, num_epoch_to_save]
        }
        dict_to_write.update(
            {'last%02d' % (num_epoch_to_save): event_dict[key]})
        with open(os.path.join(self.json_path, key + '.json'), 'w') as outfile:
          json.dump(dict_to_write, outfile, sort_keys=True, indent=4)
        if key in self.metric_of_interest:
          summary_dict.update({key: dict_to_write})
          with open(os.path.join(self.json_path, 'summary.json'),
                    'w') as outfile:
            json.dump(summary_dict, outfile, sort_keys=True, indent=4)
      # Print basic information
      logging.info('')
      logging.info('----------------------------------------------------------')
      logging.info('Train is done. Below are file path and basic test stats\n')
      logging.info('File path:\n')
      logging.info(self.file_path)
      if not isinstance(self.metric_of_interest, (list, tuple)):
        self.metric_of_interest = [self.metric_of_interest]
      for moi in self.metric_of_interest:
        del summary_dict[moi]['last%02d' % (num_epoch_to_save)]
      logging.info('Eval stats:\n')
      logging.info(json.dumps(summary_dict, sort_keys=True, indent=4))
      logging.info('----------------------------------------------------------')
      logging.info()
    else:
      with tf.io.gfile.GFile(os.path.join(self.json_path, 'summary.json'),
                             'w') as outfile:
        json.dump(self.eval_metrics, outfile, sort_keys=True, indent=4)
    with tf.io.gfile.GFile(os.path.join(self.json_path, 'hparams.json'),
                           'w') as outfile:
      json.dump(self.hparams, outfile, indent=4, sort_keys=True)

  def train_epoch(self):
    """Called for model training per epoch."""
    time_init = time.time()
    for _ in trange(
        self.num_batch,
        leave=False,
        desc='Epoch (train) %d/%d' % (self.epoch + 1, self.num_epoch)):
      self.train_step(self.train_iterator)
    self.monitor['step_per_second'] = self.num_batch / (time.time() - time_init)

  def train_epoch_begin(self):
    """Called at the beginning of epoch.

    - Reset metrics
    - Adjust learning rate
    """
    for _, metric in self.metrics.items():
      metric.reset_states()
    self.epoch = self.get_current_train_epoch()
    self.step = self.get_current_train_step()
    self.monitor['learning_rate'] = self.optimizer.learning_rate(
        self.optimizer.iterations).numpy()

  def train_epoch_end(self, is_eval=False, is_save=False):
    """Evaluates and monitors performance at the end of epoch."""
    if is_save:
      self.manager.save()
    if is_eval:
      self.eval_epoch(trainset=self.cls_iterator, testset=self.test_iterator)
    self.monitor_progress(verbose=True)

  @tf.function
  def train_step(self, iterator):
    """Executes each train step."""

    def step_fn(data):
      replica_context = tf.distribute.get_replica_context()
      xo, xc = data[0], data[1]
      x = tf.concat((xo, xc), axis=0)
      y = tf.concat((tf.zeros(
          xo.shape[0], dtype=tf.int32), tf.ones(xc.shape[0], dtype=tf.int32)),
                    axis=0)
      with tf.GradientTape() as tape:
        logits = self.model(x, training=True)['logits']
        loss_xe = tf.keras.losses.sparse_categorical_crossentropy(
            y, logits, from_logits=True)
        loss_xe = tf.divide(
            tf.reduce_sum(loss_xe),
            self.cross_replica_concat(loss_xe,
                                      replica_context=replica_context).shape[0])
        loss_l2 = self.loss_l2(self.model.trainable_weights)
        loss = loss_xe + self.weight_decay * loss_l2
      grad = tape.gradient(loss, self.model.trainable_weights)
      self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
      # monitor
      self.metrics['loss.train'].update_state(loss)
      self.metrics['loss.xe'].update_state(loss_xe)
      self.metrics['loss.L2'].update_state(loss_l2)
      self.metrics['acc.train'].update_state(y, tf.argmax(logits, axis=1))

    # Call one step
    self.strategy.run(step_fn, args=(next(iterator),))

  def loss_l2(self, var_list):
    for c in self.weight_decay_constraint:
      var_list = [v for v in var_list if c not in v.name]
    loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in var_list])
    return tf.divide(loss_l2, self.strategy.num_replicas_in_sync)

  def squared_difference(self, a, b, do_normalization=True):
    """Computes (a-b) ** 2."""
    if do_normalization:
      a = tf.nn.l2_normalize(a, axis=1)
      b = tf.nn.l2_normalize(b, axis=1)
      return -2. * tf.matmul(a, b, transpose_b=True)
    return tf.norm(
        a, axis=1, keepdims=True)**2 + tf.transpose(
            tf.norm(b, axis=1, keepdims=True)**2) - 2. * tf.matmul(
                a, b, transpose_b=True)

  def eval_epoch(self, trainset, testset):
    self.eval_embed(trainset=trainset, testset=testset)

  def eval_embed(self, trainset, testset):
    """Evaluate performance on test set."""
    _, _, embeds_tr, pools_tr, _ = self.extract(trainset)
    probs, dscores, embeds, pools, labels = self.extract(testset)
    sim_embed = -0.5 * self.squared_difference(embeds, embeds_tr, True)
    sim_pool = -0.5 * self.squared_difference(pools, pools_tr, True)
    dist_embed = tf.reduce_mean(1.0 - tf.nn.top_k(sim_embed, k=1)[0], axis=1)
    dist_pool = tf.reduce_mean(1.0 - tf.nn.top_k(sim_pool, k=1)[0], axis=1)
    for key in self.eval_metrics:
      if key.startswith('logit'):
        pred = 1.0 - probs[:, 0]
      elif key.startswith('dscore'):
        pred = 1.0 - dscores
      elif key.startswith('embed'):
        pred = dist_embed
        feats_tr = embeds_tr.numpy()
        feats = embeds.numpy()
        sim = sim_embed
      elif key.startswith('pool'):
        pred = dist_pool
        feats_tr = pools_tr.numpy()
        feats = pools.numpy()
        sim = sim_pool
      if 'auc' in key:
        self.eval_metrics[key] = util_metric.roc(pr=pred, gt=labels)
      elif 'locsvm' in key and key.startswith(('embed', 'pool')):
        # Linear kernel OC-SVM.
        clf = OneClassSVM(kernel='linear').fit(feats_tr)
        scores = -clf.score_samples(feats)
        self.eval_metrics[key] = util_metric.roc(pr=scores, gt=labels)
      elif 'kocsvm' in key and key.startswith(('embed', 'pool')):
        # RBF kernel OC-SVM.
        feats_tr = tf.nn.l2_normalize(feats_tr, axis=1)
        feats = tf.nn.l2_normalize(feats, axis=1)
        # 10 times larger value of gamma.
        gamma = 10. / (tf.math.reduce_variance(feats_tr) * feats_tr.shape[1])
        clf = OneClassSVM(kernel='rbf', gamma=gamma).fit(feats_tr)
        scores = -clf.score_samples(feats)
        self.eval_metrics[key] = util_metric.roc(pr=scores, gt=labels)
      elif 'kde' in key and key.startswith(('embed', 'pool')):
        # RBF kernel density estimation.
        feats_tr = tf.nn.l2_normalize(feats_tr, axis=1)
        gamma = 10. / (tf.math.reduce_variance(feats_tr) * feats_tr.shape[1])
        scores = None
        batch_size_for_kde = 100
        num_iter = int(np.ceil(sim.shape[0] / batch_size_for_kde))
        for i in range(num_iter):
          sim_batch = sim[i * batch_size_for_kde:(i + 1) * batch_size_for_kde]
          scores_batch = -tf.divide(
              tf.reduce_logsumexp(2 * gamma * sim_batch, axis=1), gamma)
          scores = scores_batch if scores is None else tf.concat(
              (scores, scores_batch), axis=0)
        self.eval_metrics[key] = util_metric.roc(pr=scores, gt=labels)
      elif 'gde' in key and key.startswith(('embed', 'pool')):
        # Gaussian density estimation with full covariance.
        feats_tr = tf.nn.l2_normalize(feats_tr, axis=1)
        feats = tf.nn.l2_normalize(feats, axis=1)
        km = GMM(n_components=1, init_params='kmeans', covariance_type='full')
        km.fit(feats_tr)
        scores = -km.score_samples(feats)
        self.eval_metrics[key] = util_metric.roc(pr=scores, gt=labels)

  def extract(self, dataset):
    """Extract logits, embeds, pool, and labels."""
    outputs = {
        'logits': None,
        'dscore': None,
        'embeds': None,
        'pools': None,
        'labels': None
    }
    inference = self.model
    iterator, num_batch = dataset[0], dataset[1]
    if self.aug_list_for_test is not None:
      num_aug = len(self.aug_list_for_test)
    else:
      num_aug = 1
    for _ in trange(
        num_batch,
        leave=False,
        desc='Extract %d/%d' % (self.epoch + 1, self.num_epoch)):
      logits, embeds, pools, y = self.extract_step(iterator, inference)
      if num_aug > 1:
        probs = tf.nn.softmax(logits, axis=1)
        probs = tf.split(probs, num_aug)
        dscore = tf.math.exp(
            tf.reduce_sum(
                tf.math.log(
                    tf.concat([probs[i][:, i:i + 1] for i in range(len(probs))],
                              axis=1)),
                axis=1))
        logits = tf.split(logits, num_aug)[0]
        embeds = tf.split(embeds, num_aug)[0]
        pools = tf.split(pools, num_aug)[0]
      else:
        dscore = tf.nn.softmax(logits, axis=1)[:, 0]
      outputs['logits'] = self.smart_concat(outputs['logits'], logits)
      outputs['dscore'] = self.smart_concat(outputs['dscore'], dscore)
      outputs['embeds'] = self.smart_concat(outputs['embeds'], embeds)
      outputs['pools'] = self.smart_concat(outputs['pools'], pools)
      outputs['labels'] = self.smart_concat(outputs['labels'], y)
    return (tf.nn.softmax(outputs['logits'], axis=1), outputs['dscore'],
            outputs['embeds'], outputs['pools'], tf.squeeze(outputs['labels']))

  @tf.function
  def extract_step(self, iterator, inference):
    """Feature extract step."""

    def step_fn(data):
      """Step."""
      x, y = data[0:-2], data[-2]
      output = inference(tf.concat(x, axis=0), training=False)
      return (output['logits'], output['embeds'], output['pools'], y)

    out = self.strategy.run(step_fn, args=(next(iterator),))
    return [tf.concat(self.strategy.unwrap(o), axis=0) for o in out]

  def monitor_progress(self, verbose=False):
    """Monitor train/eval variables."""
    # Tensorboard
    with self.summary_writer.as_default():
      vis_step = (self.epoch + 1) * self.num_batch
      for key, metric in self.metrics.items():
        tf.summary.scalar(
            key.replace('.', '/', 1), metric.result(), step=vis_step)
      tf.summary.scalar(
          'monitor/step_per_second',
          self.monitor['step_per_second'],
          step=vis_step)
      tf.summary.scalar(
          'monitor/lr', self.monitor['learning_rate'], step=vis_step)
      if verbose:
        for key, metric in self.eval_metrics.items():
          if metric is not None:
            tf.summary.scalar(key.replace('.', '/', 1), metric, step=vis_step)

    # Command line.
    template = ('Epoch {epoch:4d}/{max_epoch:4d}\tstep(sec): '
                '{step_per_second:.3f}\tLoss: {loss:.3f}\tAcc: {acc:.3f}')
    logging.info(
        template.format(
            epoch=self.epoch + 1,
            max_epoch=self.num_epoch,
            step_per_second=self.monitor['step_per_second'],
            loss=self.metrics['loss.train'].result(),
            acc=self.metrics['acc.train'].result()))

  @staticmethod
  def smart_concat(var1, var2):
    """Smart concat."""

    def _smart_concat(var1, var2):
      return var2 if var1 is None else tf.concat((var1, var2), axis=0)

    if isinstance(var2, list):
      if var1 is not None:
        assert isinstance(var1, list)
        return [_smart_concat(v1, v2) for v1, v2 in zip(var1, var2)]
      else:
        return var2
    else:
      if var1 is not None:
        assert not isinstance(var1, list)
      return _smart_concat(var1, var2)

  @staticmethod
  def cross_replica_concat(tensor, replica_context=None):
    """Reduces a concatenation of the `tensor` across TPU cores.

    Args:
      tensor: tensor to concatenate.
      replica_context: A `replica_context`. If not set, CPU execution is
        assumed.

    Returns:
      Tensor of the same rank as `tensor` with first dimension `num_replicas`
      times larger.
    """
    if replica_context is None or replica_context.num_replicas_in_sync <= 1:
      return tensor

    num_replicas = replica_context.num_replicas_in_sync

    with tf.name_scope('cross_replica_concat'):
      # This creates a tensor that is like the input tensor but has an added
      # replica dimension as the outermost dimension. On each replica it will
      # contain the local values and zeros for all other values that need to be
      # fetched from other replicas.
      ext_tensor = tf.scatter_nd(
          indices=[[replica_context.replica_id_in_sync_group]],
          updates=[tensor],
          shape=[num_replicas] + tensor.shape.as_list())

      # As every value is only present on one replica and 0 in all others,
      # adding them all together will result in the full tensor on all replicas.
      ext_tensor = replica_context.all_reduce(tf.distribute.ReduceOp.SUM,
                                              ext_tensor)

      # Flatten the replica dimension.
      # The first dimension size will be: tensor.shape[0] * num_replicas
      # Using [-1] trick to support also scalar input.
      return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])

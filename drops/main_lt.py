# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

r"""Train and evaluate models of long-tail experiments.
"""
import functools
import os
import random
import time
from absl import app
from absl import flags
import cvxpy as cp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from drops import losses_lt
from drops import preact_resnet_models as resnet_models

FLAGS = flags.FLAGS
# Loss: currently support (1) ce; (2) focal;
# (3) bsm: balanced soft-max; (4) logit_adj: logit adjustment;
# (5) ldam; (6) cb: class-balanced loss; cb_focal: cb loss with focal;
# (7) up_ce: ce with class up-sampling;
# (8) posthoc: logit+posthoc;
# (9) posthoc_ce: ce+posthoc;
# (10) drops: posthoc with dro constrained weight;

my_flags = [
    'lr', 'model_dir', 'dataset', 'num_iters', 'loss',
    'num_classes', 'batch_size', 'eval_freq', 'run_id', 'alpha',
    'train_on_full', 'imb_ratio'
]

for attrname in my_flags:
  if attrname in flags.FLAGS:
    print('deleting {}'.format(attrname))
    delattr(flags.FLAGS, attrname)

flags.DEFINE_string('model_dir',
                    './',
                    'directory where model and logs are stored')
flags.DEFINE_float('lr', 0.1, 'initial learning rate')
flags.DEFINE_integer('num_iters', 50000, 'number of training iterations')
flags.DEFINE_integer('eval_freq', 1000, 'eval frequency')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_bool('train_on_full', False,
                  'whether to train on full training set')
flags.DEFINE_integer('run_id', 0, 'run id')
flags.DEFINE_string('dataset', 'cifar10', 'dataset')
flags.DEFINE_integer('num_classes', 10, 'number of classes')
flags.DEFINE_float('imb_ratio', 0.1, 'n_min / n_max')
flags.DEFINE_string('loss', 'drops',
                    'ce/focal/bsm/cb_focal/ldam/logits_adj/posthoc_ce/drops/')
#  bsm shorts hand for balanced soft-max
flags.DEFINE_float('alpha', 1.0, 'alpha-parameter for focal')
flags.DEFINE_float('gamma', 0.5, 'gamma-parameter for focal/ldam(C)')
flags.DEFINE_float('beta', 0.9999, 'beta-parameter for cb')
flags.DEFINE_float('s', 1, 's-parameter for scaling logits of ldam')
flags.DEFINE_string('re_weight_type', 'prior',
                    'type of class-re-weight: none/prior/sqrt')
flags.DEFINE_integer('warmup', '0', 'number of iterations for warmup')
flags.DEFINE_bool('is_upsampling', False, 'whether upsamping or not')

# args for drops
flags.DEFINE_string('dro_div', 'kl',
                    'div_type for dro metric: kl/l2/l1')
flags.DEFINE_float('eps', 0.9, 'perturbation for dro metric')
#  Fix metric base as recip_prior for now
flags.DEFINE_string('metric_base', 'uniform',
                    'class weights for dro metric: prior, recip_prior, uniform')
flags.DEFINE_float('eta_g', 0.01, 'step size for eg updates of eta_g')
flags.DEFINE_float('eta_lambda', 0.01, 'step size for eg updates of eta_lambda')
flags.DEFINE_float('n_it_update', 1, 'frequency of g, lambda updates')
flags.DEFINE_float('tau', 1.0, 'constant for logit adjust term')
flags.DEFINE_string('weight_type', 'ce',
                    'type of class-re-weight: 0_1/ce')
flags.DEFINE_string('g_type', 'not-eg',
                    'simplied version or EG style, is eg or not')


IMAGE_SIZE = 32
EPOCH_SIZE = 50000
_COARSE_CLASSES = [[4, 72, 55, 30, 95], [32, 1, 67, 73,
                                         91], [70, 82, 54, 92, 62],
                   [9, 10, 16, 28, 61], [0, 83, 51, 53,
                                         57], [39, 40, 86, 22, 87],
                   [5, 20, 84, 25, 94], [6, 7, 14, 18, 24], [97, 3, 42, 43, 88],
                   [68, 37, 12, 76, 17], [33, 71, 49, 23, 60],
                   [38, 15, 19, 21, 31], [64, 66, 34, 75, 63],
                   [99, 77, 45, 79, 26], [2, 35, 98, 11, 46],
                   [44, 78, 93, 27, 29], [65, 36, 74, 80, 50],
                   [96, 47, 52, 56, 59], [90, 8, 13, 48, 58],
                   [69, 41, 81, 85, 89]]


def preprocess_fn(*features, mean, std, augment=False):
  """Preprocess datasets."""
  features = features[0]
  image = features['image']
  label = tf.cast(features['label'], tf.int32)
  image = tf.cast(image, tf.float32) / 255.0
  if augment:
    image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE + 4,
                                             IMAGE_SIZE + 4)
    image = tf.image.random_crop(image,
                                 [image.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.per_image_standardization(image)
    image = (image - mean) / std
  else:
    image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    # image = tf.image.per_image_standardization(image)
    image = (image - mean) / std
  return dict(image=image, label=label)


def ce_loss(labels, preds, from_logits=True):
  labels_oh = maybe_one_hot(labels, depth=FLAGS.num_classes)
  cce = tf.keras.backend.categorical_crossentropy(
      labels_oh, preds, from_logits=from_logits)
  return tf.reduce_mean(cce)


def get_cls_num(dataset_name, y_train, imb_factor=None):
  """Get a list of image numbers for each class.

  Given cifar version Num of imgs follows emponential distribution
  img max: 5000 / 500 * e^(-imb * 0);
  img min: 5000 / 500 * e^(-imb * int(dataset_name - 1))
  exp(-imb * (int(dataset_name) - 1)) = img_max / img_min

  Args:
    dataset_name: str, 'cifar10', 'cifar100
    y_train: the training label
    imb_factor: float, imbalance factor: img_min/img_max, None if geting
      default cifar data number

  Returns:
    img_num_per_cls: a list of number of images per class
  """
  cls_num = 10 if dataset_name == 'cifar10' else 100
  img_max = len(list(np.where(np.array(y_train) == 0)[0]))
  if imb_factor is None:
    return [img_max] * cls_num
  img_num_per_cls = []
  for cls_idx in range(cls_num):
    idx_this_class = list(np.where(np.array(y_train) == cls_idx)[0])
    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
    img_num_per_cls.append(min(int(num), len(idx_this_class)))
  return img_num_per_cls


def get_cls_idx(num_per_cls, y_train, is_upsampling=False):
  """Get the seleted index for the training images.

  Given number of selected images per class, return the image indexes adopted.

  Args:
    num_per_cls: a list of number of images per class
    y_train: the clean label for class selection
    is_upsampling: whether up sampling to strick a balance

  Returns:
    selected_idx: a list of selected images for training use
  """
  cls_num = len(num_per_cls)
  selected_idx = []
  for cls_idx in range(cls_num):
    idx_this_class = list(np.where(np.array(y_train) == cls_idx)[0])
    indices = random.sample(idx_this_class, num_per_cls[cls_idx])
    if is_upsampling:
      up_samling = random.choices(indices, k=int(50000/cls_num)-len(indices))
      selected_idx.extend(up_samling)
    selected_idx.extend(indices)
  return selected_idx


def get_dataset(batch_size,
                data='cifar10',
                imbalance_factor=0.1,
                train_on_full=False):
  """Generate train/valid/eval dataloader in the longtail setting."""
  if data == 'cifar10':
    mean = tf.constant(
        np.reshape([0.4914, 0.4822, 0.4465], [1, 1, 1, 3]), dtype=tf.float32)
    std = tf.constant(
        np.reshape([0.2023, 0.1994, 0.2010], [1, 1, 1, 3]), dtype=tf.float32)
  elif data == 'cifar100':
    mean = tf.constant(
        np.reshape([0.5071, 0.4865, 0.4409], [1, 1, 1, 3]), dtype=tf.float32)
    std = tf.constant(
        np.reshape([0.2673, 0.2564, 0.2762], [1, 1, 1, 3]), dtype=tf.float32)
  preproc_fn_train = functools.partial(
      preprocess_fn, mean=mean, std=std, augment=True)
  if train_on_full:
    ds = tfds.load(data, split='train', as_supervised=True, batch_size=-1)
  else:
    ds = tfds.load(data, split='train[:90%]', as_supervised=True, batch_size=-1)
  x_train, y_clean = tfds.as_numpy(ds)
  img_num_per_cls = get_cls_num(dataset_name=data, y_train=y_clean,
                                imb_factor=imbalance_factor)
  selected_idx = get_cls_idx(num_per_cls=img_num_per_cls, y_train=y_clean,
                             is_upsampling=FLAGS.is_upsampling)
  random.shuffle(selected_idx)
  x_train_imb = x_train[selected_idx]
  y_clean_imb = y_clean[selected_idx]
  ds_imb = tf.data.Dataset.from_tensor_slices({'image': x_train_imb,
                                               'label': y_clean_imb})
  ds_imb = ds_imb.repeat().shuffle(
      batch_size * 4, seed=1).batch(
          batch_size,
          drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  ds_imb = ds_imb.map(preproc_fn_train)
  ds_valid = tfds.load(data, split='train[90%:]', with_info=False)
  ds_valid = ds_valid.shuffle(
      10000, seed=1).batch(
          batch_size,
          drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
  ds_valid = ds_valid.map(
      functools.partial(
          preprocess_fn, mean=mean, std=std))

  ds_tst = tfds.load(data, split='test', with_info=False)
  ds_tst = ds_tst.shuffle(
      10000, seed=1).batch(
          batch_size,
          drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
  ds_tst = ds_tst.map(functools.partial(preprocess_fn, mean=mean, std=std))
  return ds_imb, ds_valid, ds_tst, img_num_per_cls


def maybe_one_hot(labels, depth):
  if len(labels.shape) > 1:
    return labels
  else:
    return tf.one_hot(labels, depth=depth)


def main(_):
  np.random.RandomState(seed=1335)
  FLAGS.model_dir += f'{FLAGS.dataset}_{FLAGS.imb_ratio}/'
  FLAGS.model_dir += f'_loss_{FLAGS.loss}_{FLAGS.run_id}'
  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)
  save_dir = os.path.join(FLAGS.model_dir, 'model')
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  # log file
  fp_log_res = open(os.path.join(FLAGS.model_dir, 'results_log.txt'), 'w')

  # get dataset
  train_ds, valid_ds, eval_ds, samples_per_cls = get_dataset(
      FLAGS.batch_size,
      data=FLAGS.dataset,
      imbalance_factor=FLAGS.imb_ratio,
      train_on_full=FLAGS.train_on_full)

  # build model
  model = resnet_models.create_resnet18(
      input_shape=(32, 32, 3), num_classes=FLAGS.num_classes, norm='batch')

  # revise class-level weights
  if FLAGS.re_weight_type == 'none' or FLAGS.is_upsampling:
    if FLAGS.dataset == 'cifar10':
      samples_per_cls = [5000] * FLAGS.num_classes
    elif FLAGS.dataset == 'cifar100':
      samples_per_cls = [500] * FLAGS.num_classes
  if FLAGS.re_weight_type == 'sqrt':
    samples_per_cls = tf.sqrt(tf.dtypes.cast(samples_per_cls, dtype=tf.float64))
    samples_per_cls = samples_per_cls.numpy().tolist()
  # Initialize g_y to be the same as uniform | prior p(y)| 1/p(y)
  # Default is set to be 'uniform'
  if FLAGS.metric_base == 'uniform':
    g_y = [1] * FLAGS.num_classes
  elif FLAGS.metric_base == 'prior' and FLAGS.re_weight_type == 'prior':
    g_y = [1/i for i in samples_per_cls]
  elif FLAGS.metric_base == 'recip_prior' and FLAGS.re_weight_type == 'prior':
    g_y = samples_per_cls
  g_y = g_y / np.sum(g_y)
  alpha_y = [1/i for i in samples_per_cls]
  alpha_y = tf.cast(alpha_y, dtype=tf.float32)
  alpha_y *= tf.reduce_sum(tf.cast(samples_per_cls, dtype=tf.float32))
  lambd = 1.0
  # set the r_list to be the u in the constraint D(u, g) < delta
  r_list = g_y
  # loss (initialized loss with)
  loss_op = losses_lt.MakeLossFunc(FLAGS.loss, samples_per_cls, FLAGS.gamma,
                                   FLAGS.beta, FLAGS.s, FLAGS.tau)
  boundaries = [(30 * EPOCH_SIZE) // FLAGS.batch_size,
                (80 * EPOCH_SIZE) // FLAGS.batch_size,
                (110 * EPOCH_SIZE) // FLAGS.batch_size]
  values = [0.1, 0.01, 0.001, 0.0001]
  lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries, values)
  optimizer = tf.keras.optimizers.SGD(
      learning_rate=lr_schedule, momentum=0.9, nesterov=True)
  # summary writers
  train_summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.model_dir, 'summaries/train'))
  eval_summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.model_dir, 'summaries/eval'))

  def train_step(images, labels, it):
    """Performs single training step."""
    with tf.GradientTape() as tape:
      logits = model([images, True])
      # Add the warm-up procedure
      if it < FLAGS.warmup:
        loss = ce_loss(labels, logits)
      else:
        loss = loss_op(labels, logits)
      if FLAGS.loss in ['posthoc', 'drops', 'posthoc_ce', 'logit_adj']:
        loss = loss + tf.reduce_sum(model.losses)
      loss_mean = loss.numpy()
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, loss_mean

  def save_logits(ds, model_name, file_name):
    """save model prediction logits and labels."""
    full_logits = []
    full_labels = []
    iterator = iter(ds)
    for batch in iterator:
      images, labels = batch['image'], batch['label']
      logits = model_name([images, False])
      full_logits.append(logits)
      full_labels.append(labels)

    full_logits = tf.concat([
        tf.concat(logits, axis=0) for logits in full_logits], axis=0)
    full_labels = tf.cast(
        tf.concat([
            tf.concat(labels, axis=0)
            for labels in full_labels], axis=0), tf.int64)
    new_name1 = file_name + 'logits.txt'
    if os.path.exists(new_name1):
      os.remove(new_name1)
    with open(new_name1, 'w') as logits_file:
      np.savetxt(logits_file, np.array(full_logits.numpy()))
    new_name2 = file_name + 'labels.txt'
    if os.path.exists(new_name2):
      os.remove(new_name2)
    with open(new_name2, 'w') as labels_file:
      np.savetxt(labels_file, np.array(full_labels.numpy()))
    return full_logits

  def eval_metrics(ds, g_y, sample_per_cls):
    """Evaluate accuracy on test set."""
    iterator = iter(ds)
    avg_loss = 0.
    num_samples = 0
    num_correct = 0
    for batch in iterator:
      images, labels = batch['image'], batch['label']
      logits = model([images, False])
      # Added the posthoc adjustment
      if FLAGS.loss == 'drops':
        # a_y * exp(logit) is equivalent to log(a_y) + logit
        final_alpha_y = [a / b for a, b in zip(g_y, sample_per_cls)]
        final_alpha_y = tf.cast(final_alpha_y, dtype=tf.float32)
        final_alpha_y *= tf.reduce_sum(tf.cast(sample_per_cls,
                                               dtype=tf.float32))
        #  The sign is positive because class prior is in the denominator.
        logits = logits + FLAGS.tau * tf.math.log(tf.cast(final_alpha_y + 1e-12,
                                                          dtype=tf.float32))
      if FLAGS.loss in ['posthoc', 'posthoc_ce']:
        spc = tf.cast(sample_per_cls, dtype=tf.float32)
        # Generate class prior (a list of probabilities: P(Y=i))
        spc_norm = spc / tf.reduce_sum(spc)
        logits = logits - FLAGS.tau * tf.math.log(tf.cast(spc_norm + 1e-12,
                                                          dtype=tf.float32))
      avg_loss += loss_op(labels, logits).numpy()
      num_samples += len(labels)
      num_correct += tf.reduce_sum(
          tf.cast(
              tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels),
              tf.int32)).numpy()
    acc = num_correct / float(num_samples) * 100.
    return avg_loss, acc

  def eval_worst_metrics(ds, g_y, sample_per_cls):
    """Worst group acc on test set."""
    iterator = iter(ds)
    num_samples = [0] * FLAGS.num_classes
    num_correct = [0] * FLAGS.num_classes
    for batch in iterator:
      images, labels = batch['image'], batch['label']
      logits = model([images, False])
      # Added the posthoc adjustment
      if FLAGS.loss == 'drops':
        # alpha_y * exp(logit) is equivalent to log(alpha_y) + logit
        final_alpha_y = [a / b for a, b in zip(g_y, sample_per_cls)]
        final_alpha_y = tf.cast(final_alpha_y, dtype=tf.float32)
        final_alpha_y *= tf.reduce_sum(tf.cast(sample_per_cls,
                                               dtype=tf.float32))
        logits = logits + FLAGS.tau * tf.math.log(tf.cast(final_alpha_y + 1e-12,
                                                          dtype=tf.float32))
      if FLAGS.loss in ['posthoc', 'posthoc_ce']:
        spc = tf.cast(sample_per_cls, dtype=tf.float32)
        spc_norm = spc / tf.reduce_sum(spc)
        logits = logits - FLAGS.tau * tf.math.log(tf.cast(spc_norm + 1e-12,
                                                          dtype=tf.float32))
      acc_list = tf.cast(
          tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels),
          tf.int32)
      for cls_idx in range(FLAGS.num_classes):
        idx_this_class = tf.where(labels == cls_idx)
        idx_this_class = tf.reshape(idx_this_class, [-1])
        num_samples[cls_idx] += len(idx_this_class)
        acc_this_class = tf.gather(acc_list, idx_this_class, axis=0,
                                   batch_dims=acc_list.shape.rank)
        num_correct[cls_idx] += tf.reduce_sum(acc_this_class).numpy()
    acc_per_cls = [a * 100. / b for a, b in zip(num_correct, num_samples)]
    return min(acc_per_cls), acc_per_cls

  def eval_dro_metrics(ds, eps, g_y, sample_per_cls):
    """Eval worst (with least num of train samples) group acc on test set."""
    # get the value of base list
    base_list = [1 for i in range(FLAGS.num_classes)]
    base_weight = tf.cast(base_list, dtype=tf.float64)
    base_weight_norm = base_weight / tf.reduce_sum(base_weight)
    # prepare per class accuracy and number of samples
    iterator = iter(ds)
    num_samples_cls = [0] * FLAGS.num_classes
    num_correct_cls = [0] * FLAGS.num_classes
    for batch in iterator:
      images, labels = batch['image'], batch['label']
      logits = model([images, False])
      # Added the posthoc adjustment
      if FLAGS.loss == 'drops':
        # alpha_y * exp(logit) is equivalent to log(alpha_y) + logit
        final_alpha_y = [a / b for a, b in zip(g_y, sample_per_cls)]
        final_alpha_y = tf.cast(final_alpha_y, dtype=tf.float32)
        final_alpha_y *= tf.reduce_sum(tf.cast(sample_per_cls,
                                               dtype=tf.float32))
        logits = logits + FLAGS.tau * tf.math.log(tf.cast(final_alpha_y + 1e-12,
                                                          dtype=tf.float32))
      if FLAGS.loss in ['posthoc', 'posthoc_ce']:
        spc = tf.cast(sample_per_cls, dtype=tf.float32)
        # Generate class prior (a list of probabilities: P(Y=i))
        spc_norm = spc / tf.reduce_sum(spc)
        logits = logits - FLAGS.tau * tf.math.log(tf.cast(spc_norm + 1e-12,
                                                          dtype=tf.float32))
      acc_list = tf.cast(
          tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels),
          tf.int32)
      for j in range(FLAGS.num_classes):
        idx = tf.where(labels == j)
        idx = tf.reshape(idx, [-1])
        num_samples_cls[j] += len(idx)
        acc_j = tf.gather(acc_list, idx, axis=0,
                          batch_dims=acc_list.shape.rank)
        num_correct_cls[j] += tf.reduce_sum(acc_j).numpy()
    acc_per_cls = [a / b for a, b in zip(num_correct_cls, num_samples_cls)]

    # get worst class weights from the optimization with divergence constrants
    v = cp.Variable(FLAGS.num_classes)
    v.value = v.project(base_weight_norm)
    constraints = [v >= 0, cp.sum(v) == 1]
    if FLAGS.dro_div == 'l2':
      constraints.append(cp.sum(cp.square(v - base_weight_norm)) <= eps)
    elif FLAGS.dro_div == 'l1':
      constraints.append(cp.sum(cp.abs(v - base_weight_norm)) <= eps)
    elif FLAGS.dro_div == 'reverse-kl':
      # D(g, u)=sum_i u_i * [log(u_i) - log(g_i)],
      # g is the parameter v we aim to solve, u_i is the base_weight_norm.
      constraints.append(
          cp.sum(cp.kl_div(base_weight_norm, v)) <= eps)
    elif FLAGS.dro_div == 'kl':
      # D(g, u)=sum_i g_i * [log(g_i) - log(u_i)],
      # g is the parameter v we aim to solve, u_i is the base_weight_norm.
      constraints.append(
          cp.sum(cp.kl_div(v, base_weight_norm)) <= eps)
    tf.print(acc_per_cls)
    obj = cp.Minimize(cp.sum(cp.multiply(v, np.array(acc_per_cls))))
    prob = cp.Problem(obj, constraints)
    try:
      v.value = v.project(base_weight_norm)
      prob.solve(warm_start=True)
    except cp.error.SolverError:
      prob.solve(solver='SCS', warm_start=True)
    print(v.value)
    print(acc_per_cls)
    dro_acc = tf.reduce_sum(tf.multiply(v.value, acc_per_cls))
    return dro_acc.numpy()

  def updateg(ds, g_y, alpha_y, r_list, lambd, delta,
              eta_g=0.01, eta_lambda=0.01, weight_type=FLAGS.weight_type):
    """Pesuedo code 1: (Updating g, lambda).

    Args:
      ds: dataloader
      g_y: weights for y in class list [1, ..., K];
      alpha_y: alpha_y for y in class list [1, ..., K];
      r_list: any prior probability of the weight list;
      lambd: the lagrangian multuplier;
      delta: purturbation level for the divergence constraint;
      eta_g: step size for updating g;
      eta_lambda: step size for updating eta;
      weight_type: ce or 0-1 loss for weight calculation optimization task.

    Returns:
      updated g_y, lambd, alpha_y
    """
    loss_y_list = [0] * FLAGS.num_classes
    num_y_list = [0] * FLAGS.num_classes
    iterator = iter(ds)
    for batch in iterator:
      images, y_true = batch['image'], batch['label']
      y_pred = model([images, False])
      # Main steps for a mini-batch:
      #   Step 1: Prepare w_y (optional--for ce loss)
      #           For each class y in class list [1, ..., K],
      #           generate a sample weight w_y for the batch:
      #               [val_i for i in range(batch_size)]
      #               val_i =1 if the target for sample x_i is y;
      if weight_type == 'ce':
        w_y_list = [-1] * FLAGS.num_classes
        for i in range(FLAGS.num_classes):
          idx = tf.where(y_true == i)
          idx = tf.reshape(idx, [-1])
          idx = idx.numpy().tolist()
          weight_list = [0] * len(y_true)
          for val in idx:
            weight_list[val] = 1
          w_y_list[i] = weight_list
      # Step 2: Update softmaxed of logits get model prediction (post-shift)
      #       softmax_y = softmax of logits
      #       model prediction is: argmax_y (alpha_y * softmax_y)
      y_pred_prob = tf.nn.softmax(y_pred, axis=-1)
      y_weighted_pred = y_pred_prob * alpha_y
      y_weighted_pred = y_weighted_pred / tf.reduce_sum(y_weighted_pred,
                                                        1, keepdims=True)
      arg_pred = tf.argmax(y_pred_prob * alpha_y, axis=1,
                           output_type=tf.int32)

      # Step 3: Calculate L_y (per class loss), L is the 0-1 loss
      #         Get the 0-1 loss for the mini-batch with sample weight w_y;
      #                achieve with 0-1 loss:
      #                  get the index for class y: idx_y
      #                  L_y = 0-1_loss(y_pred[idx_y], y_true[idx_y])
      if weight_type == 'ce':
        labels_oh = maybe_one_hot(y_true, depth=FLAGS.num_classes)
        tmp_loss = tf.keras.backend.categorical_crossentropy(
            labels_oh, y_weighted_pred, from_logits=False)
        for i in range(FLAGS.num_classes):
          tmp = tf.reduce_sum(tf.cast(w_y_list[i], dtype=tf.float32) * tmp_loss)
          num_y_list[i] += len(tf.where(y_true == i))
          loss_y_list[i] += tmp
      else:  # 0-1 loss for L
        acc_list = tf.cast(tf.equal(arg_pred, y_true), tf.int32)
        for i in range(FLAGS.num_classes):
          idx = tf.where(y_true == i)
          num_y_list[i] += len(idx)
          idx = tf.reshape(idx, [-1])
          tmp_loss = 1 - tf.gather(acc_list, idx, axis=0,
                                   batch_dims=acc_list.shape.rank)
          tmp = tf.cast(tf.reduce_sum(tmp_loss), dtype=tf.float32)
          loss_y_list[i] += tmp
    # Continue with accumulated loss for the whole validation set
    loss_y_list /= tf.cast(num_y_list, dtype=tf.float32)
    #       Step 4: Get the Lagrangian constraint term
    #       cons = lambda * (D(r, g) - delta)
    if FLAGS.dro_div == 'l2':
      div = tf.reduce_sum(tf.square(r_list - g_y))
    elif FLAGS.dro_div == 'l1':
      div = tf.reduce_sum(tf.abs(r_list - g_y))
    elif FLAGS.dro_div == 'reverse-kl':
      # D(g_y, r_list)
      tmp = tf.math.log((r_list + 1e-12) / (g_y + 1e-12))
      div = tf.reduce_sum(r_list * tmp)
    elif FLAGS.dro_div == 'kl':
      # D(g_y, r_list)
      tmp = tf.math.log((g_y + 1e-12) / (r_list + 1e-12))
      div = tf.reduce_sum(g_y * tmp)

    div = tf.cast(div, dtype=tf.float32)
    cons = tf.cast(lambd * (div - tf.cast(delta, dtype=tf.float32)),
                   dtype=tf.float32)

    #       Step 5: Get the Lagrangian (with L_y list of size K)
    #       Lagrangian = tf.reduce_sum(g_y_list * L_y_list) + cons
    lagrangian = tf.cast(tf.reduce_sum(g_y * loss_y_list), dtype=tf.float32)
    lagrangian -= cons

    #       Step 6: EG update on g_y
    if FLAGS.g_type == 'eg':
      if FLAGS.dro_div == 'kl':
        part1 = tf.cast(eta_g * loss_y_list, dtype=tf.float32)
        log_r = tf.cast(tf.math.log(r_list + 1e-12), dtype=tf.float32)
        part2 = tf.cast(lambd * eta_g * log_r, dtype=tf.float32)
        neu = tf.add(part1, part2)
        neu += tf.cast(tf.math.log(g_y + 1e-12), dtype=tf.float32)
        den = tf.cast(lambd * eta_g + 1., dtype=tf.float32)
        g_y_updated = tf.exp(tf.divide(neu, den) - 1.)
        g_y_updated /= tf.reduce_sum(g_y_updated)
      elif FLAGS.dro_div == 'reverse-kl':
        part1 = tf.cast(eta_g * lambd * (np.log(r_list + 1e-12).tolist()),
                        dtype=tf.float32)
        neu = tf.add(tf.cast(g_y, dtype=tf.float32), part1)
        den = tf.cast(eta_g * loss_y_list, dtype=tf.float32)
        g_y_updated = tf.divide(neu, den)
        g_y_updated /= tf.reduce_sum(g_y_updated)
    else:
      g_y_updated = r_list * tf.exp(loss_y_list / lambd)
      g_y_updated /= tf.reduce_sum(g_y_updated)
#       Step 7: EG update on lambda
#               lambda <- lambda - eta_lambda * gradient(Lagrangian)_lambda
    lambd_updated = lambd + eta_lambda * cons
    #       Step 8: Update alpha_y
    #               For now: alpha_y = g_y/pi_y
    alpha_y_updated = [a / b for a, b in zip(g_y_updated, samples_per_cls)]
    alpha_y_updated = tf.cast(alpha_y_updated, dtype=tf.float32)
    alpha_y_updated *= tf.reduce_sum(tf.cast(samples_per_cls, dtype=tf.float32))
    return g_y_updated, lambd_updated, alpha_y_updated

  # training
  train_iterator = iter(train_ds)
  best_acc_valid = 0
  best_acc_eval_at_valid = 0
  best_acc_valid_worst = 0
  best_acc_eval_at_valid_worst = 0
  # Used for model selection
  best_acc_valid_dro_sel = 0
  best_acc_eval_at_valid_dro_list = [0] * 11
  best_acc_eval_at_valid_dro_sel = 0
  t0 = time.time()
  for it in range(1, FLAGS.num_iters + 1):
    batch = next(train_iterator)
    images, labels = batch['image'], batch['label']
    loss, _ = train_step(images, labels, it)
    if it % FLAGS.n_it_update == 0 and FLAGS.loss == 'drops':
      # update g_y, lambd, and alpha_y w.r.t. val dataloader
      g_y, lambd, alpha_y = updateg(ds=valid_ds, g_y=g_y, lambd=lambd,
                                    delta=FLAGS.eps,
                                    alpha_y=alpha_y, r_list=r_list,
                                    eta_g=FLAGS.eta_g,
                                    eta_lambda=FLAGS.eta_lambda)
      # g_y, lambd, alpha_y = g_y_u, lambd_u, alpha_y_up
      loss_op = losses_lt.MakeLossFunc(FLAGS.loss, samples_per_cls, FLAGS.gamma,
                                       FLAGS.beta, FLAGS.s, FLAGS.tau)
      info_str = 'It: {}, g_y: {}, lambda: {}'.format(it, g_y, lambd)
      print(info_str)
      fp_log_res.write(info_str + '\n')
    if it % 100 == 0:
      with train_summary_writer.as_default():
        tf.summary.scalar('loss/train', loss.numpy(), step=it)
      info_str = 'It: {}, loss: {:.5f}, time elapsed: {:.3f}'.format(
          it, loss.numpy(),
          time.time() - t0)
      print(info_str)
      fp_log_res.write(info_str + '\n')
    if FLAGS.dro_div == 'reverse-kl':
      eps_list = [i/2 for i in range(21)]
      eps_list[10] = eps_list[-1]
      eps_list[11] = FLAGS.eps
    elif FLAGS.dro_div == 'kl':
      upp_v = np.log(FLAGS.num_classes)
      eps_list = [tmp_v * upp_v / 20 for tmp_v in range(21)]
      eps_list[10] = eps_list[-1]
      eps_list[11] = 1.0
    if it % FLAGS.eval_freq == 0:
      loss_valid, acc_valid = eval_metrics(valid_ds, g_y, samples_per_cls)
      loss_eval, acc_eval = eval_metrics(eval_ds, g_y, samples_per_cls)
      acc_valid_worst, _ = eval_worst_metrics(valid_ds, g_y, samples_per_cls)
      acc_eval_worst, acc_all = eval_worst_metrics(eval_ds, g_y,
                                                   samples_per_cls)
      acc_eval_dro_list = [0] * len(best_acc_eval_at_valid_dro_list)
      for kk in range(len(acc_eval_dro_list)):
        acc_eval_dro_list[kk] = eval_dro_metrics(eval_ds, eps_list[kk],
                                                 g_y, samples_per_cls)
      acc_eval_dro_sel = eval_dro_metrics(eval_ds, eps_list[11],
                                          g_y, samples_per_cls)
      acc_valid_dro_sel = eval_dro_metrics(valid_ds, eps_list[11],
                                           g_y, samples_per_cls)
      # save model
      if acc_valid > best_acc_valid:
        best_acc_valid = acc_valid
        best_acc_eval_at_valid = acc_eval
        model.save(save_dir + '/mean_best.h5')
        _ = save_logits(valid_ds, model, save_dir + '/val_mean_best_')
        _ = save_logits(eval_ds, model, save_dir + '/test_mean_best_')
      if acc_valid_worst > best_acc_valid_worst:
        best_acc_valid_worst = acc_valid_worst
        best_acc_eval_at_valid_worst = acc_eval_worst
        model.save(save_dir + '/worst_best.h5')
        _ = save_logits(valid_ds, model, save_dir + '/val_worst_best_')
        _ = save_logits(eval_ds, model, save_dir + '/test_worst_best_')
      if acc_valid_dro_sel > best_acc_valid_dro_sel:
        model.save(save_dir + '/dro_best.h5')
        _ = save_logits(valid_ds, model, save_dir + '/val_dro_best_')
        _ = save_logits(eval_ds, model, save_dir + '/test_dro_best_')
        best_acc_valid_dro_sel = acc_valid_dro_sel
        for kk in range(len(best_acc_eval_at_valid_dro_list)):
          best_acc_eval_at_valid_dro_list[kk] = acc_eval_dro_list[kk]
        best_acc_eval_at_valid_dro_sel = acc_eval_dro_sel
      with eval_summary_writer.as_default():
        tf.summary.scalar('loss/valid', loss_valid, step=it)
        tf.summary.scalar('loss/eval', loss_eval, step=it)
        tf.summary.scalar('acc/valid', acc_valid, step=it)
        tf.summary.scalar('acc/eval', acc_eval, step=it)
        tf.summary.scalar(
            'acc/best_eval_at_valid', best_acc_eval_at_valid, step=it)
        tf.summary.scalar('acc/valid_worst', acc_valid_worst, step=it)
        tf.summary.scalar('acc/eval_worst', acc_eval_worst, step=it)
        tf.summary.scalar(
            'acc/best_eval_at_valid_worst',
            best_acc_eval_at_valid_worst, step=it)
        for kk in range(len(best_acc_eval_at_valid_dro_list)):
          tf.summary.scalar(f'acc/best_eval_at_valid_dro_{kk}',
                            best_acc_eval_at_valid_dro_list[kk], step=it)
        tf.summary.scalar('acc/best_eval_at_valid_dro_sel',
                          best_acc_eval_at_valid_dro_sel, step=it)

      info_str = (
          'It: {}, Valid loss: {:.3f}, Valid acc: {:.3f}, Valid W-acc: {:.3f}, '
          'Eval loss: {:.3f}, Eval acc: {:.3f}, '
          'Best Valid acc: {:.3f}, Best Valid acc_worst: {:.3f}, '
          'Best Eval acc: {:.3f}, Best Eval acc_worst: {:.3f},'
          ).format(
              it, loss_valid, acc_valid, acc_valid_worst,
              loss_eval, acc_eval, best_acc_valid, best_acc_valid_worst,
              best_acc_eval_at_valid, best_acc_eval_at_valid_worst)
      for kk in range(len(best_acc_eval_at_valid_dro_list)):
        info_str += 'Best Eval acc_dro_{:.3f}: {:.3f}'.format(
            eps_list[kk], best_acc_eval_at_valid_dro_list[kk])
      info_str += (
          'Best Eval acc_dro_sel: {:.3f}, '
          'time elapsed: {:.3f}, Acc list: {}').format(
              best_acc_eval_at_valid_dro_sel, time.time() - t0, acc_all)
      print(info_str)
      fp_log_res.write(info_str + '\n')
  fp_log_res.close()


if __name__ == '__main__':
  app.run(main)

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

r"""Train and evaluate models of long-tail experiments with DROPS only at test time.
"""
import os
import time
from absl import app
from absl import flags
import cvxpy as cp
import numpy as np
import tensorflow as tf
from drops import losses_lt


FLAGS = flags.FLAGS
# Loss: currently support (1) ce; (2) drops

my_flags = [
    'lr', 'model_dir', 'dataset', 'num_iters', 'loss',
    'num_classes', 'batch_size', 'run_id',
    'imb_ratio'
]

for attrname in my_flags:
  if attrname in flags.FLAGS:
    print('deleting {}'.format(attrname))
    delattr(flags.FLAGS, attrname)

flags.DEFINE_string('model_dir',
                    './', 'directory where model and logs are stored')
flags.DEFINE_float('lr', 0.1, 'initial learning rate')
flags.DEFINE_integer('num_iters', 25, 'number of training iterations')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('run_id', 0, 'run id')
flags.DEFINE_string('dataset', 'cifar10', 'dataset: imagenet-lt or cifar')
flags.DEFINE_integer('num_classes', 10, 'number of classes')
flags.DEFINE_float('imb_ratio', 0.01, 'n_min / n_max')
flags.DEFINE_string('loss', 'drops', 'ce/drops/posthoc')
flags.DEFINE_string('prior_type', 'train', 'prior type: val/train')
flags.DEFINE_string('cal_type', 'none', 'cal type: temp/scale/none')

# args for drops
flags.DEFINE_string('dro_div', 'kl', 'div_type for dro metric: kl/l2/l1')
flags.DEFINE_float('eps', 0.2, 'perturbation for dro metric')
#  Fix metric base as recip_prior for now
flags.DEFINE_string(
    'metric_base', 'uniform',
    'class weights for dro metric: prior, recip_prior, uniform')
flags.DEFINE_float('eta_g', 0.01, 'step size for eg updates of eta_g')
flags.DEFINE_float('eta_lambda', 0.1, 'step size for eg updates of eta_lambda')
flags.DEFINE_float('eta_lambda_mult', 1.,
                   'factor to scale eta_lambda every iteration')
flags.DEFINE_string('weight_type', 'ce', 'type of class-re-weight: 0_1/ce')
flags.DEFINE_string('g_type', 'simple',
                    'simplied version or EG style, is eg or not')
flags.DEFINE_string('req', 'dro',
                    'requirement of metric for selected models: mean/dro/worst')


def ce_loss(labels, preds, from_logits=True):
  labels_oh = maybe_one_hot(labels, depth=FLAGS.num_classes)
  cce = tf.keras.backend.categorical_crossentropy(
      labels_oh, preds, from_logits=from_logits)
  return tf.reduce_mean(cce)


def maybe_one_hot(labels, depth):
  if len(labels.shape) > 1:
    return labels
  else:
    return tf.one_hot(labels, depth=depth)


def calibrate_temp(logits, labels, logits_test):
  """Temperature calibration."""
  num_iters = 100
  lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=0.1,
      decay_steps=num_iters,
      end_learning_rate=0.01,
      power=1.0,
      cycle=False)
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
  temp = tf.Variable(initial_value=1., trainable=True)
  for _ in range(num_iters):
    with tf.GradientTape() as tape:
      loss = ce_loss(labels, logits * temp, from_logits=True)
    grads = tape.gradient(loss, [temp])
    optimizer.apply_gradients(zip(grads, [temp]))
    print('loss:', loss.numpy(), 'temp:', temp.numpy())
  return logits * temp, logits_test * temp


def calibrate_shifts(logits, labels, logits_test):
  """Shift calibration."""
  num_iters = 100
  num_classes = logits.shape[-1]
  lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=0.1,
      decay_steps=num_iters,
      end_learning_rate=0.01,
      power=1.0,
      cycle=False)
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
  shifts = tf.Variable(
      initial_value=0.01 * np.random.rand(1, num_classes), trainable=True)
  for _ in range(num_iters):
    with tf.GradientTape() as tape:
      loss = ce_loss(labels, logits - shifts, from_logits=True)
    grads = tape.gradient(loss, [shifts])
    optimizer.apply_gradients(zip(grads, [shifts]))
    print('loss:', loss.numpy(), 'shifts:', np.min(shifts.numpy()),
          np.max(shifts.numpy()))
  return logits - shifts, logits_test - shifts


def calibrate_scales(logits, labels, logits_test):
  """Scale calibration."""
  num_iters = 100
  num_classes = logits.shape[-1]
  lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=0.1,
      decay_steps=num_iters,
      end_learning_rate=0.01,
      power=1.0,
      cycle=False)
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
  scales = tf.Variable(initial_value=tf.ones((1, num_classes)), trainable=True)
  for _ in range(num_iters):
    with tf.GradientTape() as tape:
      loss = ce_loss(labels, logits * scales, from_logits=True)
    grads = tape.gradient(loss, [scales])
    optimizer.apply_gradients(zip(grads, [scales]))
    print('loss:', loss.numpy(), 'scales:', np.min(scales.numpy()),
          np.max(scales.numpy()))
  return logits * scales, logits_test * scales


def calibrate_shifts_scales(logits, labels, logits_test):
  """Shift_scale calibration."""
  num_iters = 100
  num_classes = logits.shape[-1]
  lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=0.1,
      decay_steps=num_iters,
      end_learning_rate=0.01,
      power=1.0,
      cycle=False)
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
  shifts = tf.Variable(
      initial_value=np.random.rand(1, num_classes), trainable=True)
  scales = tf.Variable(initial_value=tf.ones((1, num_classes)), trainable=True)
  for _ in range(num_iters):
    with tf.GradientTape() as tape:
      loss = ce_loss(labels, logits * scales - shifts, from_logits=True)
    grads = tape.gradient(loss, [scales])
    optimizer.apply_gradients(zip(grads, [scales]))
    print('loss:', loss.numpy(), 'scales:', np.min(scales.numpy()),
          np.max(scales.numpy()), 'shifts:', np.min(shifts.numpy()),
          np.max(shifts.numpy()))
  return logits * scales - shifts, logits_test * scales - shifts


def calibrate(logits, labels, logits_test, cal_type='temp'):
  """Perform calibration."""
  if cal_type == 'temp':
    logits, logits_test = calibrate_temp(logits, labels, logits_test)
  elif cal_type == 'shift':
    logits, logits_test = calibrate_shifts(logits, labels, logits_test)
  elif cal_type == 'scale':
    logits, logits_test = calibrate_scales(logits, labels, logits_test)
  elif cal_type == 'shift_scale':
    logits, logits_test = calibrate_scales(logits, labels, logits_test)

  return logits, logits_test


def get_cls_num(dataset_name, imb_factor=None):
  """Get a list of image numbers for each class.

  Given cifar version Num of imgs follows emponential distribution
  img max: 5000 / 500 * e^(-imb * 0);
  img min: 5000 / 500 * e^(-imb * int(dataset_name - 1))
  exp(-imb * (int(dataset_name) - 1)) = img_max / img_min

  Args:
    dataset_name: str, '10', '100
    imb_factor: float, imbalance factor: img_min/img_max, None if geting
      default cifar data number

  Returns:
    img_num_per_cls: a list of number of images per class
  """
  cls_num = 10 if dataset_name == 'cifar10' else 100
  img_max = int(50000 / cls_num)
  if imb_factor is None:
    return [img_max] * cls_num
  img_num_per_cls = []
  for cls_idx in range(cls_num):
    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
    img_num_per_cls.append(min(int(num), img_max))
  return img_num_per_cls


def get_pre_train(base_folder, is_val=True, req='dro'):
  """get the pre-trained logits.

  Args:
    base_folder: the base directory;
    is_val: bool, True if is validation set, else test dataset;
    req: criterion for model selection (mean/dro/worst)

  Returns:
    validation/test logits/labels for training/evaluation.
  """
  # Get logits
  if is_val:
    txt_file = base_folder + f'/val_{req}_best_logits.txt'
  else:
    txt_file = base_folder + f'/test_{req}_best_logits.txt'
  with open(txt_file, encoding='UTF-8') as f:
    full_results = []
    for line in f:
      tmp = line.split(' ')
      for kk in range(len(tmp)):
        full_results.append(float(tmp[kk]))
  if FLAGS.dataset == 'cifar10':
    data_shape = (int(len(full_results) / 10), 10)
  else:
    data_shape = (int(len(full_results) / 100), 100)
  logits = np.reshape(full_results, data_shape)

  # Get labels
  if is_val:
    txt_file = base_folder + f'/val_{req}_best_labels.txt'
  else:
    txt_file = base_folder + f'/test_{req}_best_labels.txt'
  with open(txt_file, encoding='UTF-8') as f:
    full_results = []
    for line in f:
      tmp = line.split(' ')
      for kk in range(len(tmp)):
        full_results.append(float(tmp[kk]))
  labels = np.array(full_results).astype(int)
  return logits, labels


def main(_):
    # Set np seed
  np.random.seed(123)
  FLAGS.model_dir += f'{FLAGS.dataset}_{FLAGS.imb_ratio}/'
  # Default: load the ce model
  FLAGS.model_dir += f'_loss_ce_{FLAGS.run_id}'
  save_dir = os.path.join(FLAGS.model_dir, 'model')

  # log file
  fname = 'results_' + 'loss={},delta_train={},prior_type={},cal_type={},eta_lam={:.2f},eta_lam_mult={:.2f}'.format(
      FLAGS.loss, FLAGS.eps, FLAGS.prior_type, FLAGS.cal_type, FLAGS.eta_lambda,
      FLAGS.eta_lambda_mult)
  fname += FLAGS.weight_type
  fname += '.txt'
  fp_log_res = open(os.path.join(FLAGS.model_dir, fname), 'w+')
  test_logits, test_labels = get_pre_train(base_folder=save_dir, is_val=False)
  val_logits, val_labels = get_pre_train(base_folder=save_dir, is_val=True)

  num_test, num_classes = test_logits.shape[0], test_logits.shape[1]
  num_val = val_logits.shape[0]
  ind_val = [kk for kk in range(num_val)]
  ind_test = [kk for kk in range(num_test)]
  per_class_acc_val = np.zeros(num_classes)
  per_class_acc_test = np.zeros(num_classes)
  val_preds = np.argmax(val_logits, axis=-1)
  test_preds = np.argmax(test_logits, axis=-1)
  for i in range(num_classes):
    ind_v = np.where(val_labels == i)[0]
    ind_t = np.where(test_labels == i)[0]
    per_class_acc_val[i] = np.mean(val_preds[ind_v] == val_labels[ind_v])
    per_class_acc_test[i] = np.mean(test_preds[ind_t] == test_labels[ind_t])
  np.random.shuffle(ind_val)
  np.random.shuffle(ind_test)
  num_val = len(ind_val)
  num_test = len(ind_test)

  print('val set size:', num_val, ', test set size:', num_test)
  print('min/max/mean per class accuracy (val):', min(per_class_acc_val),
        max(per_class_acc_val), np.mean(per_class_acc_val))
  print('min/max/mean per class accuracy (test):', min(per_class_acc_test),
        max(per_class_acc_test), np.mean(per_class_acc_test))

  logits_val = val_logits
  labels_val = val_labels

  logits_test = test_logits
  labels_test = test_labels
  # samples per class in val set and test set
  n_per_class_val = np.zeros(num_classes, dtype=np.int32)
  n_per_class_test = np.zeros(num_classes, dtype=np.int32)
  for i in range(num_classes):
    n_per_class_val[i] = np.sum(labels_val == i)
    n_per_class_test[i] = np.sum(labels_test == i)
  print('min/max per class (val):', min(n_per_class_val), max(n_per_class_val))
  print('min/max per class (test):', min(n_per_class_test),
        max(n_per_class_test))

  # revise class-level weights
  label_probs = get_cls_num(FLAGS.dataset, imb_factor=FLAGS.imb_ratio)
  label_probs /= np.sum(label_probs)
  if FLAGS.prior_type == 'train':
    samples_per_cls = label_probs
  else:
    samples_per_cls = np.ones(num_classes, dtype=np.float32) / num_classes
  if FLAGS.cal_type != 'none':
    logits_val, logits_test = calibrate(
        logits_val, labels_val, logits_test, cal_type=FLAGS.cal_type)
    if isinstance(logits_val) != np.ndarray:
    # if type(logits_val) != np.ndarray:
      logits_val = logits_val.numpy()
      logits_test = logits_test.numpy()
  # Initialize g_y to be the same as uniform | prior p(y)| 1/p(y)
  # Default is set to be 'uniform'
  if FLAGS.metric_base == 'uniform':
    g_y = [1] * FLAGS.num_classes
  elif FLAGS.metric_base == 'prior':
    g_y = [1 / i for i in samples_per_cls]
  elif FLAGS.metric_base == 'recip_prior':
    g_y = samples_per_cls
  g_y = g_y / np.sum(g_y)
  g_y = tf.convert_to_tensor(g_y, dtype=tf.float32)
  # fix alpha_y to be g_y for now
  alpha_y = [1 / i for i in samples_per_cls]
  alpha_y = tf.cast(alpha_y, dtype=tf.float32)
  alpha_y *= tf.reduce_sum(tf.cast(samples_per_cls, dtype=tf.float32))
  lambd = 1.0
  # set the r_list to be the u in the constraint D(u, g) < delta
  r_list = g_y
  # loss (initialized loss with)
  loss_op = losses_lt.MakeLossFunc(FLAGS.loss, samples_per_cls)

  def eval_metrics(labels, logits, g_y, sample_per_cls):
    """Evaluate accuracy on test set."""
    avg_loss = 0.
    num_samples = 0
    num_correct = 0
    if FLAGS.loss == 'drops':
      # a_y * exp(logit) is equivalent to log(a_y) + logit
      final_alpha_y = [a / b for a, b in zip(g_y, sample_per_cls)]
      final_alpha_y = tf.cast(final_alpha_y, dtype=tf.float32)
      final_alpha_y *= tf.reduce_sum(tf.cast(sample_per_cls, dtype=tf.float32))
      #  The sign is positive because class prior is in the denominator.
      logits = logits + tf.math.log(
          tf.cast(final_alpha_y + 1e-12, dtype=tf.float32))
    if FLAGS.loss == 'posthoc':
      spc = tf.cast(sample_per_cls, dtype=tf.float32)
      # Generate class prior (a list of probabilities: P(Y=i))
      spc_norm = spc / tf.reduce_sum(spc)
      logits = logits - tf.math.log(
          tf.cast(spc_norm + 1e-12, dtype=tf.float32))
    avg_loss += loss_op(labels, logits).numpy()
    num_samples += len(labels)
    num_correct += tf.reduce_sum(
        tf.cast(
            tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels),
            tf.int32)).numpy()

    # avg_loss /= num_samples
    acc = num_correct / float(num_samples) * 100.
    return avg_loss, acc

  def eval_worst_metrics(labels, logits, g_y, sample_per_cls):
    """Worst group acc on test set."""
    num_samples = [0] * FLAGS.num_classes
    num_correct = [0] * FLAGS.num_classes
    if FLAGS.loss == 'drops':
      # alpha_y * exp(logit) is equivalent to log(alpha_y) + logit
      final_alpha_y = [a / b for a, b in zip(g_y, sample_per_cls)]
      final_alpha_y = tf.cast(final_alpha_y, dtype=tf.float32)
      final_alpha_y *= tf.reduce_sum(tf.cast(sample_per_cls, dtype=tf.float32))
      logits = logits + tf.math.log(
          tf.cast(final_alpha_y + 1e-12, dtype=tf.float32))
    if FLAGS.loss == 'posthoc':
      spc = tf.cast(sample_per_cls, dtype=tf.float32)
      # Generate class prior (a list of probabilities: P(Y=i))
      spc_norm = spc / tf.reduce_sum(spc)
      logits = logits - tf.math.log(
          tf.cast(spc_norm + 1e-12, dtype=tf.float32))
    acc_list = tf.cast(
        tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels),
        tf.int32)
    for cls_idx in range(FLAGS.num_classes):
      idx_this_class = tf.where(labels == cls_idx)
      idx_this_class = tf.reshape(idx_this_class, [-1])
      num_samples[cls_idx] += len(idx_this_class)
      acc_this_class = tf.gather(
          acc_list, idx_this_class, axis=0, batch_dims=acc_list.shape.rank)
      num_correct[cls_idx] += tf.reduce_sum(acc_this_class).numpy()
    acc_per_cls = [a * 100. / b for a, b in zip(num_correct, num_samples)]
    return min(acc_per_cls), acc_per_cls

  def eval_dro_metrics(labels, logits, eps, g_y, sample_per_cls):
    """Eval worst (with least num of train samples) group acc on test set."""
    # get the value of base list
    base_list = [1 for i in range(FLAGS.num_classes)]
    base_weight = tf.cast(base_list, dtype=tf.float64)
    base_weight_norm = base_weight / tf.reduce_sum(base_weight)
    # prepare per class accuracy and number of samples
    num_samples_cls = [0] * FLAGS.num_classes
    num_correct_cls = [0] * FLAGS.num_classes
    # Added the posthoc adjustment
    if FLAGS.loss == 'drops':
      # alpha_y * exp(logit) is equivalent to log(alpha_y) + logit
      spc = tf.cast(sample_per_cls, dtype=tf.float32)
      spc_norm = spc / tf.reduce_sum(spc)
      logits = logits - (
          tf.math.log(tf.cast(spc_norm + 1e-12, dtype=tf.float32)) +
          tf.math.log(g_y))
    elif FLAGS.loss == 'posthoc':
      spc = tf.cast(sample_per_cls, dtype=tf.float32)
      # Generate class prior (a list of probabilities: P(Y=i))
      spc_norm = spc / tf.reduce_sum(spc)
      logits = logits - tf.math.log(
          tf.cast(spc_norm + 1e-12, dtype=tf.float32))
    acc_list = tf.cast(
        tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels),
        tf.int32)
    for j in range(FLAGS.num_classes):
      idx = tf.where(labels == j)
      idx = tf.reshape(idx, [-1])
      num_samples_cls[j] += len(idx)
      acc_j = tf.gather(acc_list, idx, axis=0, batch_dims=acc_list.shape.rank)
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
      constraints.append(cp.sum(cp.kl_div(base_weight_norm, v)) <= eps)
    elif FLAGS.dro_div == 'kl':
      # D(g, u)=sum_i g_i * [log(g_i) - log(u_i)],
      # g is the parameter v we aim to solve, u_i is the base_weight_norm.
      constraints.append(cp.sum(cp.kl_div(v, base_weight_norm)) <= eps)
    # tf.print(acc_per_cls)
    # print('min/max acc per class:', min(acc_per_cls), max(acc_per_cls))
    obj = cp.Minimize(cp.sum(cp.multiply(v, np.array(acc_per_cls))))
    prob = cp.Problem(obj, constraints)
    try:
      v.value = v.project(base_weight_norm)
      prob.solve(warm_start=True)
    except cp.error.SolverError:
      prob.solve(solver='SCS', warm_start=True)
    dro_acc = tf.reduce_sum(tf.multiply(v.value, acc_per_cls))
    kl_cons = kl_div(v.value, base_weight_norm)
    return dro_acc.numpy(), kl_cons

  def kl_div(p, q):
    return np.sum(p * np.log((np.abs(p) + 1e-7) / (np.abs(q) + 1e-7)))

  def updateg(labels,
              logits,
              g_y,
              alpha_y,
              r_list,
              lambd,
              delta,
              eta_g=0.01,
              eta_lambda=0.01,
              weight_type=FLAGS.weight_type):
    """Pesuedo code 1: (Updating g, lambda).

    Args:
      labels: labels;
      logits: logits;
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
    # shuffle the data
    ind_perm = np.random.permutation(len(labels))
    labels = labels[ind_perm]
    logits = logits[ind_perm]
    # num_iters = len(labels) // FLAGS.batch_size
    # running_ind = 0
    y_true = labels
    y_pred = logits
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
    y_pred_prob = tf.cast(y_pred_prob, dtype=tf.float32)
    y_weighted_pred = y_pred_prob * tf.cast(alpha_y, dtype=tf.float32)
    y_weighted_pred = y_weighted_pred / tf.reduce_sum(
        y_weighted_pred, 1, keepdims=True)
    # print('alpha_y:', min(alpha_y), max(alpha_y))
    arg_pred = tf.argmax(y_pred_prob * alpha_y, axis=1, output_type=tf.int32)

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
        tmp_loss = 1 - tf.gather(
            acc_list, idx, axis=0, batch_dims=acc_list.shape.rank)
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
    cons = tf.cast(
        # lambd * (div - tf.cast(delta, dtype=tf.float32)), dtype=tf.float32)
        (div - tf.cast(delta, dtype=tf.float32)),
        dtype=tf.float32)

    #       Step 5: Get the Lagrangian (with L_y list of size K)
    #       Lagrangian = tf.reduce_sum(g_y_list * L_y_list) + cons
    lagrangian = tf.cast(tf.reduce_sum(g_y * loss_y_list), dtype=tf.float32)
    lagrangian -= lambd * cons

    #       Step 6: EG update on g_y
    print('losses min/max:', min(loss_y_list.numpy()), max(loss_y_list.numpy()))
    print('g_y min/max:', min(g_y.numpy()), max(g_y.numpy()))
    if FLAGS.g_type == 'eg':
      if FLAGS.dro_div == 'kl':
        part1 = tf.cast(eta_g * loss_y_list, dtype=tf.float32)
        log_r = tf.cast(tf.math.log(r_list + 1e-12), dtype=tf.float32)
        part2 = tf.cast(lambd * eta_g * log_r, dtype=tf.float32)
        neu = tf.add(part1, part2)
        neu += tf.cast(tf.math.log(g_y + 1e-12), dtype=tf.float32)
        den = tf.cast(lambd * eta_g + 1., dtype=tf.float32)
        g_y_updated = tf.exp(tf.divide(neu, den) - 1.)
        # g_y_updated = g_y * tf.exp(eta_g * tf.add(g_y * loss_y_list, cons))
        g_y_updated /= tf.reduce_sum(g_y_updated)
      elif FLAGS.dro_div == 'reverse-kl':
        part1 = tf.cast(
            eta_g * lambd * (np.log(r_list + 1e-12).tolist()), dtype=tf.float32)
        neu = tf.add(tf.cast(g_y, dtype=tf.float32), part1)
        den = tf.cast(eta_g * loss_y_list, dtype=tf.float32)
        g_y_updated = tf.divide(neu, den)
        g_y_updated /= tf.reduce_sum(g_y_updated)
    else:
      #  Update g_y by adopting g_i <- g_i * exp(L_i / lambda)
      # g_y_updated = g_y * tf.exp(loss_y_list / lambd)
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
    return g_y_updated, lambd_updated, alpha_y_updated, cons

  # training
  best_acc_valid = 0
  best_acc_eval_at_valid = 0
  best_acc_valid_worst = 0
  best_acc_eval_at_valid_worst = 0
  # Used for model selection
  # best_acc_valid_dro_sel = 0
  best_acc_eval_at_valid_dro_sel = 0
  t0 = time.time()

  if FLAGS.dro_div == 'reverse-kl':
    eps_list = np.arange(0, 1.06, 0.05)
    eps_list = np.concatenate((eps_list, np.arange(1.1, 3.2, 0.1)))
    eps_list = np.concatenate((eps_list, np.arange(3.5, 7, 0.5)))
    eps_list[11] = FLAGS.eps
  elif FLAGS.dro_div == 'kl':
    eps_list = np.arange(0, 0.16, 0.01)
    eps_list = np.concatenate((eps_list, np.arange(0.2, 1.06, 0.05)))
    eps_list = np.concatenate((eps_list, np.arange(1.1, 3.2, 0.1)))
    eps_list = np.concatenate((eps_list, np.arange(3.5, 5, 0.5)))
    eps_list = np.concatenate((eps_list, np.array([FLAGS.eps])))
  print('delta evals:', eps_list)

  acc_eval_dro = np.zeros_like(eps_list)
  acc_val_dro = np.zeros_like(eps_list)
  kl_cons = 1000 * np.ones_like(eps_list)
  best_acc_eval_at_valid_dro = np.zeros_like(eps_list)
  cons = np.inf
  if FLAGS.loss == 'drops' and FLAGS.eps > 0:
    # update g_y, lambd, and alpha_y w.r.t. val dataloader
    # Initialized g_y, alpha_y are specified at Line 427
    eta_lambda = FLAGS.eta_lambda
    for it in range(FLAGS.num_iters):
      if np.abs(cons) < 1e-5:
        break
      g_y, lambd, alpha_y, cons = updateg(
          labels=labels_val,
          logits=logits_val,
          g_y=g_y,
          lambd=lambd,
          delta=FLAGS.eps,
          alpha_y=alpha_y,
          r_list=r_list,
          eta_g=FLAGS.eta_g,
          eta_lambda=eta_lambda)
      eta_lambda *= FLAGS.eta_lambda_mult
      info_str = ('It: {}, g_y: {} / {}, alpha_y: {}/ {}, lambda: {}, '
                  'constraint: {}').format(it, min(g_y), max(g_y),
                                           min(alpha_y.numpy()),
                                           max(alpha_y.numpy()), lambd, cons)
      print(info_str)
      fp_log_res.write(info_str + '\n')
  loss_valid, acc_valid = eval_metrics(labels_val, logits_val, g_y,
                                       samples_per_cls)
  loss_eval, acc_eval = eval_metrics(labels_test, logits_test, g_y,
                                     samples_per_cls)
  acc_valid_worst, _ = eval_worst_metrics(labels_val, logits_val, g_y,
                                          samples_per_cls)
  _, acc_all = eval_worst_metrics(labels_test, logits_test, g_y,
                                  samples_per_cls)
  for i in range(len(eps_list)):
    acc_eval_dro[i], kl_cons[i] = eval_dro_metrics(labels_test, logits_test,
                                                   eps_list[i], g_y,
                                                   samples_per_cls)
    acc_val_dro[i], kl_cons[i] = eval_dro_metrics(labels_val, logits_val,
                                                  eps_list[i], g_y,
                                                  samples_per_cls)

    print('calculating metric for:', i, '/', len(eps_list),
          '{:.2}'.format(eps_list[i]), 'val: {:.3f}'.format(acc_val_dro[i]),
          'test: {:.3f}'.format(acc_eval_dro[i]))
  acc_eval_dro_sel, _ = eval_dro_metrics(labels_test, logits_test, eps_list[11],
                                         g_y, samples_per_cls)
  # acc_valid_dro_sel, _ = eval_dro_metrics(labels_val, logits_val,
  #                                         eps_list[11], g_y, samples_per_cls)
  # best_acc_valid_dro_sel = acc_valid_dro_sel
  for i in range(len(eps_list)):
    best_acc_eval_at_valid_dro[i] = acc_eval_dro[i]
  best_acc_eval_at_valid_dro_sel = acc_eval_dro_sel
  it = 0
  info_str = (
      'It: {}, Valid loss: {:.3f}, Valid acc: {:.3f}, Valid W-acc: {:.3f}, '
      'Eval loss: {:.3f}, Eval acc: {:.3f}, '
      'Best Valid acc: {:.3f}, Best Valid acc_worst: {:.3f}, '
      'Best Eval acc: {:.3f}, Best Eval acc_worst: {:.3f},\n'
      'Best Eval acc_dro_sel: {:.3f}\n,'
      'time elapsed: {:.3f}, Acc min/max: {} / {}\n\n').format(
          it, loss_valid, acc_valid, acc_valid_worst, loss_eval, acc_eval,
          best_acc_valid, best_acc_valid_worst, best_acc_eval_at_valid,
          best_acc_eval_at_valid_worst, best_acc_eval_at_valid_dro_sel,
          time.time() - t0, min(acc_all), max(acc_all))
  # print(info_str)
  for i in range(len(eps_list)):
    info_str += 'DRO Acc (delta_eval = {:.3f}): {:.3f}, kl_cons: {:.3f}\n'.format(
        eps_list[i], best_acc_eval_at_valid_dro[i], kl_cons[i])
  print(info_str)

  info_str += '\ndelta_evals=\n'
  for i in range(len(eps_list)):
    info_str += '{:.3f},'.format(eps_list[i])
  info_str += '\nDRO-Acc=\n'
  for i in range(len(eps_list)):
    info_str += '{:.3f},'.format(best_acc_eval_at_valid_dro[i])
  info_str += '\nkldiv_cons=\n'
  for i in range(len(eps_list)):
    info_str += '{:.3f},'.format(kl_cons[i])

  fp_log_res.write(info_str + '\n')
  fp_log_res.close()

if __name__ == '__main__':
  app.run(main)

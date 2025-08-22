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

"""Main components and the training loop of MetaPose."""

import functools
import os
from typing import Sequence, Tuple, Callable, Mapping, Union, Literal, Any, Optional, Iterator  # pylint: disable=g-importing-member,g-multiple-import

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from metapose import data_utils
from metapose import inference_time_optimization as inf_opt
from metapose import vmap


# DATA
_DATA_ROOT = flags.DEFINE_string('data_root', '', '')
_DATASET = flags.DEFINE_string('dataset', 'h36m/opt/4', '')
_DATA_SPLITS = flags.DEFINE_list('data_splits', ['train', 'test'], '')
_N_CAM = flags.DEFINE_integer('n_cam', 6, '')
_N_JOINT = flags.DEFINE_integer('n_joint', 17, '')
_N_MIX_COMP = flags.DEFINE_integer('n_mix_comp', 4, 'number of GMM components')
_TRAIN_BATCH_SIZE = flags.DEFINE_integer('train_batch_size', 128, '')
_EVAL_BATCH_SIZE = flags.DEFINE_integer('eval_batch_size', 128, '')
_PERMUTE_CAMS_AUG = flags.DEFINE_bool(
    'permute_cams_aug', True,
    'whether to augment data by permuting cameras during training')
_STANDARDIZE_INIT_BEST = flags.DEFINE_bool(
    'standardize_init_best', False,
    'whether to rotate scale and shift init and sgd poses to unified view')
_TRAIN_REPEAT_K = flags.DEFINE_integer(
    'train_repeat_k', 100,
    'repeat training data X times before caching (if permute_cams_aug=true)')
_VALID_FIRST_N = flags.DEFINE_integer(
    'valid_first_n', 64, '`-1` means validate on eval')
_SHUFFLE_SIZE = flags.DEFINE_integer('shuffle_size', 512, '')

_DATASET_WARMUP = flags.DEFINE_bool(
    'dataset_warmup', True,
    'whether to warm-up dataset chache before training (disable during dryrun)')
_DEBUG_TAKE_N_TRAIN_BATCHES = flags.DEFINE_integer(
    'debug_take_n_train_batches', -1, '')
_DEBUG_TAKE_N_EVAL_BATCHES = flags.DEFINE_integer(
    'debug_take_n_eval_batches', -1, '')
_DEBUG_SHOW_SINGLE_FRAME_PMPJES = flags.DEFINE_bool(
    'debug_show_single_frame_pmpjes', True, '')
_DEBUG_ALLOW_MIRROR_IN_NMPJES = flags.DEFINE_bool(
    'debug_allow_mirror_in_nmpjes', True, '')
_DEBUG_REPLACE_XINIT_WITH_NOISE = flags.DEFINE_bool(
    'debug_replace_xinit_with_noise', False, '')
_DEBUG_CACHE_DATASET = flags.DEFINE_bool(
    'debug_cache_dataset', True, '')

# LOGGGING
_ = flags.DEFINE_string('experiment_name', '', '')
_TB_LOG_DIR = flags.DEFINE_string('tb_log_dir', '', '')

# TRAINING
_EARLY_STOPPING_PATIENCE = flags.DEFINE_integer(
    'early_stopping_patience', 50, '')
_EPOCHS_PER_STAGE = flags.DEFINE_integer('epochs_per_stage', 300, '')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-4, '')
_MAX_N_STAGES = flags.DEFINE_integer('max_n_stages', 10, '')
_MAX_STAGE_ATTEMPTS = flags.DEFINE_integer('max_stage_attempts', 100, '')
_USE_EQUIVARIANT_MODEL = flags.DEFINE_bool('use_equivariant_model', False, '')
_ACTIVATION_FN = flags.DEFINE_string('activation_fn', 'selu', '')
_DEBUG_ENABLE_CHECK_NUMERICS = flags.DEFINE_bool(
    'debug_enable_check_numerics', False, '')
_DEBUG_DROP_MLP_INPUTS = flags.DEFINE_list(
    'debug_drop_mlp_inputs', [],
    'A comma-separated list of inputs to the MLP to drop/ignore during '
    'training and inference. Can contain [pose, cams, heatmaps, projs, loss].')
_DEBUG_RUN_EAGERLY = flags.DEFINE_bool('debug_run_eagerly', False, '')

_LOAD_WEIGHTS_FROM = flags.DEFINE_string('load_weights_from', '', '')
_SAVE_PREDS_TO = flags.DEFINE_string('save_preds_to', '', '')
_LOAD_STAGES_N = flags.DEFINE_integer('load_stages_n', 1, '')


# training > for non-equivariant model
_MODEL_MLP_SIZES = flags.DEFINE_list(
    'model_mlp_sizes', list(map(str, [512, 512, 512, 128])), '')

# training > for equivariant model
_MAIN_MLP_SPEC = flags.DEFINE_list(
    'main_mlp_spec',
    list(map(str, [512, 512, 'cn', 512, 512, 'cn', 512, 512, 'cn', 512])), '')
_POSE_MLP_SIZES = flags.DEFINE_list(
    'pose_mlp_sizes', list(map(str, [256, 128])), '')
_POSE_EMB_SIZE = flags.DEFINE_integer('pose_emb_size', 512, '')

_LAMBDA_XOPT_LOSS = flags.DEFINE_float('lambda_xopt_loss', 0.0, '')
_ = flags.DEFINE_float('lambda_soln_pose', 1.0, '')
_ = flags.DEFINE_float('lambda_soln_scale', 0.0, '')
_ = flags.DEFINE_float('lambda_soln_log_scale', 1.0, '')
_ = flags.DEFINE_float('lambda_soln_shift', 1.0, '')
_ = flags.DEFINE_float(
    'lambda_soln_rot_3x3_mse', 0.0, '')
_ = flags.DEFINE_float(
    'lambda_soln_rot_6d_mse', 0.0, '')
_ = flags.DEFINE_float(
    'lambda_soln_rot_inv_mse', 1.0, '')
_ = flags.DEFINE_float(
    'lambda_soln_rot_inv_trace', 0.0, '')

_LAMBDA_FWD_LOSS = flags.DEFINE_float('lambda_fwd_loss', 1.0, '')
_LAMBDA_LOGP_LOSS = flags.DEFINE_float('lambda_logp_loss', 0.0, '')

_USE_BONE_LEN = flags.DEFINE_bool('use_bone_len', False, '')
_BONE_COUNT = flags.DEFINE_integer('bone_count', 16, '')
_LAMBDA_LIMB_LEN_LOSS = flags.DEFINE_float('lambda_limb_len_loss', 0.0, '')

flags.mark_flag_as_required('tb_log_dir')

tfk = tf.keras
tfkl = tf.keras.layers
tfk.backend.set_floatx('float64')


class NamedWeightedLossModel(tfk.models.Model):
  """Adds a method to add named weighted losses during model construction.

  Analogous to model.add_loss, but:
  1) automatically adds a metric with the same name
  2) allows specifying the weight of added losses at model.compile time as:
    ```
    def get_model():
      model = ...
      custom_loss = ...
      model.add_named_loss(custom_loss, 'my_fancy_loss')
      return model

    model = get_model()
    model.compile(
      ...,
      loss=losses,
      loss_weights={<weights of losses in `losses`>},
      custom_loss_weights={<weights of losses added via `add_named_loss`>}
    )
    ```
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._custom_losses = {}

  def add_named_loss(
      self, value, name, default_weight = 0.0):

    if name in self._custom_losses:
      if self._custom_losses[name][0].name != value.name:
        raise ValueError('already have a loss with name %s' % name)

    weight = self._custom_loss_weights.get(name, default_weight)
    self.add_loss(value * weight)
    self.add_metric(value, name)

  def compile(
      self,
      custom_loss_weights = None,
      **kwargs):

    custom_loss_weights = custom_loss_weights or {}
    self._custom_loss_weights = custom_loss_weights
    super().compile(**kwargs)


class TensorflowSaveWeightsFix(tfk.callbacks.Callback):
  """A fix allowing saving models with custom losses."""

  def __init__(self):
    super(TensorflowSaveWeightsFix, self).__init__()
    self._backup_loss = None

  def on_train_begin(self, logs = None):
    self._backup_loss = {**self.model.loss}

  def on_train_batch_end(self, batch, logs = None):
    self.model.loss = self._backup_loss

SolutionTaskFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


class DeepInverseSolver(NamedWeightedLossModel):
  """A meta model for training models similar to Deep Inverse Solver.

  We use a similar terminology to "Deep Feedback Inverse Problem Solver"
  by Ma et al., but the best MetaPose model did _not_ use some of components
  descripted below.

  On high level, given observations `task_vecs` this model tries to predict
  a solution `x_opt` such that the forward simulation with this solution:
  `fwd = task_fwd_fn(x_opt, task_vecs)` has low evaluation loss
  `task_loss = task_eval_fn(fwd, task_vecs)`.

  For example, for MetaPose:
    `task_vecs` - per-view pose heatmaps compressed into GMM parameters
    `x_opt` - 3D pose + camera parameters concated into a single vector
    `task_fwd_fn` - projects 3D poses using cameras (ignores `task_vecs`)
    `fwd` - a collection of such 2D projections
    `task_eval_fn` - computes the likelihood of 2D projections given GMM params

  In addition, it takes an inital guess for the solution of the inverse
  problem (`x_init`) as an input.

  All vectors above are passes as flattened to [batch_size, -1]. Let us call
  corresponding dimentions x_dim, task_dim, fwd_dim.

  During construction it takes a list of keras models `models` where each model
  maps a tensor of size [batch, x_dim + task_dim + fwd_dim + 2] into a tensor
  of shape [batch, x_dim] with an additive update to the current guess x_cur.

  During training, only the LAST model in `models` is trained, other are frozen.

  Since the model outputs both the final `x_opt` and final `fwd`, the end user
  can apply supervised losses if the know the "GT solution" (e.g. the way
  Ma et al. did) or know the "GT forward result" (the way MetaPose does).

  Additional outputs can be requested by providing `custom_output_fns`.
  """

  def __init__(
      self,
      models,
      task_fwd_fn,
      task_eval_fn,
      task_loss_name,
      custom_output_fns,
      **kwargs):
    super().__init__(**kwargs)
    self.models = models
    self.task_fwd_fn = task_fwd_fn
    self.task_eval_fn = task_eval_fn
    self.task_loss_name = task_loss_name
    self.custom_output_fns = custom_output_fns

  def call(self, inputs):
    x_inits, task_vecs = inputs['x_init'], inputs['task_vec']
    k_steps = len(self.models)
    batch_size = tf.shape(x_inits)[0]
    xs_cur = x_inits
    path = [xs_cur]
    for step_i in range(k_steps):
      fwds_cur = self.task_fwd_fn(xs_cur, task_vecs)
      vals_cur = self.task_eval_fn(fwds_cur, task_vecs)
      vals_cur = tf.reshape(vals_cur, (batch_size, 1))
      step_i_tensor = tf.convert_to_tensor(float(step_i))
      step_i_inputs = tf.repeat(step_i_tensor[None, None], batch_size, axis=0)
      inputs = [xs_cur, task_vecs, fwds_cur, vals_cur, step_i_inputs]
      inputs = [tf.cast(x, tf.float64) for x in inputs]
      inputs_tensor = tf.concat(inputs, axis=-1)
      inputs_tensor = tf.stop_gradient(inputs_tensor)
      xs_update = self.models[step_i](inputs_tensor)
      xs_cur = tf.stop_gradient(xs_cur) + xs_update
      path.append(xs_cur)

    fwds_cur = self.task_fwd_fn(xs_cur, task_vecs)
    vals_cur = self.task_eval_fn(fwds_cur, task_vecs)
    last_task_loss = tf.reduce_mean(vals_cur)

    self.add_named_loss(last_task_loss, name=self.task_loss_name)

    outputs = {'x_opt': xs_cur, 'fwd': fwds_cur}

    for output_name, custom_fn in self.custom_output_fns.items():
      outputs[output_name] = custom_fn(xs_cur, task_vecs)

    return outputs


class EvaluteOnDatasetCallback(tfk.callbacks.Callback):
  """A callback for periodically reporting all metrics on a given dataset."""

  def __init__(self,
               log_dir,
               dataset,
               ds_name,
               freq = 1):
    self.log_dir = log_dir
    self.writer = tf.summary.create_file_writer(os.path.join(log_dir, ds_name))
    self.dataset = dataset
    self.ds_name = ds_name
    self.freq = freq

  def on_epoch_end(self, epoch, logs = None):
    logs = logs or {}
    if epoch % self.freq != 0:
      return

    # workaround for a bug
    # pylint:disable=protected-access
    eval_handler = self.model._eval_data_handler
    self.model._eval_data_handler = None
    test_metric_values = self.model.evaluate(self.dataset)
    self.model._eval_data_handler = eval_handler

    metrics_names = [self.ds_name + '_' + x for x in self.model.metrics_names]
    test_metrics = dict(zip(metrics_names, test_metric_values))
    print(' - '.join('%s %.4g' % kv for kv in test_metrics.items()))
    logs.update(test_metrics)

    with self.writer.as_default(epoch):
      for name, value in zip(self.model.metrics_names, test_metric_values):
        tf.summary.scalar('epoch_' + name, value)


SummaryWriterGetter = Callable[[], tf.summary.SummaryWriter]
WriterGetterDatasetTuple = Tuple[SummaryWriterGetter, tf.data.Dataset]


class WriteStageMetrics(tfk.callbacks.Callback):
  """A callback for reporting and saving stage-wise learned models."""

  def __init__(
      self,
      dataset_writers,
      stage,
      monitor,
      prev_best_val_score
    ):
    self.dataset_writers = dataset_writers
    self.monitor = monitor
    self.prev_best_val_score = prev_best_val_score
    self.cur_best_val_score = np.inf
    self.stage = stage

  def on_epoch_end(self, epoch, logs = None):
    logs = logs or {}
    if logs[self.monitor] < self.cur_best_val_score:
      self.cur_best_val_score = logs[self.monitor]

  def on_train_end(self, logs = None):
    if self.cur_best_val_score >= self.prev_best_val_score:
      print('no improvement %.3f (new) >= %.3f, not writing to the '
            'tensorboard stage stats' %
            (self.cur_best_val_score, self.prev_best_val_score))
      return

    print('improved (!) %.3f (new) < %.3f, writing tensorboard stage stats' %
          (self.cur_best_val_score, self.prev_best_val_score))

    self.model.load_weights('/tmp/best-model')

    for _, (writer_getter, dataset) in self.dataset_writers.items():
      with writer_getter().as_default(step=self.stage):
        metric_values = self.model.evaluate(dataset, verbose=0)
        for metric_name, value in zip(self.model.metrics_names, metric_values):
          tf.summary.scalar('stage_' + metric_name, value)

# ---------

PackSpec = Sequence[Sequence[int]]  # e.g. [(3, 2), (5, )]
ShapesSpecs = Tuple[PackSpec, PackSpec]  # soln spec + task spec


def pack_tensors_unbatched(
    tensors):
  # pylint:disable=protected-access
  pack, spec = vmap._pack_tensors_across_batch([x[None] for x in tensors])
  return pack[0], spec


def unpack_and_unparam(
    cur_x, x_spec
):
  # pylint:disable=protected-access
  (x_pose, x_rot_re, x_scale_re,
   x_shift) = vmap._unpack_single_tensor(cur_x, x_spec)
  x_rot, x_scale = inf_opt.unreparam(x_rot_re, x_scale_re)
  return x_pose, x_rot, x_rot_re, x_scale, x_scale_re, x_shift


def normalized_bone_length(pose):
  bone_lens = inf_opt.get_h36m_edge_lens(pose)
  return bone_lens / tf.reduce_mean(bone_lens)


def inv_rot_weak_proj(p_orig, rot_orig,
                      rot):
  p_orig, rot_orig, rot = [
      tf.cast(x, tf.float64) for x in [p_orig, rot_orig, rot]]

  p_new = tf.einsum('jd,do->jo', p_orig, rot)
  rot_new = tf.einsum('kba,bc->kca', rot_orig, rot)
  return p_new, rot_new


def inv_shift_weak_proj(p_orig, rot_orig,
                        s_orig, t_orig,
                        shift):
  p_orig, rot_orig, s_orig, t_orig, shift = [
      tf.cast(x, tf.float64) for x in [p_orig, rot_orig, s_orig, t_orig, shift]]

  p_new = p_orig - shift[None, :]
  new_sh = tf.einsum('jd,kdo->kjo', shift[None, :], rot_orig)[:, 0, :]
  t_new = t_orig + new_sh * s_orig[:, None]
  return p_new, t_new


def inv_scale_weak_proj(p_orig, s_orig,
                        rescale):
  p_orig, s_orig, rescale = [
      tf.cast(x, tf.float64) for x in [p_orig, s_orig, rescale]]

  return p_orig / rescale, s_orig * rescale


def get_std_rotation(pose_3d):
  """Find a rotation that puts the pose into the "standard" reference frame."""
  norm = tf.linalg.norm
  z_id = 8  # Thorax
  y_ids = [1, 4]  # right hip, left hip
  old_positions = tf.gather(pose_3d, [z_id, y_ids[0], y_ids[1]])
  new_z = norm(pose_3d[z_id])
  new_hip_xy = (norm(pose_3d[y_ids[0]]) + norm(pose_3d[y_ids[1]]))/2
  new_positions = tf.convert_to_tensor(
      [[0.0, 0.0, new_z],
       [0.0, -new_hip_xy, 0.0],
       [0.0, new_hip_xy, 0.0]])

  _, u, v = tf.linalg.svd(tf.transpose(old_positions) @ new_positions)
  rot_guess = u @ v

  flip = tf.cond(
      tf.math.less(tf.linalg.det(rot_guess), 0),
      lambda: tf.constant([[-1, 1, 1]], v.dtype),
      lambda: tf.ones((1, 3), dtype=v.dtype))
  v = v * flip

  return u @ v


def shift_rot_scale_pose_to_std(
    pose, rot, scale,
    shift):
  """Puts the (pose, camera params) into the standard reference frame."""
  std_shift = pose[0]
  pose_cent, shift_std = inv_shift_weak_proj(pose, rot, scale, shift, std_shift)
  tostd_rot = get_std_rotation(pose_cent)
  pose_std, rot_std = inv_rot_weak_proj(pose_cent, rot, tostd_rot)
  # pose_std, rot_std = pose_cent, rot
  new_rescale = tf.linalg.norm(pose_std[8]) + 0.01
  # new_rescale = 1.0
  pose_std2, scale_std2 = inv_scale_weak_proj(pose_std, scale, new_rescale)
  return pose_std2, rot_std, scale_std2, shift_std


@tf.function
def convert_opt_record_to_task(
    data_rec, cam_permute_aug,
    use_bone_len, standardize_init_best
):
  """Generates a single training sample for `DeepInverseSolver`.

  Takes the output of `inference_time_optimization.run_inference_optimization`
  and produces flat `x_init`, `task_vec`, `x_best` with shapes sutable for
  `DeepInverseSolver` and (x_spec, task_spec) specifying how to "unflat" them
  using vmap._unpack_single_tensor.

  Arguments:
    data_rec: see `inference_time_optimization.run_inference_optimization`
    cam_permute_aug: bool, whether to permute camera order
    use_bone_len: bool, whether to add a bone lenths to `task_vec`
    standardize_init_best: bool, whether to standartize the init to standard RF

  Returns:
    x_init: flat initial solution (3D pose + cam params)
    task_vec: flat GMM parameters
    x_best: flat best solution found by the iterative solver
    cam_idx: the camera order
    (x_spec, task_spec): unpacking specs
  """
  init_pose, init_rot, init_scale, init_shift = [
      data_rec[x][0] for x in
      ['pose3d_opt_preds', 'cam_rot_opt_preds',
       'scale_opt_preds', 'shift_opt_preds']]

  best_pose, best_rot, best_scale, best_shift = [
      data_rec[x][-1] for x in
      ['pose3d_opt_preds', 'cam_rot_opt_preds',
       'scale_opt_preds', 'shift_opt_preds']]

  heatmap_mix_arr = data_rec['heatmaps']

  cam_idx = tf.range(tf.shape(init_rot)[0])

  if cam_permute_aug:
    to_cam_shuffle = [
        init_rot, init_scale, init_shift, best_rot,
        best_scale, best_shift, heatmap_mix_arr]

    cam_idx = tf.random.shuffle(cam_idx)
    shuffled = [tf.gather(x, cam_idx) for x in to_cam_shuffle]

    for x in to_cam_shuffle[1:]:
      tf.debugging.assert_equal(
          tf.shape(x)[0], tf.shape(to_cam_shuffle[0])[0],
          'some arguments to shuffled across cam dim have different # cams')

    (init_rot, init_scale, init_shift, best_rot,
     best_scale, best_shift, heatmap_mix_arr) = shuffled

  if standardize_init_best:
    init_pose, init_rot, init_scale, init_shift = shift_rot_scale_pose_to_std(
        init_pose, init_rot, init_scale, init_shift)
    best_pose, best_rot, best_scale, best_shift = shift_rot_scale_pose_to_std(
        best_pose, best_rot, best_scale, best_shift)

  init_rot_re, init_scale_re = inf_opt.reparam(init_rot, init_scale)
  best_rot_re, best_scale_re = inf_opt.reparam(best_rot, best_scale)

  x_init_tensors = [init_pose, init_rot_re, init_scale_re, init_shift]
  x_best_tensors = [best_pose, best_rot_re, best_scale_re, best_shift]
  task_vec_tensors = [heatmap_mix_arr]

  if use_bone_len:
    task_vec_tensors.append(normalized_bone_length(data_rec['pose3d']))

  x_init_tensors = [tf.cast(x, tf.float64) for x in x_init_tensors]
  x_best_tensors = [tf.cast(x, tf.float64) for x in x_best_tensors]

  x_init, x_spec = pack_tensors_unbatched(x_init_tensors)
  x_best, x_best_spec = pack_tensors_unbatched(x_best_tensors)
  task_vec, task_spec = pack_tensors_unbatched(task_vec_tensors)

  x_init, task_vec, x_best = [
      tf.cast(x, tf.float64) for x in [x_init, task_vec, x_best]]

  if _DEBUG_REPLACE_XINIT_WITH_NOISE.value:
    x_init = tf.random.normal(x_init.shape, dtype=tf.float64)

  assert x_best_spec == x_spec

  return x_init, task_vec, x_best, cam_idx, (x_spec, task_spec)


def get_pose_task_generator(
    opt_stats_ds, cam_permute_aug, use_bone_len,
    standardize_init_best
):
  """A dataset generator of training examples for `DeepInverseSolver`."""

  for _, opt_stats in opt_stats_ds:
    task_data_rec = convert_opt_record_to_task(
        opt_stats, cam_permute_aug, use_bone_len, standardize_init_best)
    x_init, task_vec, x_opt, cam_idx = task_data_rec[:4]
    inputs = {'x_init': x_init, 'task_vec': task_vec}
    rot_cam0_t = tf.transpose(opt_stats['cam_rot'][cam_idx[0]])
    outputs = {
        'x_opt': x_opt, 'pred': opt_stats['pose3d'],
        'pred_cam0': opt_stats['pose3d'] @ rot_cam0_t,
        'fwd': tf.gather(opt_stats['pose2d_repr'], cam_idx),
    }

    if use_bone_len:
      outputs['pred_bone_len'] = normalized_bone_length(opt_stats['pose3d'])

    yield (inputs, outputs)


def get_pose_task_dataset(
    opt_stats_ds,
    n_cam,
    n_joint,
    n_mix_comp,
    cam_permute_aug = False,
    use_bone_len = False,
    standardize_init_best = True
):
  """Produces a dataset of training examples for `DeepInverseSolver`."""

  x_shape = n_joint * 3 + n_cam * 3 * 2 + n_cam + n_cam * 3
  task_shape = n_cam * n_joint * n_mix_comp * 4

  if use_bone_len:
    task_shape += _BONE_COUNT.value

  output_types = ({'x_init': tf.float64, 'task_vec': tf.float64},
                  {'x_opt': tf.float64, 'pred': tf.float64,
                   'pred_cam0': tf.float64, 'fwd': tf.float64})

  output_shapes = ({'x_init': (x_shape,), 'task_vec': (task_shape,)},
                   {'x_opt': (x_shape,), 'pred': (n_joint, 3),
                    'pred_cam0': (n_joint, 3), 'fwd': (n_cam, n_joint, 2)})

  if use_bone_len:
    output_types[1]['pred_bone_len'] = tf.float64
    output_shapes[1]['pred_bone_len'] = (_BONE_COUNT.value,)

  generator = functools.partial(
      get_pose_task_generator,
      opt_stats_ds=opt_stats_ds,
      cam_permute_aug=cam_permute_aug,
      use_bone_len=use_bone_len,
      standardize_init_best=standardize_init_best)

  pose_task_ds = tf.data.Dataset.from_generator(
      generator, output_types, output_shapes)

  return pose_task_ds, (x_shape, task_shape)


def get_pose_forward_fn_batch(
    shapes_specs):
  """Returns batched 2D projection (`task_fwd_fn` for `DeepInverseSolver`)."""
  x_spec, task_spec = shapes_specs
  n_cam, n_joint = task_spec[0][:2]

  @tf.function
  def _forward_fn(cur_x, task_vec):
    del task_vec
    x_pose, x_rot, _, x_scale, _, x_shift = unpack_and_unparam(cur_x, x_spec)
    pose2d = inf_opt.project3d_weak(x_pose, x_rot, x_scale, x_shift)[:, :, :2]
    pose2d_flat = tf.reshape(pose2d, (n_cam * n_joint * 2,))
    return pose2d_flat

  pose_forward_batch = vmap.vmap(_forward_fn, ['cur_x', 'task_vec'])
  return pose_forward_batch


def get_logp_re_evaluate_batch(
    shapes_specs):
  """Returns batched GMM likelihood (`task_eval_fn` for `DeepInverseSolver`)."""
  _, task_spec = shapes_specs
  n_cam, n_joint = task_spec[0][:2]

  @tf.function
  def _logp_re_evaluate_single(pose2d_flat, task_vec):
    # pylint:disable=protected-access
    heatmaps_arr = vmap._unpack_single_tensor(task_vec, task_spec)[0]
    views_preds_2d_flat = tf.reshape(pose2d_flat, (n_cam * n_joint, 2))
    mv_heatmap_flat = tf.reshape(heatmaps_arr, (n_cam * n_joint, 4, 4))
    logp = inf_opt.gaussian_mixture_log_prob(
        views_preds_2d_flat, mv_heatmap_flat, 1e-8)
    neg_logp = -1 * tf.reduce_mean(logp)
    return neg_logp

  logp_re_evaluate_batch = vmap.vmap(
      _logp_re_evaluate_single, ['pose2d_flat', 'task_vec'])
  return logp_re_evaluate_batch


def get_bone_length_fn(
    shapes_specs):
  """Returns vectorized normalized bone lengths function."""
  x_spec = shapes_specs[0]

  @tf.function
  def _bone_length(cur_x):
    cur_x_pose = unpack_and_unparam(cur_x, x_spec)[0]
    return normalized_bone_length(cur_x_pose)

  bone_lenth_batch = vmap.vmap(_bone_length, ['cur_x'])
  return bone_lenth_batch


def get_pack_specs_from_opt_stats_ds(opt_stats_ds,
                                     use_bone_len):
  _, opt_stats = next(iter(opt_stats_ds))
  shapes_specs = convert_opt_record_to_task(
      opt_stats, cam_permute_aug=False, use_bone_len=use_bone_len,
      standardize_init_best=False)[-1]
  return shapes_specs


def sum_mse(x, y):
  return tf.reduce_sum(tfk.losses.mse(x, y))


def get_soln_optimizer_loss(
    weights,
    shapes_specs):
  """Returns a teacher-student loss bewteen solutions of IR and MetaPose."""
  def opt_loss_fn(cur_x, best_x):
    x_spec, _ = shapes_specs
    (cur_x_pose, cur_x_rot, cur_x_rot_re, cur_x_scale,
     cur_x_scale_re, cur_x_shift) = unpack_and_unparam(cur_x, x_spec)
    (best_x_pose, best_x_rot, best_x_rot_re, best_x_scale,
     best_x_scale_re, best_x_shift) = unpack_and_unparam(best_x, x_spec)

    idt = tf.eye(3, dtype=cur_x_rot.dtype)[None]
    # rot_inv_prod = tf.einsum('cij,cjk->cik', cur_x_rot, best_x_rot)
    # not sure why but the following is more correct (gives d(R, R) = 0)
    rot_inv_prod = tf.einsum('cij,ckj->cik', cur_x_rot, best_x_rot)

    losses = {
        'pose': sum_mse(cur_x_pose, best_x_pose),
        'scale': sum_mse(cur_x_scale, best_x_scale),
        'log_scale': sum_mse(cur_x_scale_re, best_x_scale_re),
        'shift': sum_mse(cur_x_shift, best_x_shift),
        'rot_3x3_mse': sum_mse(cur_x_rot, best_x_rot),
        'rot_6d_mse': sum_mse(cur_x_rot_re, best_x_rot_re),
        'rot_inv_mse': sum_mse(rot_inv_prod, idt),
        'rot_inv_trace': tf.reduce_sum(3 - tf.linalg.trace(rot_inv_prod))
    }
    loss_values = [v * weights[k] for k, v in losses.items()
                   if weights[k] is not None and weights[k] > 0.0]
    return tf.reduce_sum(loss_values)

  return vmap.vmap(opt_loss_fn, ['cur_x', 'best_x'])


def single_pmpjpe(gt_pose, pred_pose):
  aligned_poses = inf_opt.align_aba(pred_pose, gt_pose)[0]
  diff = aligned_poses - gt_pose
  return tf.reduce_mean(tf.linalg.norm(diff, axis=-1), axis=-1)


def get_pmpjpe_metric(
    shapes_specs):
  """Returns vectorized PMPJPE metric."""

  x_spec = shapes_specs[0]

  def pmpjpe(gt_x, cur_x):
    # gt_x: (17, 3), cur_x: (x_shape,)
    cur_x_pose = unpack_and_unparam(cur_x, x_spec)[0]
    return single_pmpjpe(gt_x, cur_x_pose)

  pmpjpe_batch = vmap.vmap(pmpjpe, ['gt_x', 'cur_x'])
  return pmpjpe_batch


def center_pose(pose3d):
  return pose3d - tf.reduce_mean(pose3d, axis=0, keepdims=True)


def single_npjpe_not_mean(gt_cam0_pose,
                          pred_cam0_pose,
                          allow_mirror = True):
  """Normalized PJPE optionally resolving "mirrored" solutions.

  Arguments:
    gt_cam0_pose: GT 3D pose in the zero's camera frame
    pred_cam0_pose: predicted 3D pose in the zero's camera frame
    allow_mirror: whether to fix "mirrored" solutions with negative depths

  Returns:
    Normalized per-joint error.
  """
  norm = tf.linalg.norm
  gt_x_cent = center_pose(gt_cam0_pose)
  cam0_rot_cent_pred = center_pose(pred_cam0_pose)
  scale_factor = norm(gt_x_cent) / norm(cam0_rot_cent_pred)
  cam0_rot_cent_scaled_pred = scale_factor * cam0_rot_cent_pred
  diff1 = gt_x_cent - cam0_rot_cent_scaled_pred
  err1 = norm(diff1, axis=-1)
  if not allow_mirror:
    return err1
  else:
    diff2 = gt_x_cent - cam0_rot_cent_scaled_pred * [[1.0, 1.0, -1.0]]
    err2 = norm(diff2, axis=-1)
    return tf.cond(tf.reduce_mean(err1) < tf.reduce_mean(err2),
                   lambda: err1, lambda: err2)


def get_nmpjpe_metric(
    shapes_specs,
    allow_mirror = True):
  """Vectorized Normalized PJPE metric."""
  x_spec = shapes_specs[0]

  def nmpjpe(gt_pose_cam0, cur_x):
    # gt_pose_cam0: (17, 3), cur_x: (x_shape,)

    x_pose, x_rot = unpack_and_unparam(cur_x, x_spec)[:2]
    cam0_rot_pred = x_pose @ x_rot[0]
    errs = single_npjpe_not_mean(gt_pose_cam0, cam0_rot_pred, allow_mirror)
    return tf.reduce_mean(errs)

  nmpjpe_batch = vmap.vmap(nmpjpe, ['gt_pose_cam0', 'cur_x'])
  return nmpjpe_batch


def get_percentage_of_correct_keypoints(
    shapes_specs,
    threshold = 150,
    allow_mirror = True):
  """Vectorized PCK metric."""
  x_spec = shapes_specs[0]
  def pck(gt_pose_cam0, cur_x):
    # gt_pose_cam0: (17, 3), cur_x: (x_shape,)
    x_pose, x_rot = unpack_and_unparam(cur_x, x_spec)[:2]
    cam0_rot_pred = x_pose @ x_rot[0]
    errs = single_npjpe_not_mean(gt_pose_cam0, cam0_rot_pred, allow_mirror)
    pck = tf.reduce_mean(tf.cast(errs < threshold, tf.float32)) * 100
    return pck

  pck_batch = vmap.vmap(pck, ['gt_pose_cam0', 'cur_x'])
  return pck_batch


def resh_mse(x_gt, fwd):
  fwd_re = tf.reshape(fwd, tf.shape(x_gt))
  return tfk.losses.mse(fwd_re, x_gt)


def print_single_view_stats(opt_stats_ds):
  """Prints pose estimation metrics of monocular 3D estimates."""
  def _map_pmpjpe_nmpjpe_pck(key, rec):
    del key
    output = []
    for i in range(_N_CAM.value):
      epi_cam_f32 = tf.cast(rec['pose3d_epi_pred'][i], tf.float64)
      rot = tf.cast(tf.transpose(rec['cam_rot'][i]), tf.float64)
      npje = single_npjpe_not_mean(
          rec['pose3d'] @ rot, epi_cam_f32, _DEBUG_ALLOW_MIRROR_IN_NMPJES.value)

      errs = [
          single_pmpjpe(rec['pose3d'], epi_cam_f32),
          tf.reduce_mean(npje),
          tf.reduce_mean(tf.cast(npje < 150, tf.float64)) * 100
      ]
      output.append(errs)
    per_cam_scores = [tf.reduce_mean(x) for x in zip(*output)]
    return per_cam_scores

  init_scores_ds = opt_stats_ds.map(_map_pmpjpe_nmpjpe_pck)
  score = tf.reduce_mean(list(init_scores_ds), axis=0)
  print('per-frame init pmpje/npjpe/pck: %.3f %.3f %.3f' % tuple(score.numpy()))


def get_datasets():
  """Get MetaPose training/validation/test datasets and original data shapes."""
  validate_on_eval = _VALID_FIRST_N.value < 0

  dataset_root = os.path.join(_DATA_ROOT.value, _DATASET.value)
  train_opt_stats_ds, eval_opt_stats_ds = [
      data_utils.read_tfrec_feature_dict_ds(os.path.join(dataset_root, split))
      for split in _DATA_SPLITS.value]

  if _DEBUG_SHOW_SINGLE_FRAME_PMPJES.value:
    for opt_stats_ds in [train_opt_stats_ds, eval_opt_stats_ds]:
      print_single_view_stats(opt_stats_ds)

  train_val_pose_task_ds, problem_shapes = get_pose_task_dataset(
      train_opt_stats_ds, _N_CAM.value, _N_JOINT.value, _N_MIX_COMP.value,
      cam_permute_aug=_PERMUTE_CAMS_AUG.value, use_bone_len=_USE_BONE_LEN.value,
      standardize_init_best=_STANDARDIZE_INIT_BEST.value)

  eval_pose_task_ds, _ = get_pose_task_dataset(
      eval_opt_stats_ds, _N_CAM.value, _N_JOINT.value, _N_MIX_COMP.value,
      cam_permute_aug=False, use_bone_len=_USE_BONE_LEN.value,
      standardize_init_best=_STANDARDIZE_INIT_BEST.value)

  if validate_on_eval:
    train_ds = (
        train_val_pose_task_ds
        .batch(_TRAIN_BATCH_SIZE.value)
        .take(_DEBUG_TAKE_N_TRAIN_BATCHES.value))
    eval_ds = (
        eval_pose_task_ds
        .batch(_EVAL_BATCH_SIZE.value)
        .take(_DEBUG_TAKE_N_EVAL_BATCHES.value))
    valid_ds = eval_ds
  else:
    train_ds = (
        train_val_pose_task_ds
        .skip(_VALID_FIRST_N.value)
        .repeat(_TRAIN_REPEAT_K.value)
        .shuffle(_SHUFFLE_SIZE.value, seed=0, reshuffle_each_iteration=True)
        .batch(_TRAIN_BATCH_SIZE.value)
        .take(_DEBUG_TAKE_N_TRAIN_BATCHES.value))
    valid_ds = (
        train_val_pose_task_ds
        .take(_VALID_FIRST_N.value)
        .repeat(_TRAIN_REPEAT_K.value)
        .batch(_EVAL_BATCH_SIZE.value)
        .take(_DEBUG_TAKE_N_EVAL_BATCHES.value))
    eval_ds = (
        eval_pose_task_ds
        .batch(_EVAL_BATCH_SIZE.value)
        .take(_DEBUG_TAKE_N_EVAL_BATCHES.value))

  if _DEBUG_CACHE_DATASET.value:
    train_ds, valid_ds, eval_ds = [
        x.cache() for x in [train_ds, valid_ds, eval_ds]]

  datasets = {'eval': eval_ds, 'train': train_ds, 'valid': valid_ds}
  pack_specs = get_pack_specs_from_opt_stats_ds(
      eval_opt_stats_ds, use_bone_len=_USE_BONE_LEN.value)
  return datasets, problem_shapes, pack_specs


def get_losses_and_metrics(
    datasets, pack_specs,
    soln_loss_weights
):
  """Get all metrics/losses. Optimally compute them for the initial guess."""
  # pylint:disable=g-long-lambda
  logp_re_evaluate_batch = get_logp_re_evaluate_batch(pack_specs)
  pose_forward_batch = get_pose_forward_fn_batch(pack_specs)
  soln_opt_loss_fn = get_soln_optimizer_loss(soln_loss_weights, pack_specs)
  pmpjpe_metric = get_pmpjpe_metric(pack_specs)
  nmpjpe_metric = get_nmpjpe_metric(
      pack_specs, allow_mirror=_DEBUG_ALLOW_MIRROR_IN_NMPJES.value)
  pck_metric = get_percentage_of_correct_keypoints(
      pack_specs, allow_mirror=_DEBUG_ALLOW_MIRROR_IN_NMPJES.value)
  bone_len_fn = get_bone_length_fn(pack_specs)

  funcs = {
      'logp_re_evaluate': logp_re_evaluate_batch,
      'pose_forward': pose_forward_batch,
      'soln_opt_loss': soln_opt_loss_fn,
      'pmpjpe': pmpjpe_metric,
      'nmpjpe': nmpjpe_metric,
      'pck': pck_metric,
      'bone_len': bone_len_fn
  }

  if not _DATASET_WARMUP.value:
    return funcs

  # and we also need this to warm up the ds cache

  # x['x_init']:       epipolar init for 3d pose + cameras
  # x['task_vec']:     heatmaps [+ bone length]

  # y['x_opt']:        sgd optimal for 3d pose + cameras
  # y['fwd']:          ground truth 2d projections
  # y['pred']:         ground truth 3d pose
  # y['pred_cam0']:    ground truth 3d pose rotated to camera 0

  warmup_funcs = {
      'initial task loss': lambda x, y: logp_re_evaluate_batch(
          pose_forward_batch(x['x_init'], x['task_vec']), x['task_vec']),
      'optimal task loss': lambda x, y: logp_re_evaluate_batch(
          pose_forward_batch(y['x_opt'], x['task_vec']), x['task_vec']),
      'gt task loss': lambda x, y: logp_re_evaluate_batch(
          y['fwd'], x['task_vec']),
      'initial pmpjpe': lambda x, y: pmpjpe_metric(y['pred'], x['x_init']),
      'optimal pmpjpe': lambda x, y: pmpjpe_metric(y['pred'], y['x_opt']),
      'initial nmpjpe': lambda x, y: nmpjpe_metric(y['pred_cam0'], x['x_init']),
      'optimal nmpjpe': lambda x, y: nmpjpe_metric(y['pred_cam0'], y['x_opt']),
      'initial pck': lambda x, y: pck_metric(y['pred_cam0'], x['x_init']),
      'optimal pck': lambda x, y: pck_metric(y['pred_cam0'], y['x_opt']),
      'initial mse 2d': lambda x, y: resh_mse(
          y['fwd'], pose_forward_batch(x['x_init'], x['task_vec'])),
      'optimal mse 2d': lambda x, y: resh_mse(
          y['fwd'], pose_forward_batch(y['x_opt'], x['task_vec']))
  }

  if _USE_BONE_LEN.value:
    warmup_funcs.update({
        'initial normalized bone len loss': lambda x, y: resh_mse(
            y['pred_bone_len'], bone_len_fn(x['x_init'])),
        'optimal normalized bone len loss': lambda x, y: resh_mse(
            y['pred_bone_len'], bone_len_fn(y['x_opt']))
    })

  for split, ds in datasets.items():
    print(split, 'len', len(list(ds)))
    for name, fn in warmup_funcs.items():
      print(split, name, '%.4f' % tf.reduce_mean(list(ds.map(fn).unbatch())))

  return funcs


def get_callbacks(log_dir,
                  datasets,
                  stage_i,
                  best_val_score):
  """Get all MetaPose callbacks for the stage i."""
  # pylint:disable=protected-access

  tb_cb = tfk.callbacks.TensorBoard(log_dir)
  # to disable very slow evaluation_*_vs_iteration reporting
  tb_cb.on_test_end = lambda *argv, **kwargs: None

  ds_writer_getters = {
      # not created yet, so passing getters instead of actual writers
      'train': (lambda: tb_cb._train_writer, datasets['train']),
      'validation': (lambda: tb_cb._val_writer, datasets['valid']),
  }

  if _VALID_FIRST_N.value < 0:
    # valid = eval, no further actions needed
    eval_cbs = []
  else:
    eval_cb = EvaluteOnDatasetCallback(
        log_dir, datasets['eval'], ds_name='test')
    ds_writer_getters['test'] = (lambda: eval_cb.writer, datasets['eval'])
    eval_cbs = [eval_cb]

  cbs = [
      TensorflowSaveWeightsFix(),
      *eval_cbs,
      tfk.callbacks.ModelCheckpoint(
          '/tmp/best-model', monitor='val_pred_pmpjpe',
          save_best_only=True, save_weights_only=True,
          verbose=True),
      tfk.callbacks.EarlyStopping(
          monitor='val_pred_pmpjpe',
          patience=_EARLY_STOPPING_PATIENCE.value,
          verbose=True),
  ]

  cbs.extend([
      # this order to ensure summary writers are not closed
      WriteStageMetrics(ds_writer_getters, stage_i,
                        'val_pred_pmpjpe', best_val_score),
      tb_cb
  ])

  return cbs


def get_simple_stage_model(mlp_sizes, x_shape,
                           task_shape, fwd_shape):
  input_dim = x_shape + task_shape + fwd_shape + 2
  model = tfk.Sequential([
      tfkl.Dense(mlp_sizes[0], _ACTIVATION_FN.value, input_shape=(input_dim,)),
      *[tfkl.Dense(k, _ACTIVATION_FN.value) for k in mlp_sizes[1:]],
      tfkl.Dense(x_shape)
  ])
  return model


def _unpack_tensor_batch(batch, spec):
  # pylint:disable=protected-access
  return tf.vectorized_map(
      lambda x: vmap._unpack_single_tensor(x, spec), batch)

# % ---


class ContextNormalization(tfk.Model):
  """A permutation-eqvivariant layer standardizing feature maps.

  See "Learning to Find Good Correspondences" by Yi et al.
  """

  def call(self, inputs):
    tf.assert_rank(inputs, 3)
    average = tf.reduce_mean(inputs, axis=1)[:, None, :]
    std = tf.math.reduce_std(inputs, axis=1)[:, None, :]
    normalized = (inputs - average) / (std + 1e-10)
    return normalized


class ContextConcatenation(tfk.Model):
  """A permutation-eqvivariant layer concatenating moments to feature maps."""

  def call(self, inputs):
    tf.assert_rank(inputs, 3)
    ctx_size = tf.shape(inputs)[1]
    average = tf.reduce_mean(inputs, axis=1)[:, None, :]
    std = tf.math.reduce_std(inputs, axis=1)[:, None, :]
    average, std = [tf.repeat(x, ctx_size, axis=1) for x in [average, std]]
    concated = tf.concat([inputs, average, std], axis=2)
    return concated

EquivariantMLPSpec = Sequence[Union[int, Literal['cn'], Literal['ccat']]]


def EquivariantMLP(input_shape,  # pylint: disable=invalid-name
                   output_dim,
                   mlp_spec,
                   activation):
  """Produces a permutation-eqvivariant MLP."""
  assert isinstance(mlp_spec[0], int)
  layers = [tfkl.Dense(mlp_spec[0], activation, input_shape=input_shape)]
  for size_or_cn in mlp_spec:
    if size_or_cn == 'cn':
      layers.append(ContextNormalization())
    elif size_or_cn == 'ccat':
      layers.append(ContextConcatenation())
    elif isinstance(size_or_cn, int):
      layers.append(tfkl.Dense(size_or_cn, _ACTIVATION_FN.value))
    else:
      raise ValueError('unsupported eqvivar mlp spec %s' % mlp_spec)

  layers.append(tfkl.Dense(output_dim))
  return tfk.Sequential(layers)


class InvariantStageModel(tfk.Model):
  """A permutation-equivariant model compatible with `DeepInverseSolver`."""

  def __init__(self, main_mlp_spec, pose_emb_size,
               pose_mlp_sizes, pack_specs,
               **kwargs):
    super().__init__(**kwargs)
    self.main_equiv_mlp_spec = main_mlp_spec
    self.pack_specs = pack_specs
    self.pose_emb_size = pose_emb_size

    n_cam, n_joint, n_comp = pack_specs[1][0][:3]
    fwd_shape = n_cam * n_joint * 2
    x_shape = n_joint*3 + n_cam*(6 + 1 + 3)
    task_shape = n_cam * n_joint * n_comp * 4

    if _USE_BONE_LEN.value:
      task_shape += _BONE_COUNT.value
      used_bone_count = _BONE_COUNT.value
    else:
      used_bone_count = 0

    self.x_shape = x_shape
    self.task_shape = task_shape
    self.fwd_shape = fwd_shape

    mlp_input_data_dim = 0
    input_data_parts_dims = {
        'pose': n_joint * 3,
        'cams': 10,  # 6 + 3 + 1
        'heatmaps': n_joint * n_comp * 4,
        'projs': n_joint * 2,
        'loss': 2
    }

    drop_mlp_inputs = _DEBUG_DROP_MLP_INPUTS.value
    if any(x not in input_data_parts_dims for x in drop_mlp_inputs):
      raise ValueError('unknown droppable %s' % drop_mlp_inputs)

    for name, input_part_size in input_data_parts_dims.items():
      if name not in drop_mlp_inputs:
        mlp_input_data_dim += input_part_size

    if _USE_BONE_LEN.value:
      mlp_input_data_dim += used_bone_count

    assert mlp_input_data_dim > 0
    main_mlp_input_shape = (n_cam, mlp_input_data_dim)
    main_mlp_output_dim = pose_emb_size + (6 + 1 + 3)

    print('Eqvivariant spec: ',
          [main_mlp_spec, pose_emb_size, pose_mlp_sizes, _ACTIVATION_FN.value,
           main_mlp_input_shape, main_mlp_output_dim,
           drop_mlp_inputs])

    # can't change the name or checkpoints will break
    self.main_mlp = EquivariantMLP(
        main_mlp_input_shape, main_mlp_output_dim,
        main_mlp_spec, _ACTIVATION_FN.value)

    self.pose_mlp = tfk.Sequential([
        tfkl.Dense(pose_mlp_sizes[0], _ACTIVATION_FN.value,
                   input_shape=(pose_emb_size,)),
        *[tfkl.Dense(k, _ACTIVATION_FN.value) for k in pose_mlp_sizes[1:]],
        tfkl.Dense(n_joint*3)
    ])

    print('Model parameter counts: ',
          self.main_mlp.count_params(), self.pose_mlp.count_params())

  def call(self, inputs):
    # inputs: [B, x_shape + task_shape + fwd_shape + 2]

    x_spec, task_spec = self.pack_specs
    n_cam, n_joint = task_spec[0][0], task_spec[0][1]
    batch_size = tf.shape(inputs)[0]
    cur_x, task_vec, cur_fwd, loss_stage = tf.split(
        inputs, [self.x_shape, self.task_shape, self.fwd_shape, 2], axis=1)

    # [B, x_shape] = [B, J*3 + ...]
    # [B, J, 3], [B, C, 2, 3], [B, C], [B, C, 3]
    x_pose, x_rot_re, x_scale_re, x_shift = _unpack_tensor_batch(cur_x, x_spec)
    # [B, C, J, K, 4]
    heatmaps_arr = _unpack_tensor_batch(task_vec, task_spec)[0]
    # [B, C, J, 2]
    cur_fwd_resh = tf.reshape(cur_fwd, (batch_size, n_cam, n_joint, 2))

    mlp_inputs = {
        'invariant': {
            'pose': [x_pose],
            'loss': [loss_stage]
        },
        'eqvivariant': {
            'cams': [x_rot_re, x_scale_re, x_shift],
            'heatmaps': [heatmaps_arr],
            'projs': [cur_fwd_resh]
        }
    }

    mlp_inputs_filtered = {k: [] for k in mlp_inputs}
    for kind, kind_input_dict in mlp_inputs.items():
      for input_name, input_arrs in kind_input_dict.items():
        if input_name not in _DEBUG_DROP_MLP_INPUTS.value:
          mlp_inputs_filtered[kind].extend(input_arrs)

    if _USE_BONE_LEN.value:
      bone_len_arr = _unpack_tensor_batch(task_vec, task_spec)[1]
      mlp_inputs_filtered['invariant'].append(bone_len_arr)  # [B, S]

    glob_vecs = mlp_inputs_filtered['invariant']
    cam_vecs = mlp_inputs_filtered['eqvivariant']

    for cam_dep_vec in cam_vecs:
      tf.assert_equal(tf.shape(cam_dep_vec)[1], n_cam)

    # [B, J*3 + 2 + S]
    # glob_concated = tf.concat(
    #     [tf.reshape(x, (batch_size, -1)) for x in glob_vecs], axis=1)

    # [B, C, J*3 + 2 + S]
    # glob_concated_repeated = tf.repeat(glob_concated[:, None], n_cam, axis=1)

    # # [B, C, 6 + 1 + 3 + J*K*4 + J*2]
    # cam_vecs_concated = tf.concat([
    #     tf.reshape(x, (batch_size, n_cam, -1)) for x in cam_vecs], axis=2)

    # [[B, C, J*3], [B, C, 2], [B, C, S]]
    # where S = _BONE_COUNT.value or zero
    glob_flat_repeated = [
        tf.repeat(tf.reshape(x, (batch_size, 1, -1)), n_cam, axis=1)
        for x in glob_vecs
    ]

    # [B, C, 6 + 1 + 3], [B, C, J*K*4], [B, C, J*2]]
    cam_vecs_flat = [tf.reshape(x, (batch_size, n_cam, -1)) for x in cam_vecs]

    # [B, C, J*3 + 2 + S + 6 + 1 + 3 + J*K*4 + J*2]
    mlp_input = tf.concat([*glob_flat_repeated, *cam_vecs_flat], axis=2)

    # [B, C, pose_emb_size + 6 + 1 + 3]
    mlp_output = self.main_mlp(mlp_input)
    # [B, C, pose_emb_size], [B, C, 6 + 1 + 3]
    x_pose_emb_per_cam, x_cur_rest = tf.split(
        mlp_output, [self.pose_emb_size, 10], axis=2)
    # [B, C, 6], [B, C, 1], [B, C, 3]
    d_x_cams = tf.split(x_cur_rest, [6, 1, 3], axis=2)
    # [B, C, pose_emb_size]
    x_pose_emb_per_cam_avg = tf.reduce_mean(x_pose_emb_per_cam, axis=1)
    # [B, n_joint*3]
    d_x_pose = self.pose_mlp(x_pose_emb_per_cam_avg)

    final_to_concat = [d_x_pose, *d_x_cams]
    final_pred = tf.concat([
        tf.reshape(x, (batch_size, -1)) for x in final_to_concat], axis=1)

    return final_pred


def get_fresh_stage_model(pack_specs,
                          model_mlp_sizes,
                          main_mlp_spec, pose_emb_size,
                          pose_mlp_sizes):
  """Get a permutation-equivariant (or a non-equivariant) stage model."""

  if _USE_EQUIVARIANT_MODEL.value:
    return InvariantStageModel(
        main_mlp_spec=main_mlp_spec,
        pose_emb_size=pose_emb_size,
        pose_mlp_sizes=pose_mlp_sizes,
        pack_specs=pack_specs)
  else:
    n_cam, n_joint, n_comp = pack_specs[1][0][:3]
    fwd_shape = n_cam * n_joint * 2
    x_shape = n_joint*3 + n_cam*(6 + 1 + 3)
    task_shape = n_cam * n_joint * n_comp * 4

    if _USE_BONE_LEN.value:
      task_shape += _BONE_COUNT.value

    return get_simple_stage_model(
        model_mlp_sizes, x_shape, task_shape, fwd_shape)


def main(_):
  for mod_name, mod_flags in flags.FLAGS.flags_by_module_dict().items():
    if 'metapose' in mod_name:
      for flag in mod_flags:
        print(flag.name, '=', flag.value, type(flag.value))

  if _DEBUG_DROP_MLP_INPUTS.value and not _USE_EQUIVARIANT_MODEL.value:
    raise NotImplementedError('Can not drop inputs for non-eqvivariant model.')

  if _DEBUG_ENABLE_CHECK_NUMERICS.value:
    tf.debugging.enable_check_numerics()

  soln_loss_flag_names = [
      'pose', 'scale', 'log_scale', 'shift', 'rot_3x3_mse',
      'rot_6d_mse', 'rot_inv_mse', 'rot_inv_trace']

  soln_loss_weights = {
      name: getattr(flags.FLAGS, 'lambda_soln_' + name)
      for name in soln_loss_flag_names}

  loss_weights = {
      'x_opt': _LAMBDA_XOPT_LOSS.value,
      'fwd': _LAMBDA_FWD_LOSS.value
  }
  custom_loss_weights = {'logp': _LAMBDA_LOGP_LOSS.value}

  model_mlp_sizes = _MODEL_MLP_SIZES.value
  max_n_stages = _MAX_N_STAGES.value
  epochs_per_stage = _EPOCHS_PER_STAGE.value
  learning_rate = _LEARNING_RATE.value

  model_mlp_sizes = list(map(int, _MODEL_MLP_SIZES.value))
  pose_mlp_sizes = list(map(int, _POSE_MLP_SIZES.value))
  main_mlp_spec = [int(x) if x.isnumeric() else x for x in _MAIN_MLP_SPEC.value]

  datasets, _, pack_specs = get_datasets()
  funcs = get_losses_and_metrics(datasets, pack_specs, soln_loss_weights)

  log_dir = _TB_LOG_DIR.value

  models = []
  best_performance = [np.inf]

  custom_output_fns = {
      'pred': lambda cur_xs, task_vecs: cur_xs,
      'pred_cam0': lambda cur_xs, task_vecs: cur_xs,
      'pred_bone_len': lambda cur_xs, task_vecs: funcs['bone_len'](cur_xs)
  }

  losses = {'x_opt': funcs['soln_opt_loss'], 'fwd': resh_mse}
  metrics = {
      'pred': funcs['pmpjpe'],
      'pred_cam0': [funcs['nmpjpe'], funcs['pck']]
  }

  if _USE_BONE_LEN.value:
    losses['pred_bone_len'] = tfk.losses.mse
    loss_weights['pred_bone_len'] = _LAMBDA_LIMB_LEN_LOSS.value

  print('======== STARTING ========')
  print(log_dir)
  print(model_mlp_sizes, max_n_stages, epochs_per_stage, learning_rate,
        loss_weights, custom_loss_weights)

  for attempt_i in range(_MAX_STAGE_ATTEMPTS.value):
    stage_i = len(models)
    if stage_i > max_n_stages:
      break

    if attempt_i == 0 and _LOAD_WEIGHTS_FROM.value:
      prepend_stages = _LOAD_STAGES_N.value
    else:
      prepend_stages = 1

    for _ in range(prepend_stages):
      models.append(get_fresh_stage_model(
          pack_specs, model_mlp_sizes, main_mlp_spec,
          _POSE_EMB_SIZE.value, pose_mlp_sizes))

    solver = DeepInverseSolver(
        models, funcs['pose_forward'], funcs['logp_re_evaluate'],
        task_loss_name='logp', custom_output_fns=custom_output_fns)

    solver.compile(
        optimizer=tfk.optimizers.Adam(learning_rate),
        loss=losses,
        metrics=metrics,
        loss_weights=loss_weights,
        custom_loss_weights=custom_loss_weights,
        run_eagerly=_DEBUG_RUN_EAGERLY.value
    )

    if attempt_i == 0 and _LOAD_WEIGHTS_FROM.value:
      print('restoring from %s' % _LOAD_WEIGHTS_FROM.value)
      solver.load_weights(_LOAD_WEIGHTS_FROM.value)
      # following line does not work because of solver save bug
      # result.assert_existing_objects_matched()

      # have to do it twice because of the bug in keras
      for _ in range(2):
        for key, ds in [('eval', datasets['eval'])]:
          print('post-restore %s' % key, solver.evaluate(ds, return_dict=True))

    if not _USE_EQUIVARIANT_MODEL.value:
      print(models[-1].summary())

    if _EPOCHS_PER_STAGE.value > 0:
      cbs = get_callbacks(log_dir, datasets, stage_i, best_performance[-1])

      print('= STAGE %d =' % stage_i)
      solver.fit(datasets['train'],
                 validation_data=datasets['valid'],
                 initial_epoch=stage_i*epochs_per_stage,
                 epochs=(stage_i+1)*epochs_per_stage,
                 callbacks=cbs,
                 verbose=2)

    valid_metrics = dict(zip(
        solver.metrics_names, solver.evaluate(datasets['valid'])))
    cur_performance = valid_metrics['pred_pmpjpe']
    print('%.3f' % cur_performance,
          ' '.join('%.3f' % x for x in best_performance))

    if _EPOCHS_PER_STAGE.value > 0:
      if cur_performance > best_performance[-1]:
        print('retraining last because %.3f > %.3f'
              % (cur_performance, best_performance[-1]))
        models = models[:-1]
      else:
        print('keeping last because %.3f < %.3f'
              % (cur_performance, best_performance[-1]))
        best_performance.append(cur_performance)
        solver.save_weights(log_dir + '/model')

  if _SAVE_PREDS_TO.value:
    n_cam = pack_specs[0][1][0].numpy()
    eval_preds = solver.predict(datasets['eval'])
    eval_preds_ds = tf.data.Dataset.from_tensor_slices(eval_preds)
    pred_names = ['pose3d', 'rot', 'rot_re', 'scale', 'scale_re', 'shift']

    unpack = lambda x: unpack_and_unparam(x, pack_specs[0])
    eval_preds_ds = eval_preds_ds.map(
        lambda d: {**d, **dict(zip(pred_names, unpack(d['x_opt'])))})

    eval_preds_ds = eval_preds_ds.map(
        lambda d: {**d, 'pose2d': tf.reshape(d['fwd'], (n_cam, 17, 2))})

    x_pred_y_ds = tf.data.Dataset.zip((
        eval_preds_ds, datasets['eval'].map(lambda x, y: y).unbatch()
    ))
    pmpjes_ds = x_pred_y_ds.map(
        lambda dx, dy: single_pmpjpe(dy['pred'], dx['pose3d']))
    print(tf.reduce_mean(list(pmpjes_ds)))

    eval_preds_ds = (
        tf.data.Dataset.zip((eval_preds_ds, pmpjes_ds))
        .map(lambda d, v: {**d, 'pmpje': v}))

    print({k: v.shape for k, v in next(iter(eval_preds_ds)).items()})
    tf.data.experimental.save(eval_preds_ds, _SAVE_PREDS_TO.value)

if __name__ == '__main__':
  app.run(main)

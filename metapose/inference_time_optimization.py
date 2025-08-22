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

"""Utilities for performing inference time optimization."""
# pylint: disable=invalid-name

import inspect
from typing import Sequence, Tuple, Callable, Mapping, Optional, Any  # pylint: disable=g-importing-member,g-multiple-import

import tensorflow as tf
import tensorflow_probability as tfp


def procrustes(a,
               b):
  """Computes rotation that aligns mean-centered `a` to mean-centered `b`."""
  tf.debugging.assert_shapes([
      (a, ('joints', 'dim')),
      (b, ('joints', 'dim'))
  ])
  a_m = tf.reduce_mean(a, axis=0)
  b_m = tf.reduce_mean(b, axis=0)
  a_c = a - a_m
  b_c = b - b_m
  A = tf.tensordot(a_c, b_c, axes=(0, 0))
  _, U, V = tf.linalg.svd(A)
  R = tf.tensordot(U, V, axes=(1, 1))
  return R, a_m, b_m


def align_aba(
    a,
    b,
    rescale = True
):
  """Produces `a` after optimal alignment to `b` and vice-versa."""
  R, a_m, b_m = procrustes(a, b)
  scale = tf.reduce_mean((tf.linalg.norm(b - b_m)) / tf.linalg.norm(a - a_m))
  scale = scale if rescale else 1.0
  a2b = (a - a_m) @ R * scale + b_m
  b2a = (b - b_m) @ tf.transpose(R) / scale + a_m
  return a2b, b2a, (a_m, b_m, R, scale)


# [batch, 6] -> [batch, 3, 3]
# see “On the Continuity of Rotation Representations in Neural Networks”
# by Zhou et al. 2019
def vec6d_to_rot_mat(vec):
  """Converts a batch of 6D parameterized rots to rotation matrices."""
  x, y = vec[:, :3], vec[:, 3:]
  xn = tf.linalg.normalize(x, axis=-1)[0]
  z = tf.linalg.cross(xn, y)
  zn = tf.linalg.normalize(z, axis=-1)[0]
  yn = tf.linalg.cross(zn, xn)
  mat = tf.stack([xn, yn, zn], axis=1)
  return mat


def minimize_step_fn(func,
                     opt,
                     opt_arg_names,
                     args, kwargs
                     ):
  """Minimizes the scalar-valued `func` using `opt` wrt `opt_arg_names`.

  Produces a step_fn function that when called performs a single step of
  minimizes a function `func`

  Arguments:
    func: a scalar-valued callable
    opt: an instance of a tf.keras.optimizers.Optimizer
    opt_arg_names: names of arguments to optimize with respect to
    args: function argument list
    kwargs: function argument dict

  Returns:
    A dict of variables corresponding to opt_arg_names.

  The func must be such that `func(*args, **kwargs)` succeeds.
  For names in in `opt_arg_names` corresponding arguments are used as
  starting values.

  Example:

    def func(a, b, c): return a + b * c
    opt = tf.keras.optimizers.Adam(...)
    step_fn, var_dict = minimize_step_fn(
      func, opt, ['b'], [1.0, 3.0], {'c': 5.0})

    step_fn()  # evaluates d func(a, b, c) / d b at (1.0, 3.0, 5.0)
    print(var_dict['b'])  # updated value
  """
  full_kwargs = inspect.signature(func).bind(*args, **kwargs).arguments
  extra_keys = set(opt_arg_names) - full_kwargs.keys()
  if extra_keys:
    raise ValueError('args %s are not in the original func' % extra_keys)
  var_dict = {n: tf.Variable(full_kwargs[n], name=n) for n in opt_arg_names}
  full_kwargs.update(var_dict)
  loss_fn = lambda: func(**full_kwargs)  # pylint: disable=unnecessary-lambda
  def step_fn():
    pre_loss = loss_fn()
    opt.minimize(loss_fn, list(var_dict.values()))
    return pre_loss

  return step_fn, var_dict


def reparam(R, scale):
  """Reparameterize rotation (as 6D) and scale (as inv_softplus(scale))."""
  R_reparam = R[:, :2, :]
  scales_reparam = tf.math.log(tf.math.exp(scale) - 1.0)  # inv softmax
  return R_reparam, scales_reparam


def unreparam(R_re,
              scale_re):
  """Un-reparameterize rotation (as 6D) and scale (as inv_softplus(scale))."""
  R = vec6d_to_rot_mat(tf.reshape(R_re, (-1, 6)))
  scale = tf.math.softplus(scale_re)
  return R, scale


def initial_epi_estimate(
    multi_view_pose3d_preds
):
  """(Stage 1) Estimate initial pose and cameras from per-view monocular 3D.

  Assumes that the zero's camera frame is canonical (R=I, scale=1, shift=(0,0)).
  Procrustes aligns each pose to the pose in zero's camera frame.

  Arguments:
    multi_view_pose3d_preds: (n_cam, n_joints, n_dim) monocular

  Returns:
    mean_pose3d_centered: (n_joints, n_dim) initial pose
    R: (n_cam, 3, 3) initial camera rots
    scale: (n_cam, ) initial camera scales
    shift: (n_cam, 2) initial camera shifts
  """
  all_aligned_preds = []
  params = []
  for view_id in tf.range(1, tf.shape(multi_view_pose3d_preds)[0]):
    pred = multi_view_pose3d_preds[view_id]
    _, aligned_pred, view_params = align_aba(multi_view_pose3d_preds[0], pred)
    all_aligned_preds.append(aligned_pred)
    params.append(view_params)

  view0_mean = params[0][0]
  mean_pose3d_centered = tf.reduce_mean(all_aligned_preds, axis=0) - view0_mean
  first_view_params = [tf.zeros(3), view0_mean, tf.eye(3), 1.0]
  all_view_params = [first_view_params] + params
  _, shift, R, scale = map(tf.stack, zip(*all_view_params))
  return mean_pose3d_centered, (R, scale, shift)


def project3d_weak(pose_pred, R, scale,
                   shift):
  """Performs true weak projection of poses using given camera params."""
  tf.debugging.assert_shapes([
      (pose_pred, ('joints', 3)),
      (shift, ('cams', 3)),
      (R, ('cams', 3, 3)),
      (scale, ('cams',)),
  ])
  rot_views = tf.einsum('jd,kdo->kjo', pose_pred, R)  # (k=4, j=17, d=o=3)
  back_rot_preds = rot_views * scale[:, None, None] + shift[:, None, :]
  # > (4, 17, 3)
  return back_rot_preds


# [batch, 2], [batch, n_comp, 4] -> [batch, ]
def gaussian_mixture_log_prob(points,
                              params,
                              eps):
  """Computes the likelihood of `points` given GMM `params`."""
  tf.debugging.assert_shapes([
      (points, ('batch_size', 2)),
      (params, ('batch_size', 'n_comp', 4)),  # [comp_weight, mu_x, mu_y, cov]
      (eps, ())
  ])
  mix_probs = params[:, :, 0]  # [b, c]
  mu_s = params[:, :, 1:3]     # [b, c, 2]
  covs = tf.sqrt(params[:, :, 3])  # [b, c]
  diag_s = tf.repeat(covs[:, :, None], repeats=2, axis=2)  # [b, c, 2]
  norm = tfp.distributions.MultivariateNormalDiag(loc=mu_s, scale_diag=diag_s)
  test_points = tf.repeat(points[:, None, :], params.shape[1], 1)
  gauss_log_probs = norm.log_prob(tf.cast(test_points, tf.float64))    # [b, c]
  log_gauss_plus_mix = gauss_log_probs + tf.math.log(mix_probs + eps)  # [b, c]
  final_log_probs = tf.reduce_logsumexp(log_gauss_plus_mix, axis=1)    # [b, ]
  return tf.cast(final_log_probs, tf.float32)


def total_frame_loss(pose_pred,
                     R,
                     scale,
                     shift,
                     mv_heatmaps_mixture):
  """The objective optimized by the full probabilistic iterative solver."""
  tf.debugging.assert_shapes([
      (pose_pred, ('joints', 3)),
      (shift, ('cams', 3)),
      (R, ('cams', 3, 3)),
      (scale, ('cams',)),
      (mv_heatmaps_mixture, ('cams', 'joints', 4, 4))
  ])
  n_cam = mv_heatmaps_mixture.shape[0]
  n_joint = pose_pred.shape[0]

  views_preds = project3d_weak(pose_pred, R, scale, shift)
  views_preds_2d_flat = tf.reshape(views_preds[:, :, :2], (n_cam * n_joint, 2))
  mv_heatmap_flat = tf.reshape(mv_heatmaps_mixture, (n_cam * n_joint, 4, 4))
  logp = gaussian_mixture_log_prob(views_preds_2d_flat, mv_heatmap_flat, 1e-8)
  return -1 * tf.reduce_mean(logp)


h36m_edges = tf.convert_to_tensor(
    [[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [8, 14], [14, 15],
     [15, 16], [11, 12], [12, 13], [0, 4], [0, 1], [1, 2], [2, 3],
     [4, 5], [5, 6]])


def get_h36m_edge_lens(pose3d):
  """Computes an array of bone lenghts."""
  flat_joints = tf.gather(pose3d, tf.reshape(h36m_edges, (-1,)))
  edge_coords = tf.reshape(flat_joints, (-1, 2, 3))
  edge_vecs = edge_coords[:, 0, :] - edge_coords[:, 1, :]
  return tf.linalg.norm(edge_vecs, axis=-1)


def get_edge_len_loss(gt_edge_lens,
                      pose_pred):
  """Computes the scale-invariant bone length distance."""
  pred_edge_lens = get_h36m_edge_lens(pose_pred)
  norm_gt_edge_lens = gt_edge_lens / tf.reduce_mean(gt_edge_lens)
  norm_pred_edge_lens = pred_edge_lens / tf.reduce_mean(pred_edge_lens)
  len_err = tf.linalg.norm(norm_gt_edge_lens - norm_pred_edge_lens)
  return len_err


def optimize_heatmap_logp(init_pose,
                          init_params,
                          heatmap_mix_arr,
                          gt_edge_lens,
                          edge_lens_lambda,
                          opt_steps,
                          report_n_results,
                          opt
                         ):
  """Performs the full probabilistic iterative bundle adjustment.

  Arguments:
    init_pose: (n_joints, n_dim) initial human pose
    init_params: a list or tuple of tensors [R, scale, shift]
      as returned by `initial_epi_estimate`
    heatmap_mix_arr: (n_cam, n_comp, 4) a tensor of GMM parameters
    gt_edge_lens: (n_bones, ) the lenghts of all bones
    edge_lens_lambda: the weight of the bone length loss
    opt_steps: how many GD steps to take
    report_n_results: how many steps to report
    opt: an instance of a tf.keras.optimizers.Optimizer

  Returns:
    A tuple of six tensors:
      steps_i: (report_n_results,)
      losses: (report_n_results,)
      pose_preds: (report_n_results, n_joints, 3)
      Rs: (report_n_results, 3, 3)
      scales: (report_n_results, 1)
      shifts: (report_n_results, 2)
  """
  def objective(pose_pred, R_re, scale_re, shift, heatmaps):
    R, scale = unreparam(R_re, scale_re)
    logp_loss = total_frame_loss(pose_pred, R, scale, shift, heatmaps)
    losses = [logp_loss]
    if gt_edge_lens is not None:
      edge_len_loss = get_edge_len_loss(gt_edge_lens, pose_pred)
      losses.append(edge_lens_lambda * edge_len_loss)

    loss = sum(losses)
    return loss

  re_params_init = [*reparam(*init_params[:2]), init_params[2]]
  opt_args = ['pose_pred', 'R_re', 'scale_re', 'shift']
  init_argv = [init_pose] + re_params_init + [heatmap_mix_arr]
  step_fn, var_dict = minimize_step_fn(objective, opt, opt_args, init_argv, {})
  collect_every_n = opt_steps // (report_n_results - 1)

  results = []
  loss = objective(*init_argv)
  for step_i in range(opt_steps):
    if step_i % collect_every_n == 0 or step_i == (opt_steps - 1):
      cur_re_params = [tf.identity(var_dict[v]) for v in opt_args]
      cur_params = [cur_re_params[0],
                    *unreparam(*cur_re_params[1:3]),
                    cur_re_params[3]]
      results.append([step_i, loss, *cur_params])

    loss = step_fn()

  result_arrs = list(map(tf.stack, zip(*results)))
  return result_arrs


def convert_rec_pose2d_to_bbox_axis(
    input_rec):
  """Converts coordinates / mixture params in a record from pixels to [0, 1]."""
  # full keys list:
  # 'pose3d', 'cam_pose3d', 'cam_rot', 'cam_intr', 'cam_kd', 'pose2d_gt',
  # 'pose2d_repr', 'heatmaps', 'pose2d_pred', 'keys',
  # 'bboxes', 'pose3d_epi_pred'

  bboxes, pose2d_gt, pose2d_repr, mix_params, mean_pred2d = [
      input_rec[x] for x in
      ['bboxes', 'pose2d_gt', 'pose2d_repr', 'heatmaps', 'pose2d_pred']]

  sizes = tf.math.maximum(
      bboxes[:, 1] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 2])
  origins = tf.stack([bboxes[:, 2], bboxes[:, 0]], axis=-1)

  sizes, origins = [tf.cast(x, tf.float64) for x in [sizes, origins]]

  shifted_mixture_means = mix_params[:, :, :, 1:3] - origins[:, None, None, :]
  mix_params_new = tf.concat([
      mix_params[:, :, :, 0, None],
      (shifted_mixture_means / sizes[:, None, None, None]),
      mix_params[:, :, :, 3, None] / sizes[:, None, None, None]**2
  ], axis=-1)

  pose2d_gt_new = (pose2d_gt - origins[:, None, :]) / sizes[:, None, None]
  pose2d_repr_new = (pose2d_repr - origins[:, None, :]) / sizes[:, None, None]
  mean_pred_new = (mean_pred2d - origins[:, None, :]) / sizes[:, None, None]

  override_rec = {
      'pose2d_gt': pose2d_gt_new,
      'pose2d_repr': pose2d_repr_new,
      'heatmaps': mix_params_new,
      'pose2d_pred': mean_pred_new,
  }

  assert not(override_rec.keys() - input_rec.keys()), 'override has new keys'
  return {**input_rec, **override_rec}


def take_camera_subset(input_rec,
                       subset = None
                       ):
  """Returns a record with a subset of cameras."""
  subset_idx = tf.convert_to_tensor(subset or list(range(4)))
  subset_keys = ['cam_pose3d', 'cam_rot', 'cam_intr', 'cam_kd', 'pose2d_gt',
                 'pose2d_repr', 'heatmaps', 'pose2d_pred', 'keys',
                 'bboxes', 'pose3d_epi_pred']
  override_rec = {
      k: tf.gather(input_rec[k], subset_idx, axis=0) for k in subset_keys
  }

  assert not(override_rec.keys() - input_rec.keys()), 'override has new keys'
  return {**input_rec, **override_rec, 'cam_subset': subset_idx}


def mean_norm(tensor, norm_axis, mean_axis):
  return tf.reduce_mean(tf.linalg.norm(tensor, axis=norm_axis), axis=mean_axis)


def batch_pmpjpe(poses3d, pose3d_gt):
  """Batch procrustes aligned mean per joint position error."""
  aligned_poses = [align_aba(pose, pose3d_gt)[0] for pose in poses3d]
  aligned_poses = tf.convert_to_tensor(aligned_poses)
  pose_err_norms = mean_norm(aligned_poses - pose3d_gt[None], -1, -1)
  return pose_err_norms


def compute_opt_stats(input_rec,
                      iter_opt_results
                      ):
  """Computes per-step metrics for the output of `optimize_heatmap_logp`.

  See `run_inference_optimization` for the full spec of inputs and outputs.

  Args:
    input_rec: an input dict of heatmaps, monocular 3D poses, etc.
    iter_opt_results: an output dict produced by optimize_heatmap_logp.

  Returns:
    A dict of tensors combining input record with predictions and metrics.
  """
  pose3d_gt, pose2d_gt, mean_posenet_pred2d = [
      tf.cast(input_rec[x], tf.float32)
      for x in ['pose3d', 'pose2d_gt', 'pose2d_pred']
  ]

  iters, losses, *opt_results = iter_opt_results
  n_report = losses.shape[0]

  # [n_report, 4, 17, 2]
  opt_pose2d_preds = tf.convert_to_tensor([
      project3d_weak(*[x[viz_id] for x in opt_results])[Ellipsis, :2]
      for viz_id in range(n_report)
  ])

  gt_aligned_projected2d = []
  for viz_id in range(n_report):
    gt_aligned_pose = align_aba(pose3d_gt, opt_results[0][viz_id])[0]
    other_opt_params = [x[viz_id] for x in opt_results[1:]]
    projected2d = project3d_weak(gt_aligned_pose, *other_opt_params)
    gt_aligned_projected2d.append(projected2d[Ellipsis, :2])

  gt_aligned_projected2d = tf.convert_to_tensor(
      gt_aligned_projected2d, dtype=tf.float32)

  iter_pmpjpe = batch_pmpjpe(opt_results[0], pose3d_gt)

  iter_pose2d_err = mean_norm(opt_pose2d_preds - pose2d_gt, 3, (1, 2))
  mean_posenet_gt_err = mean_norm(pose2d_gt - mean_posenet_pred2d, -1, None)
  iter_mean_posenet_err = mean_norm(
      opt_pose2d_preds - mean_posenet_pred2d, 3, (1, 2))
  iter_gt2d_gt_aligned_proj_err = mean_norm(
      gt_aligned_projected2d - pose2d_gt, 3, (1, 2))

  augment_rec = {
      'loss': losses,
      'iters': iters,
      'pose3d_opt_preds': opt_results[0],
      'cam_rot_opt_preds': opt_results[1],
      'scale_opt_preds': opt_results[2],
      'shift_opt_preds': opt_results[3],

      'pose2d_opt_preds': opt_pose2d_preds,
      'pose3d_gt_aligned_pred_3d_proj': gt_aligned_projected2d,
      'pose3d_pred_pmpjpe': iter_pmpjpe,
      'pose2d_pred_err': iter_pose2d_err,
      'pose2d_pred_vs_posenet_err': iter_mean_posenet_err,
      'pose2d_gt_posenet_err_mean': mean_posenet_gt_err,
      'pose3d_gt_backaligned_pose2d_gt_err': iter_gt2d_gt_aligned_proj_err,
  }

  assert not(augment_rec.keys() & input_rec.keys()), 'augment overrides keys'
  return {**augment_rec, **input_rec}


def apply_distortion(xy, kd):
  """Apply full-perspective radial distortion."""
  xx, yy = xy[:, 0], xy[:, 1]
  k1, k2, p1, p2, k3 = [kd[i] for i in range(5)]
  r_sq = xx**2 + yy**2  # r^2
  c_radial = (1 + k1 * r_sq + k2 * r_sq**2 + k3 * r_sq**3)
  x_kd = xx*c_radial + 2*p1*xx*yy + p2*(r_sq + 2*(xx**2))
  y_kd = yy*c_radial + 2*p2*xx*yy + p1*(r_sq + 2*(yy**2))
  xy = tf.stack([x_kd, y_kd], axis=1)
  return xy


def project_3d_tf(
    points,
    cam_pose,
    cam_rot,
    ffpp,
    kd,
    weak = False,
    eps = 1e-8):
  """Apply full-perspective projection. Use mean depth if weak=True."""
  points_cent = (points - cam_pose[None])
  cam_rot_mat_t = tf.transpose(cam_rot)
  cam_frame_points = points_cent @ cam_rot_mat_t
  xy_3d, zz = cam_frame_points[:, 0:2], cam_frame_points[:, 2, None]
  zz_div = tf.reduce_mean(zz, keepdims=True) if weak else zz

  k_mat = tf.cast(tf.convert_to_tensor([ffpp[:2]]), points.dtype)
  pp = tf.cast(tf.convert_to_tensor([ffpp[2:]]), points.dtype)

  # order of application as in:
  # http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
  xy = xy_3d / (zz_div + eps)
  xy = apply_distortion(xy, kd)
  xy = xy * k_mat
  xy = xy + pp

  return xy, zz[:, 0]


def get_fake_gt_heatmaps(
    input_rec, gt_err_std = 0.0):
  """Get "fake" GMM parameters corresponding to (GT + normal(0, std))."""
  pose_gt = input_rec['pose2d_repr']  # ('cams', 'joints', 2)
  n_cams, n_joints = pose_gt.shape[:2]
  gt_noise = tf.random.normal(pose_gt.shape, 0, gt_err_std, dtype=pose_gt.dtype)
  pose_gt_noisy = pose_gt + gt_noise
  mix_params_new = tf.concat([
      0.25 * tf.ones((n_cams, n_joints, 4, 1), dtype=pose_gt.dtype),
      tf.repeat(pose_gt_noisy[:, :, None, :], repeats=4, axis=2),
      0.0003 * tf.ones((n_cams, n_joints, 4, 1), dtype=pose_gt.dtype),
  ], axis=-1)
  return mix_params_new


def recompute_repr_with_weak_proj(
    data_rec):

  """Estimate what 2D GT would have been if the true camera model was weak."""
  new_repr = []
  for i in range(data_rec['cam_pose3d'].shape[0]):
    new_repr.append(
        project_3d_tf(
            data_rec['pose3d'],
            data_rec['cam_pose3d'][i],
            data_rec['cam_rot'][i],
            data_rec['cam_intr'][i],
            data_rec['cam_kd'][i],
            weak=True))
  dt = data_rec['pose2d_repr'].dtype
  return tf.convert_to_tensor([x[0] for x in new_repr], dtype=dt)


def get_single_fake_perfect_epi_init(
    data_rec, cam_id):

  """Estimate what a perfect monocular 3D prediction would have been."""
  cams = [data_rec[k][cam_id]
          for k in ['cam_pose3d', 'cam_rot', 'cam_intr', 'cam_kd']]
  bbox = data_rec['bboxes'][cam_id]

  f_mean = cams[2][:2].numpy().mean()
  proj_weak, zz = project_3d_tf(data_rec['pose3d'], *cams, weak=True)
  mean_z = zz.numpy().mean()
  proj_weak = proj_weak.numpy()
  z_scaled = zz[:, None].numpy() / mean_z * f_mean
  cam_frame_xyz = tf.concat([proj_weak, z_scaled], axis=1)
  cam_frame_xyz = tf.cast(cam_frame_xyz, tf.float32)

  size = tf.math.maximum(bbox[1] - bbox[0], bbox[3] - bbox[2])
  origin = tf.stack([bbox[2], bbox[0], 0], axis=-1)
  size, origin = [tf.cast(x, tf.float32) for x in [size, origin]]
  unit_xyz = (cam_frame_xyz - origin) / size
  return unit_xyz


def get_full_fake_gt_init(data_rec):
  n_cam = data_rec['pose3d_epi_pred'].shape[0]
  epi_init = [get_single_fake_perfect_epi_init(data_rec, cam_id)
              for cam_id in range(n_cam)]
  dt = data_rec['pose3d_epi_pred'].dtype
  return tf.convert_to_tensor(epi_init, dtype=dt)


def run_inference_optimization(data_rec,
                               opt_steps = 100,
                               report_n_results = 50,
                               cam_subset = None,
                               edge_lens_lambda = 0.0,
                               fake_gt_heatmaps = False,
                               fake_gt_ht_std = 0.0,
                               recompute_weak_repr = False,
                               learning_rate = 1e-2,
                               fake_gt_init = False,
                               random_init = False
                               ):
  """Perform full probabilistic bundle adjustment with ablations.

  Arguments:
    data_rec: a dict with the following signature
      * heatmaps (4, 17, 4, 4) float64 - pre-view (4) per-joint (17) pixel
          coordinates with uncertainties in the format
          {(mean_x, mean_y, sigma, pi)}_k^K; obtained by fitting a K-component
          (for K=4) spherical GMMs to join location heatmaps predicted by a
          stacked hourglass 2D pose estimation net
      * pose2d_pred (4, 17, 2) float64 - pre-view per-joint pixel location (x,y)
          predictions estimated from these heatmaps by computing the expected
          value of each GMM
      * keys (4,) string - corresponding frames of the original H36M dataset
          in the format 's%07d_f%07d_c%02d' % (sequence, frame, camera)
      * bboxes (4, 4) int32 - human bounding box used to estimate
          2D pose heatmaps
      * pose3d_epi_pred (4, 17, 3) float32 - per-view predictions of a
          pre-trained monocular 3D pose estimation network
      * pose3d (17, 3) float64 - GT 3D pose in the reference frame (RF) attached
          to the center of mass of the subject, rotated to align the y-axis
          with the hip line, and re-scaled to meters
      * cam_pose3d (4, 3) float64 - GT 3D camera poses in the same RF
      * cam_rot (4, 3, 3) float64 - GT 3D camera rotations in the same RF
      * cam_intr (4, 4) float64 - GT 3D camera intrinsic parameters
      * cam_kd (4, 5) float64 - GT 3D camera distortion parameters
      * pose2d_gt (4, 17, 2) float64 - GT 2D per-frame human pose
      * pose2d_repr (4, 17, 2) float64 - reprojected GT poses

    opt_steps: the number of gradient decent steps to take
    report_n_results: the number of optimizer steps to report
    cam_subset: None (all cameras) or a tuple of cam_ids to use, e.g. (2, 3)
    edge_lens_lambda: the weight of the (personalized) bone lengths loss
    fake_gt_heatmaps: whether to replace GMM parameters with GT GMM with noise
    fake_gt_ht_std: if `fake_gt_heatmaps=True`, the std of the noise added to GT
    recompute_weak_repr: whether to replace GT used to compute "fake GT GMMs"
      with GT one would have had if the true camera model was weak
    learning_rate: Adam learning rate
    fake_gt_init: whether to use perfect (GT) initialization
    random_init: whether to use completely random initialization

  Returns:
    A dict with the same keys as in data_rec and following additional keys:
      * loss (51,) float32 - probabilistic bundle adjustment losses
      * iters (51,) int32 - iteration numbers
      * pose3d_opt_preds (51, 17, 3) float32 - 3D pose predictions
      * cam_rot_opt_preds (51, 4, 3, 3) float32 - camera rotation predictions
      * scale_opt_preds (51, 4) float32 - predictions for the scale weak
          camera parameter
      * shift_opt_preds (51, 4, 3) float32 - predictions for the shift
          weak camera parameter
      * pose2d_opt_preds (51, 4, 17, 2) float32 - predicted 3D pose reprojected
          using predicted camera parameters
      * pose3d_gt_aligned_pred_3d_proj (51, 4, 17, 2) float32  - predicted 3D
          pose reprojected using predicted camera parameters and aligned to
          GT 3D using Procrustes alignment
      * pose3d_pred_pmpjpe (51,) float32 - Procrustes-aligned Mean Per Joint
          Position Error
      * pose2d_pred_err (51,) float32 - 2D reprojection error wrt GT
      * pose2d_pred_vs_posenet_err (51,) float32 - 2D error of the stack
          hourglass network
      * pose2d_gt_posenet_err_mean () float32 - final 2D error
      * pose3d_gt_backaligned_pose2d_gt_err (51,) float32 - 2D error of
          the predicted 3D projected using GT camera parameters
      * cam_subset (4,) int32 - a subset of cameras used for inference
  """

  if recompute_weak_repr:
    # in case we want to train with "ground truth" weak projections
    # to _simulate_ zero camera model error; never use in production
    data_rec['pose2d_repr'] = recompute_repr_with_weak_proj(data_rec)

  if fake_gt_init:
    # project ground truth to get perfect mono 3d estimates
    data_rec['pose3d_epi_pred'] = get_full_fake_gt_init(data_rec)

  if random_init:
    data_init = data_rec['pose3d_epi_pred']
    data_rec['pose3d_epi_pred'] = tf.random.normal(
        data_init.shape, dtype=data_init.dtype)

  data_rec = convert_rec_pose2d_to_bbox_axis(data_rec)
  data_rec = take_camera_subset(data_rec, cam_subset)
  cn_mean_pred, init_cam = initial_epi_estimate(data_rec['pose3d_epi_pred'])

  if fake_gt_heatmaps:
    # replace real predicted joint heatmaps with gaussians around ground truth
    data_rec['heatmaps'] = get_fake_gt_heatmaps(data_rec, fake_gt_ht_std)

  if edge_lens_lambda > 0:
    gt_edge_lens = tf.cast(get_h36m_edge_lens(data_rec['pose3d']), tf.float32)
  else:
    gt_edge_lens = None

  opt = tf.keras.optimizers.Adam(learning_rate)
  iter_opt_results = optimize_heatmap_logp(
      cn_mean_pred, init_cam, data_rec['heatmaps'],
      gt_edge_lens, edge_lens_lambda,
      report_n_results=report_n_results, opt=opt, opt_steps=opt_steps)

  opt_stats = compute_opt_stats(data_rec, iter_opt_results)
  opt_stats = {k: tf.convert_to_tensor(v).numpy()
               for k, v in opt_stats.items()}

  # for k, v in opt_stats.items():
  #   print(k, v.dtype)
  return opt_stats

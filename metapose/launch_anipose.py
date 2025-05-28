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

"""Launch script for running a classical bundle adjustment baseline."""
from absl import app
from absl import flags
from aniposelib import cameras as anipose_cameras
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from metapose import data_utils
from metapose import inference_time_optimization as inf_opt

_INPUT_PATH = flags.DEFINE_string('input_path', '', '')
_OUTPUT_PATH = flags.DEFINE_string('output_path', None, '')
_DEBUG_FIRST_N = flags.DEFINE_integer('debug_first_n', None,
                                      'read only first n records')
_CAM_SUBSET = flags.DEFINE_list('cam_subset', list(map(str, range(4))), '')
_ADJUST_CAM = flags.DEFINE_bool('adjust_cam', True,
                                'true - BA on cams, false - triangulation')
_GT_INIT = flags.DEFINE_bool('gt_init', True, '')
_GT_HT = flags.DEFINE_bool('gt_ht', False, '')

flags.mark_flag_as_required('output_path')


def get_anipose_gt_camera(data_rec, cam_id):
  """Get AniPose.Camera with GT initialization."""
  cam_pose3d, cam_rot, cam_intr = [
      data_rec[x] for x in ['cam_pose3d', 'cam_rot', 'cam_intr']
  ]

  rot_mat = cam_rot[cam_id].numpy()
  t = cam_pose3d[cam_id].numpy()
  rvec = cv2.Rodrigues(rot_mat)[0]
  tvec = np.dot(rot_mat, -t)
  f = np.mean(cam_intr[cam_id][:2])
  k1 = 0

  params = np.concatenate([rvec.ravel(), tvec.ravel(), [f, k1]])
  camera = anipose_cameras.Camera()
  camera.set_params(params)

  return camera


def perturb_anipose_camera(cam):
  ani_params = cam.get_params()
  new_ani_params = np.concatenate([np.random.normal(size=6), ani_params[6:]])
  cam.set_params(new_ani_params)


def get_pppx_pose2d_pred_and_gt(data_rec, cam_id):
  """Get 2D pose GT and mean predicted per-view 2D pose in pixels."""
  bbox = data_rec['bboxes'][cam_id].numpy()
  size = np.maximum(bbox[1] - bbox[0], bbox[3] - bbox[2])
  origin = np.stack([bbox[2], bbox[0]], axis=-1)

  pose2d_mean_pred = data_rec['pose2d_pred'][cam_id].numpy()
  pose2d_gt = data_rec['pose2d_repr'][cam_id].numpy()
  pp = data_rec['cam_intr'][cam_id][2:].numpy()

  pose2d_mean_pred_pix = pose2d_mean_pred * size + origin - pp
  pose2d_gt_pix = pose2d_gt * size + origin - pp
  return pose2d_mean_pred_pix, pose2d_gt_pix


def run_anipose_bundle_adjustment(data_rec,
                                  ba_cam=True,
                                  gt_init=True,
                                  gt_ht=False):
  """Run AniPose bundle adjustment."""
  n_cam = data_rec['cam_rot'].shape[0]
  cameras = [get_anipose_gt_camera(data_rec, i) for i in range(n_cam)]

  if not gt_init:
    for cam in cameras:
      perturb_anipose_camera(cam)

  gt_idx = 1 if gt_ht else 0
  pose2d_preds = np.array(
      [get_pppx_pose2d_pred_and_gt(data_rec, i)[gt_idx] for i in range(n_cam)])

  camera_group = anipose_cameras.CameraGroup(cameras)

  if ba_cam:
    error = camera_group.bundle_adjust_iter(pose2d_preds)

  pose3d_pred = camera_group.triangulate(pose2d_preds)
  ani_params_cam0 = camera_group.cameras[0].get_params()
  rot_mat_cam0 = cv2.Rodrigues(ani_params_cam0[:3])[0]
  pose3d_pred_cam0 = pose3d_pred @ rot_mat_cam0.T

  all_ani_params = np.array([c.get_params() for c in camera_group.cameras])

  return pose3d_pred_cam0, all_ani_params, error


def pmpje(pose3d_pred, pose3d_gt):
  aligned_pose = inf_opt.align_aba(pose3d_pred, pose3d_gt)[0]
  diff = aligned_pose - pose3d_gt
  return tf.reduce_mean(tf.linalg.norm(diff, axis=-1), axis=-1)


def center_pose(pose3d):
  return pose3d - tf.reduce_mean(pose3d, axis=0, keepdims=True)


def nmpje_pck(pose3d_pred_cam0, pose3d_gt_cam0, threshold=150):
  """Compute Normalized MPJE and PCK metrics."""
  norm = tf.linalg.norm
  pose3d_gt_cent = center_pose(pose3d_gt_cam0)
  pose3d_pred_cent = center_pose(pose3d_pred_cam0)
  scale_factor = norm(pose3d_gt_cent) / norm(pose3d_pred_cent)
  pose3d_pred_cent_scaled = scale_factor * pose3d_pred_cent
  diff = pose3d_gt_cent - pose3d_pred_cent_scaled
  err = errs = tf.linalg.norm(diff, axis=-1)
  nmpje = tf.reduce_mean(err, axis=-1)
  pck = tf.reduce_mean(tf.cast(errs < threshold, tf.float32)) * 100
  return nmpje, pck


def run_and_evaluate(data_rec, cam_subset, adjust_cam, gt_init, gt_ht):
  """Run AniPose bundle adjustment and compute metrics."""
  data_rec = inf_opt.convert_rec_pose2d_to_bbox_axis(data_rec)
  data_rec = inf_opt.take_camera_subset(data_rec, cam_subset)

  pose3d_gt = data_rec['pose3d'].numpy()
  pose3d_gt_cam0 = (pose3d_gt @ tf.transpose(data_rec['cam_rot'][0]))

  pose3d_pred_cam0, ani_params_pred, ba_error = run_anipose_bundle_adjustment(
      data_rec, adjust_cam, gt_init, gt_ht)

  errs = np.array([
      ba_error,
      pmpje(pose3d_pred_cam0, pose3d_gt),
      *nmpje_pck(pose3d_pred_cam0, pose3d_gt_cam0)
  ])

  output = {
      **data_rec,
      'pose3d_pred_cam0': pose3d_pred_cam0,
      'ani_params_pred': ani_params_pred,
      'ba_pmpje_nmpje_pck_errs': errs,
  }

  output_np = {k: np.array(v) for k, v in output.items()}
  return output_np


def main(_):
  cam_subset = list(map(int, _CAM_SUBSET.value))
  n_cam = len(cam_subset)

  output_shape_dtype = {
      # anipose results
      'pose3d_pred_cam0': ([17, 3], tf.float64),
      'ani_params_pred': ([n_cam, 8], tf.float64),
      'ba_pmpje_nmpje_pck_errs': ([
          4,
      ], tf.float64),

      # input data
      'pose3d': ([17, 3], tf.float64),
      'cam_pose3d': ([n_cam, 3], tf.float64),
      'cam_rot': ([n_cam, 3, 3], tf.float64),
      'cam_intr': ([n_cam, 4], tf.float64),
      'cam_kd': ([n_cam, 5], tf.float64),
      'pose2d_gt': ([n_cam, 17, 2], tf.float64),
      'pose2d_repr': ([n_cam, 17, 2], tf.float64),
      'heatmaps': ([n_cam, 17, 4, 4], tf.float64),
      # note! pose2d_pred is actually the "mean heatmap" 2D pred
      'pose2d_pred': ([n_cam, 17, 2], tf.float64),
      'keys': ([n_cam], tf.string),
      'bboxes': ([n_cam, 4], tf.int32),
      'pose3d_epi_pred': ([n_cam, 17, 3], tf.float32),

      # config
      'cam_subset': ([n_cam], tf.int32),
  }

  output_spec = tfds.features.FeaturesDict({
      k: tfds.features.Tensor(shape=s, dtype=d)
      for k, (s, d) in output_shape_dtype.items()
  })

  ds = data_utils.read_tfrec_feature_dict_ds(_INPUT_PATH.value)

  if _DEBUG_FIRST_N.value is not None:
    ds = ds.take(_DEBUG_FIRST_N.value)

  dataset = []
  for _, data_rec in ds:
    opt_stats = run_and_evaluate(data_rec, cam_subset, _ADJUST_CAM.value,
                                 _GT_INIT.value, _GT_HT.value)

    print('ba / pmpje / nmpje / pck', opt_stats['ba_pmpje_nmpje_pck_errs'])
    dataset.append(opt_stats)

  data_utils.write_tfrec_feature_dict_ds(dataset, output_spec,
                                         _OUTPUT_PATH.value)


if __name__ == '__main__':
  app.run(main)

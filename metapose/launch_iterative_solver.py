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

"""Launch script for running a full probabilistic iterative solver baseline."""
from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_datasets as tfds

from metapose import data_utils
from metapose import inference_time_optimization as inf_opt


_INPUT_PATH = flags.DEFINE_string(
    'input_path', '',
    'path to an folder containing a tfrec file and a features.json file')

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None,
    'path to the output a dataset with refined 3d poses')

_N_STEPS = flags.DEFINE_integer('n_steps', 100, 'optimizer (adam) steps')
_DEBUG_FIRST_N = flags.DEFINE_integer(
    'debug_first_n', None, 'read only first n records')
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 1e-2, 'optimizer (adam) learning rate')

_REPORT_N_APPROX = flags.DEFINE_integer(
    'report_n_approx', 50,
    'number of intermediate optimization results to report')
_CAM_SUBSET = flags.DEFINE_list(
    'cam_subset', list(map(str, range(4))),
    'comma-separated list of camera ids to use, e.g. 3,4,5')
_GT_HEATMAPS = flags.DEFINE_bool(
    'gt_heatmaps', False,
    'whether to replace heatmaps with fake ground truth heatmaps')
_FAKE_GT_HT_STD = flags.DEFINE_float(
    'fake_gt_ht_std', 0.0,
    'how much noise to add to positions of means of fake gt heatmaps')
_USE_WEAK_REPR = flags.DEFINE_bool(
    'use_weak_repr', False,
    'whether to use weak projection to get ground truth heatmaps')
_FAKE_GT_INIT = flags.DEFINE_bool(
    'fake_gt_init', False,
    'whether to use ground truth instead of monocular 3d predictions')
_RANDOM_INIT = flags.DEFINE_bool(
    'random_init', False,
    'whether to use random noise instead of monocular 3d predictions')
_EDGE_LENS_LAMBDA = flags.DEFINE_float(
    'edge_lens_lambda', 0.0,
    'weight of the normalized limb length loss during refinement')

flags.mark_flag_as_required('output_path')


def main(_):
  cam_subset = list(map(int, _CAM_SUBSET.value))
  n_cam = len(cam_subset)
  report_n = (
      _N_STEPS.value // (_N_STEPS.value // (_REPORT_N_APPROX.value - 1)) + 1)

  output_shape_dtype = {
      # optimization results
      'loss': ([report_n], tf.float32),
      'iters': ([report_n], tf.int32),
      'pose3d_opt_preds': ([report_n, 17, 3], tf.float32),
      'cam_rot_opt_preds': ([report_n, n_cam, 3, 3], tf.float32),
      'scale_opt_preds': ([report_n, n_cam], tf.float32),
      'shift_opt_preds': ([report_n, n_cam, 3], tf.float32),

      # metrics
      'pose2d_opt_preds': ([report_n, n_cam, 17, 2], tf.float32),
      'pose3d_gt_aligned_pred_3d_proj': ([report_n, n_cam, 17, 2], tf.float32),
      'pose3d_pred_pmpjpe': ([report_n], tf.float32),
      'pose2d_pred_err': ([report_n], tf.float32),
      'pose2d_pred_vs_posenet_err': ([report_n], tf.float32),
      'pose2d_gt_posenet_err_mean': ([], tf.float32),
      'pose3d_gt_backaligned_pose2d_gt_err': ([report_n], tf.float32),

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
    opt_stats = inf_opt.run_inference_optimization(
        data_rec=data_rec,
        opt_steps=_N_STEPS.value,
        report_n_results=_REPORT_N_APPROX.value,
        cam_subset=cam_subset,
        edge_lens_lambda=_EDGE_LENS_LAMBDA.value,
        fake_gt_heatmaps=_GT_HEATMAPS.value,
        fake_gt_ht_std=_FAKE_GT_HT_STD.value,
        fake_gt_init=_FAKE_GT_INIT.value,
        random_init=_RANDOM_INIT.value,
        recompute_weak_repr=_USE_WEAK_REPR.value,
        learning_rate=_LEARNING_RATE.value)

    print('pmpjpe', opt_stats['pose3d_pred_pmpjpe'][-1])
    dataset.append(opt_stats)

  data_utils.write_tfrec_feature_dict_ds(
      dataset, output_spec, _OUTPUT_PATH.value)

if __name__ == '__main__':
  app.run(main)

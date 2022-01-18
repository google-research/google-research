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

"""SMURF multi-frame self-supervision utility functions."""

import functools
from typing import Callable, Tuple

import gin
import tensorflow as tf

from smurf import smurf_plotting
from smurf import smurf_utils
from smurf.data_conversion_scripts import conversion_utils
from smurf.multiframe_training import tiny_model
from smurf.smurf_net import SMURFNet

gin.enter_interactive_mode()


def get_occlusion_inference_function(
    boundaries_occluded = False
):
  """Returns a function that estimates occlusions based on flow fields.

  Args:
    boundaries_occluded: If false pixels moving outside the image frame will be
      considered non-occluded.

  Returns:
    Function that estimates an occlusion maskbased on flow fields.
  """
  return functools.partial(
      smurf_utils.compute_occlusions,
      occlusion_estimation='brox',
      occlusions_are_zeros=True,
      boundaries_occluded=boundaries_occluded)


def get_flow_inference_function(
    checkpoint, height,
    width):
  """Restores a raft model from a checkpoint and returns the inference function.

  Args:
    checkpoint: Path to the checkpoint that will be used.
    height: Image height that should be used for inference.
    width: Image width that will be used for inference.

  Returns:
    Inference function of the restored model.
  """
  tf.keras.backend.clear_session()
  gin.parse_config('raft_model_parameters.max_rec_iters = 32')
  smurf = SMURFNet(
      checkpoint, flow_architecture='raft', feature_architecture='raft')
  smurf.restore()
  return functools.partial(
      smurf.infer_no_tf_function,
      input_height=height,
      input_width=width,
      resize_flow_to_img_res=True,
      infer_occlusion=False,
      infer_bw=False)


def run_multiframe_fusion(
    images, infer_flow,
    infer_mask
):
  """Computes flow field and masks and runs the multiframe fusion.

  Args:
    images: Image triplet.
    infer_flow: Computes a flow field for a given image pair.
    infer_mask: Computes an occlusion mask for a given forward/backward flow
      pair.

  Returns:
    A flow field with the associated mask.
  """
  # Compute all required flow fields. This includes temporal forward and
  # backward flow fields with the respective inverse frame order.
  flow_t1_t2_fw = infer_flow(images[1], images[2])
  flow_t1_t2_bw = infer_flow(images[2], images[1])
  flow_t1_t0_fw = infer_flow(images[1], images[0])
  flow_t1_t0_bw = infer_flow(images[0], images[1])

  mask_t1_t2 = infer_mask(flow_t1_t2_fw[None], flow_t1_t2_bw[None])
  mask_t1_t0 = infer_mask(flow_t1_t0_fw[None], flow_t1_t0_bw[None])

  return tiny_model.train_and_run_tiny_model(flow_t1_t2_fw[None],
                                             flow_t1_t0_fw[None], mask_t1_t2,
                                             mask_t1_t0)


def create_output_sequence_example(
    images,
    flow,
    mask,
    add_visualization = True):
  """Creates a SequenceExample for the self-supervised training data.

  Args:
    images: Image triplet.
    flow: Flow field of the middle frame to the last frame.
    mask: Mask associated with the flow field indicating which locations hold a
      valid flow vector.
    add_visualization: If true adds a visualization of the flow field to the
      sequence example.

  Returns:
    Tensorflow SequenceExample holding the training data created of the triplet.
  """
  height = tf.shape(images)[-3]
  width = tf.shape(images)[-2]

  # Compute a flow visualization.
  if add_visualization:
    flow_visualization = tf.image.convert_image_dtype(
        smurf_plotting.flow_to_rgb(flow)[0], tf.uint8)
    flow_visualization_png = tf.image.encode_png(flow_visualization)

  context_features = {
      'height': conversion_utils.int64_feature(height),
      'width': conversion_utils.int64_feature(width),
      'flow_uv': conversion_utils.bytes_feature(flow[0].numpy().tobytes()),
      'flow_valid': conversion_utils.bytes_feature(mask.numpy().tobytes()),
  }
  if add_visualization:
    context_features['flow_viz'] = (
        conversion_utils.bytes_feature(flow_visualization_png.numpy()))

  sequence_features = {
      'images':
          tf.train.FeatureList(feature=[
              conversion_utils.bytes_feature(
                  tf.image.encode_png(
                      tf.image.convert_image_dtype(images[1],
                                                   tf.uint8)).numpy()),
              conversion_utils.bytes_feature(
                  tf.image.encode_png(
                      tf.image.convert_image_dtype(images[2],
                                                   tf.uint8)).numpy())
          ])
  }
  return tf.train.SequenceExample(
      context=tf.train.Features(feature=context_features),
      feature_lists=tf.train.FeatureLists(feature_list=sequence_features))

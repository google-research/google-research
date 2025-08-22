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

r"""Scripts to export saved models.

The anchor implementation is based on:
https://github.com/tensorflow/tpu/blob/master/models/official/detection/dataloader/anchor.py
"""

import enum
import logging
from typing import Dict, Any

from absl import app
from absl import flags
from flax.training import checkpoints
import gin
import jax
import numpy as np
import tensorflow as tf
from utils import saved_model_lib


_INPUT_DIR = flags.DEFINE_string(
    'input_dir', None, 'Path under which to load the JAX model.'
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Path under which to save the SavedModel.'
)
_MODEL_NAME = flags.DEFINE_string(
    'model_name', 'resnet_50', 'The name of the backbone model to export.'
)
_IMAGE_SIZE = flags.DEFINE_integer(
    'image_size', 1024, 'Image size to serve the model at.'
)
_VLM_WEIGHT = flags.DEFINE_float(
    'vlm_weight',
    0.65,
    'A float between [0, 1] as a tradeoff between open/closed-set detection.',
)
_SERVING_BATCH_SIZE = flags.DEFINE_integer(
    'serving_batch_size',
    1,
    'For what batch size to prepare the serving signature.',
)
_MAX_NUM_CLASSES = flags.DEFINE_integer(
    'max_num_classes', 30, 'Maximum number of classes to feed in by the user.'
)
_INCLUDE_MASK = flags.DEFINE_bool(
    'include_mask', True, 'Whether to include mask.'
)
_MODEL_CONFIG_PATH = flags.DEFINE_string(
    'model_config_path',
    './configs/export_model.gin',
    'The path to model gin config.',
)
_CONFIG_OVERRIDES = flags.DEFINE_multi_string(
    'config_overrides',
    None,
    'Gin bindings to override the config given in model_config_path flag.',
)


@gin.constants_from_enum
class ExecutionMode(enum.Enum):
  """Defines the model execution mode."""
  TRAIN = 1
  EVAL = 2
  PREDICT = 3


class Anchor:
  """Anchor class for anchor-based object detectors."""

  def __init__(self,
               min_level,
               max_level,
               num_scales,
               aspect_ratios,
               anchor_size,
               image_size):
    """Constructs multiscale anchors.

    Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instance, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of float numbers representing the aspect ratio anchors
        added on each level. The number indicates the ratio of width to height.
        For instance, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each
        scale level.
      anchor_size: float number representing the scale of size of the base
        anchor to the feature stride 2^level.
      image_size: a list of integer numbers or Tensors representing
        [height, width] of the input image size.The image_size should be divided
        by the largest feature stride 2^max_level.
    """
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_size = anchor_size
    self.image_size = image_size
    self.boxes = self._generate_boxes()

  def _generate_boxes(self):
    """Generates multiscale anchor boxes.

    Returns:
      a Tensor of shape [N, 4], representing anchor boxes of all levels
      concatenated together.
    """
    boxes_all = []
    for level in range(self.min_level, self.max_level + 1):
      boxes_l = []
      for scale in range(self.num_scales):
        for aspect_ratio in self.aspect_ratios:
          stride = 2 ** level
          intermidate_scale = 2 ** (scale / float(self.num_scales))
          base_anchor_size = self.anchor_size * stride * intermidate_scale
          aspect_x = aspect_ratio ** 0.5
          aspect_y = aspect_ratio ** -0.5
          half_anchor_size_x = base_anchor_size * aspect_x / 2.0
          half_anchor_size_y = base_anchor_size * aspect_y / 2.0
          x = tf.range(stride / 2, self.image_size[1], stride)
          y = tf.range(stride / 2, self.image_size[0], stride)
          xv, yv = tf.meshgrid(x, y)
          xv = tf.cast(tf.reshape(xv, [-1]), dtype=tf.float32)
          yv = tf.cast(tf.reshape(yv, [-1]), dtype=tf.float32)
          # Tensor shape Nx4.
          boxes = tf.stack([yv - half_anchor_size_y, xv - half_anchor_size_x,
                            yv + half_anchor_size_y, xv + half_anchor_size_x],
                           axis=1)
          boxes_l.append(boxes)
      # Concat anchors on the same level to tensor shape NxAx4.
      boxes_l = tf.stack(boxes_l, axis=1)
      boxes_l = tf.reshape(boxes_l, [-1, 4])
      boxes_all.append(boxes_l)
    return tf.concat(boxes_all, axis=0)

  def unpack_labels(self, labels,
                    is_box = False):
    """Unpacks an array of labels into multiscales labels.

    Args:
      labels: labels to unpack.
      is_box: to unpack anchor boxes or not. If it is true, will unpack to 2D,
        otherwise, will unpack to 3D.

    Returns:
      unpacked_labels: a dictionary contains unpack labels in different levels.
    """
    unpacked_labels = {}
    count = 0
    for level in range(self.min_level, self.max_level + 1):
      feat_size_y = tf.cast(self.image_size[0] / 2 ** level, tf.int32)
      feat_size_x = tf.cast(self.image_size[1] / 2 ** level, tf.int32)
      steps = feat_size_y * feat_size_x * self.anchors_per_location
      if is_box:
        unpacked_labels[level] = tf.reshape(labels[count:count + steps],
                                            [-1, 4])
      else:
        unpacked_labels[level] = tf.reshape(labels[count:count + steps],
                                            [feat_size_y, feat_size_x, -1])
      count += steps
    return unpacked_labels

  @property
  def anchors_per_location(self):
    return self.num_scales * len(self.aspect_ratios)

  @property
  def multilevel_boxes(self):
    return self.unpack_labels(self.boxes, is_box=True)


@gin.configurable
def generate_anchors_info(
    min_level=2,
    max_level=6,
    aspect_ratios=(1.0, 2.0, 0.5),
):
  """Generate anchors and image info."""
  original_height, original_width = 512, 640
  input_anchor = Anchor(
      min_level=min_level,
      max_level=max_level,
      num_scales=1,
      aspect_ratios=list(aspect_ratios),
      anchor_size=8,
      image_size=(_IMAGE_SIZE.value, _IMAGE_SIZE.value),
  )
  anchor_boxes = input_anchor.multilevel_boxes
  for key in anchor_boxes:
    anchor_boxes[key] = anchor_boxes[key].numpy()

  scale = min(_IMAGE_SIZE.value / original_height,
              _IMAGE_SIZE.value / original_width)
  image_info = np.array([[[original_height, original_width],
                          [_IMAGE_SIZE.value, _IMAGE_SIZE.value],
                          [scale, scale], [0, 0]]])

  return anchor_boxes, image_info


def load_gin_configs():
  """Load gin configs for F-VLM/DITO model."""
  clip_model_embed_dim = {
      'resnet_50': (1024, 32, 7),
      'resnet_50x4': (640, 40, 9),
      'resnet_50x16': (768, 48, 12),
      'resnet_50x64': (1024, 64, 14),
      'vit_l16': (1024, 0, 14),
  }
  config_path = _MODEL_CONFIG_PATH.value
  text_dim, model_num_heads, roi_size = clip_model_embed_dim[_MODEL_NAME.value]
  gin.parse_config_files_and_bindings(
      [config_path], _CONFIG_OVERRIDES.value, finalize_config=False
  )
  gin.parse_config(f'CATG_PAD_SIZE = {_MAX_NUM_CLASSES.value}')
  gin.parse_config(f'CLIP_NAME = "{_MODEL_NAME.value}"')
  gin.parse_config(f'TEXT_DIM = {text_dim}')
  gin.parse_config(f'AttentionPool.num_heads = {model_num_heads}')
  gin.parse_config(f'ClipFasterRCNNHead.roi_output_size = {roi_size}')
  gin.parse_config(f'ClipFasterRCNNHead.novel_vlm_weight = {_VLM_WEIGHT.value}')
  gin.parse_config(f'INCLUDE_MASK = {_INCLUDE_MASK.value}')

  return _MAX_NUM_CLASSES.value, text_dim


def generate_rng_dict(base_rng):
  """Generates a dictionary of rngs to pass in to `nn.Module`s.

  Stochastic layers in Flax Modules use separate stream of random number
  generators (e.g. dropout requires an rng named 'dropout'). This function
  generates all rngs needed for stochastic layers.

  Args:
    base_rng: The base rng to split.

  Returns:
    A dictionary of rngs to be used in calling modules.
  """
  keys = ('dropout', 'stochastic_depth', 'rng')
  rngs = jax.random.split(base_rng, len(keys))
  return {key: rngs[i] for i, key in enumerate(keys)}


@gin.configurable
def create_predict_step(model_fn = gin.REQUIRED):
  """Get prediction step function.

  Args:
    model_fn: A flax.deprecated.nn.module of forward model to use.

  Returns:
    model_outputs: A dictionary of model_outputs.
  """
  def predict_step_v2(variables, batch, rng):
    features, _ = batch if isinstance(batch, tuple) else (batch, {})
    rng, _ = jax.random.split(rng)
    pred_model_fn = model_fn(mode=ExecutionMode.EVAL)
    model_outputs = pred_model_fn.apply(
        variables,
        **features,
        mutable=False,
        _do_remap=True,
        rngs=generate_rng_dict(rng))
    return model_outputs

  return predict_step_v2


def get_fvlm_predict_fn(serving_batch_size):
  """Get predict function and input signatures for F-VLM model."""
  num_classes, text_dim = load_gin_configs()
  predict_step = create_predict_step()
  anchor_boxes, image_info = generate_anchors_info()

  def predict_fn(params, input_dict):
    input_dict['labels'] = {
        'detection': {
            'anchor_boxes': anchor_boxes,
            'image_info': image_info,
        }
    }
    output = predict_step(params, input_dict, jax.random.PRNGKey(0))
    output = output['detection']
    output.pop('rpn_score_outputs')
    output.pop('rpn_box_outputs')
    output.pop('class_outputs')
    output.pop('box_outputs')
    return output

  input_signatures = {
      'image':
          tf.TensorSpec(
              shape=(serving_batch_size, _IMAGE_SIZE.value, _IMAGE_SIZE.value,
                     3),
              dtype=tf.bfloat16,
              name='image'),
      'text':
          tf.TensorSpec(
              shape=(serving_batch_size, num_classes, text_dim),
              dtype=tf.float32,
              name='queries'),
  }
  return predict_fn, input_signatures


def restore_checkpoint(restore_dir):
  """Restore checkpoint into variables.

  Args:
    restore_dir: A string of path to restore checkpoint from.

  Returns:
    variables: A nested dictionary of restore parameters and model states.
  """
  restored_train_state = checkpoints.restore_checkpoint(restore_dir, None)
  variables = {'params': restored_train_state['optimizer']['target']}
  model_state = restored_train_state['model_state']
  variables.update(model_state)
  return variables


def main(argv):
  del argv
  logging.info('Creating predict_fn.')
  predict_fn, input_signatures = get_fvlm_predict_fn(_SERVING_BATCH_SIZE.value)
  logging.info('Loading model for %s.', _INPUT_DIR.value)
  predict_params = restore_checkpoint(_INPUT_DIR.value)
  logging.info('Saving model for %s.', _OUTPUT_DIR.value)
  saved_model_lib.convert_and_save_model(
      predict_fn,
      predict_params,
      _OUTPUT_DIR.value,
      input_signatures=[input_signatures],
      polymorphic_shapes=None,
  )


if __name__ == '__main__':
  app.run(main)
  flags.mark_flag_as_required('input_dir')
  flags.mark_flag_as_required('output_dir')

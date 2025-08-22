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

"""Demo for the FindIt model.
"""

import enum
import functools

from absl import app
import jax

from findit import inputs
from findit.multitask_model import MultitaskModel


class ExecutionMode(enum.Enum):
  """Defines the model execution mode."""
  TRAIN = 1
  EVAL = 2
  PREDICT = 3


def predict_step(features, model_fn):
  """Prediction step function.

  Args:
    features: A three-tuple of (image, labels, text):
      image - A float array of normalized image of shape [1, normalized_height,
        [normalized_width, 3].
      labels - A dictionary of tensors used for training. The following are the
        {key: value} pairs in the dictionary.
        image_info - A 2D `Tensor` that encodes the information of the image and
          the applied preprocessing. It is in the format of
          [[original_height, original_width], [scaled_height, scaled_width],
        anchor_boxes - An ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, 4] representing anchor boxes at each level.
        groundtruths - A dictionary with keys
          source_id - Groundtruth source id.
          height - Original image height.
          width - Original image width.
          boxes - Groundtruth bounding box annotations. The box is represented
             in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled
             image that is fed to the network. The tennsor is padded with -1 to
             the fixed dimension [self._max_num_instances, 4].
          classes - Groundtruth classes annotations. The tennsor is padded
            with -1 to the fixed dimension [self._max_num_instances].
          areas - Box area or mask area depend on whether mask is present.
          is_crowds - Whether the ground truth label is a crowd label.
          num_groundtruths - Number of ground truths in the image.
      text - A tokenized texts array of shape
        (1, num_boxes, num_expressions, sentence_length)
    model_fn: A flax.deprecated.nn.module of forward model to use.

  Returns:
    model_outputs: A dictionary of model_outputs.
  """
  model_outputs, _ = model_fn(mode=ExecutionMode.PREDICT).init_with_output(
      jax.random.PRNGKey(0), *features)
  return model_outputs


def main(_):
  image, text = inputs.fake_image_text()
  loader_fn = functools.partial(
      inputs.tfds_from_tensor_dict,
      tensor_dict=inputs.refexp_input_dict_from_image_text(image, text))
  dataset = inputs.get_input(loader_fn=loader_fn, map_fn=inputs.ref_expr_map_fn)
  for data in dataset:
    model_outputs = predict_step(data, MultitaskModel)
    print('================================================')
    print('Top 3 predicted boxes:')
    print(model_outputs['refexp']['detection_boxes'][0, :3])
    print('================================================')
    break


if __name__ == '__main__':
  app.run(main)

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

# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Demo for the RO-ViT paper (CVPR 2023).

Paper link: https://arxiv.org/abs/2305.07011.

This demo takes a sample image, and produces detections using a pretrained
RO-ViT model.


"""

from collections.abc import Sequence
import functools
import json

from absl import app
from absl import flags
from absl import logging
from demo_utils import input_utils
from demo_utils import vis_utils
import jax
import numpy as np
from PIL import Image
import tensorflow as tf

_DEMO_IMAGE_NAME = flags.DEFINE_string('demo_image_name', 'citrus.jpg',
                                       'The image file name under data/.')
_MODEL = flags.DEFINE_enum('model', 'vit-large', ['vit-large'],
                           'RO-ViT model size to use.')
_MODEL_NAME = 'rovit'
_TEXT_EMBED = flags.DEFINE_enum('text_embed', 'lvis',
                                ['lvis', 'lvis-base'],
                                'Text embeddings to use.')
_MAX_BOXES_TO_DRAW = flags.DEFINE_integer('max_boxes_to_draw', 25,
                                          'Max number of boxes to draw.')
_MIN_SCORE_THRESH = flags.DEFINE_float('min_score_thresh', 0.05,
                                       'Min score threshold.')
_MAX_TEXT_EMBEDDING = 1204  # Max text embeddings allowed by the saved model.


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  demo_image_path = f'./data/{_DEMO_IMAGE_NAME.value}'
  output_image_path = demo_image_path.replace('data', 'output')
  output_image_path = (
      output_image_path[:-4]
      + f'_{_MODEL_NAME}{_MODEL.value.replace("vit", "")}'
      + output_image_path[-4:]
  )
  with tf.io.gfile.GFile(demo_image_path, 'rb') as f:
    np_image = np.array(Image.open(f))

  # Load text embeddings.
  with tf.io.gfile.GFile(
      f'./rovit/embeddings/{_TEXT_EMBED.value}_embed.npy', 'rb') as f:
    text_embeddings = np.load(f)
    text_embeddings = text_embeddings[np.newaxis, :_MAX_TEXT_EMBEDDING]

  # Load category names.
  with tf.io.gfile.GFile(
      f'./datasets/{_TEXT_EMBED.value}_mapping.json', 'r') as f:
    id_mapping = json.load(f)
    # Convert to integer key for visualization.
    id_mapping = {int(k): v for k, v in id_mapping.items()}
    id_mapping[0] = 'background'

  # Parse the image data.
  parser_fn = input_utils.get_rovit_parser()
  data = parser_fn({'image': np_image, 'source_id': np.array([0])})
  np_data = jax.tree.map(lambda x: x.numpy()[np.newaxis, Ellipsis], data)
  np_data['text'] = text_embeddings
  np_data['image'] = np_data.pop('images')
  labels = np_data.pop('labels')
  image = np_data['image']

  logging.info('Loading saved model.')
  saved_model_dir = f'./rovit/checkpoints/{_MODEL.value}'
  model = tf.saved_model.load(saved_model_dir)

  logging.info('Computing forward pass.')
  output = model(np_data)

  category_index = input_utils.get_category_index(id_mapping)
  maskrcnn_visualizer_fn = functools.partial(
      vis_utils.visualize_boxes_and_labels_on_image_array,
      category_index=category_index,
      use_normalized_coordinates=False,
      max_boxes_to_draw=_MAX_BOXES_TO_DRAW.value,
      min_score_thresh=_MIN_SCORE_THRESH.value,
      skip_labels=False)
  vis_image = vis_utils.visualize_instance_segmentations(
      output, image, labels['image_info'], maskrcnn_visualizer_fn,
      offset=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
  )
  pil_vis_image = Image.fromarray(vis_image, mode='RGB')
  pil_vis_image.save(output_image_path)
  logging.info('Completed saving the output image at %s.', output_image_path)


if __name__ == '__main__':
  app.run(main)

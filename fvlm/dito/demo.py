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
"""Demo for the DITO paper.

Paper link: https://arxiv.org/abs/2310.00161.

This demo takes a sample image, and produces detections using a pretrained
DITO model.


"""

from collections.abc import Sequence
import functools

from absl import app
from absl import flags
from absl import logging
from demo_utils import input_utils
from demo_utils import vis_utils
import jax
import numpy as np
from PIL import Image
import tensorflow as tf
import tqdm
from utils import clip_utils


_DEMO_IMAGE_NAME = flags.DEFINE_string('demo_image_name', 'citrus.jpg',
                                       'The image file name under data/.')
_MODEL = flags.DEFINE_enum('model', 'vit-large', ['vit-large'],
                           'DITO model size to use.')
_MODEL_NAME = 'dito'
_CATEGORY_NAME_STRING = flags.DEFINE_string(
    'category_name_string', '',
    'Comma separated list of categories, e.g. "person, car, oven".')
_MAX_BOXES_TO_DRAW = flags.DEFINE_integer('max_boxes_to_draw', 25,
                                          'Max number of boxes to draw.')
_MAX_NUM_CLS = flags.DEFINE_integer('max_num_classes', 1204,
                                    'Max number of classes users can input.')
_MIN_SCORE_THRESH = flags.DEFINE_float('min_score_thresh', 0.15,
                                       'Min score threshold.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  demo_image_path = f'./data/{_DEMO_IMAGE_NAME.value}'
  output_image_path = demo_image_path.replace('data', 'output')
  output_image_path = (
      output_image_path[:-4]
      + f'_{_MODEL_NAME}_{_MODEL.value}'
      + output_image_path[-4:]
  )
  with tf.io.gfile.GFile(demo_image_path, 'rb') as f:
    np_image = np.array(Image.open(f))

  if _CATEGORY_NAME_STRING.value:
    # Parse string.
    categories = _CATEGORY_NAME_STRING.value.split(',')
  else:
    # Use default text prompts.
    try:
      categories = input_utils.category_dict[_DEMO_IMAGE_NAME.value]
    except KeyError:
      raise KeyError(
          'Default categories do not exist. Please specify!'
      ) from None

  class_clip_features = []
  logging.info('Computing custom category text embeddings.')
  tokenizer_dir = './dito/checkpoints/tokenizer/dito_tokenizer.model'
  tokenizer = clip_utils.get_tokenizer(tokenizer_dir)
  text_model_dir = './dito/checkpoints/text_model/'
  text_model = tf.saved_model.load(text_model_dir)
  for cls_name in tqdm.tqdm(categories, total=len(categories)):
    cls_feat = clip_utils.tokenize_pad_fn(tokenizer, text_model, cls_name)
    class_clip_features.append(cls_feat)

  logging.info('Preparing input data.')
  text_embeddings = np.concatenate(class_clip_features, axis=0)
  embed_path = './dito/embeddings/bg_empty_embed.npy'
  background_embedding, empty_embeddings = np.load(embed_path)
  background_embedding = background_embedding[np.newaxis, Ellipsis]
  empty_embeddings = empty_embeddings[np.newaxis, Ellipsis]
  tile_empty_embeddings = np.tile(
      empty_embeddings, (_MAX_NUM_CLS.value - len(categories) - 1, 1)
  )
  # Concatenate 'background' and 'empty' embeddings.
  text_embeddings = np.concatenate(
      (background_embedding, text_embeddings, tile_empty_embeddings), axis=0
  )
  text_embeddings = text_embeddings[np.newaxis, Ellipsis]
  # Parse the image data.
  parser_fn = input_utils.get_rovit_parser()
  data = parser_fn({'image': np_image, 'source_id': np.array([0])})
  np_data = jax.tree.map(lambda x: x.numpy()[np.newaxis, Ellipsis], data)
  np_data['text'] = text_embeddings
  np_data['image'] = np_data.pop('images')
  labels = np_data.pop('labels')
  image = np_data['image']

  logging.info('Loading saved model.')
  saved_model_dir = f'./dito/checkpoints/{_MODEL.value}'
  model = tf.saved_model.load(saved_model_dir)

  logging.info('Computing forward pass.')
  output = model(np_data)

  logging.info('Preparing visualization.')
  id_mapping = {(i + 1): c for i, c in enumerate(categories)}
  id_mapping[0] = 'background'
  for k in range(len(categories) + 2, _MAX_NUM_CLS.value):
    id_mapping[k] = 'empty'
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

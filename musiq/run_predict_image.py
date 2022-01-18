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

"""Run inference on a single image with a MUSIQ checkpoint."""

import collections
import io
from typing import Dict, Sequence, Text, TypeVar

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf

import musiq.model.multiscale_transformer as model_mod
import musiq.model.preprocessing as pp_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt_path', '', 'Path to checkpoint.')
flags.DEFINE_string('image_path', '', 'Path to input image.')
flags.DEFINE_integer(
    'num_classes', 1,
    'Number of scores to predict. 10 for AVA and 1 for the other datasets.')

# Image preprocessing config.
_PP_CONFIG = {
    'patch_size': 32,
    'patch_stride': 32,
    'hse_grid_size': 10,
    # The longer-side length for the resized variants.
    'longer_side_lengths': [224, 384],
    # -1 means using all the patches from the full-size image.
    'max_seq_len_from_original_res': -1,
}

# Model backbone config.
_MODEL_CONFIG = {
    'hidden_size': 384,
    'representation_size': None,
    'resnet_emb': {
        'num_layers': 5
    },
    'transformer': {
        'attention_dropout_rate': 0,
        'dropout_rate': 0,
        'mlp_dim': 1152,
        'num_heads': 6,
        'num_layers': 14,
        'num_scales': 3,
        'spatial_pos_grid_size': 10,
        'use_scale_emb': True,
        'use_sinusoid_pos_emb': False,
    }
}

T = TypeVar('T')  # Declare type variable


def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  This function is useful to analyze checkpoints that are saved by our programs
  without need to access the exact source code of the experiment. In particular,
  it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
  subtree of parameters.

  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.

  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if '/' not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split('/', 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def prepare_image(image_path, pp_config):
  """Processes image to multi-scale representation.

  Args:
    image_path: input image path.
    pp_config: image preprocessing config.

  Returns:
    An array representing image patches and input position annotations.
  """

  with tf.compat.v1.gfile.FastGFile(image_path, 'rb') as f:
    encoded_str = f.read()

  data = dict(image=tf.constant(encoded_str))
  pp_fn = pp_lib.get_preprocess_fn(**pp_config)
  data = pp_fn(data)
  image = data['image']
  # Shape (1, length, dim)
  image = tf.expand_dims(image, axis=0)
  image = image.numpy()
  return image


def run_model_single_image(model_config, num_classes, pp_config, params,
                           image_path):
  """Runs the model.

  Args:
    model_config: the parameters used in building the model backbone.
    num_classes: number of outputs. 1 for single mos prediction.
    pp_config: image preprocessing config.
    params: model parameters loaded from checkpoint.
    image_path: input image path.

  Returns:
    Model prediction for MOS score.
  """
  image = prepare_image(image_path, pp_config)
  model = model_mod.Model.partial(
      num_classes=num_classes, train=False, **model_config)
  logits = model.call(params, image)
  preds = logits
  if num_classes > 1:
    preds = jax.nn.softmax(logits)
  score_values = jnp.arange(1, num_classes + 1, dtype=np.float32)
  preds = jnp.sum(preds * score_values, axis=-1)
  return preds[0]


def get_params_and_config(ckpt_path):
  """Returns (model config, preprocessing config, model params from ckpt)."""
  model_config = ml_collections.ConfigDict(_MODEL_CONFIG)
  pp_config = ml_collections.ConfigDict(_PP_CONFIG)

  with tf.compat.v1.gfile.FastGFile(ckpt_path, 'rb') as f:
    data = f.read()
  values = np.load(io.BytesIO(data))
  params = recover_tree(*zip(*values.items()))
  params = params['opt']['target']
  if not model_config.representation_size:
    params['pre_logits'] = {}

  return model_config, pp_config, params


def main(_):
  model_config, pp_config, params = get_params_and_config(FLAGS.ckpt_path)
  pred_mos = run_model_single_image(model_config, FLAGS.num_classes, pp_config,
                                    params, FLAGS.image_path)
  print('============== Precited MOS:', pred_mos)


if __name__ == '__main__':
  app.run(main)

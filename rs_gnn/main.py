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

"""RS-GNN Implementation."""

import os
# The experiments in the paper were done when lazy_rng's default value was false
# Since then, the default value has changed to true.
# Setting it back to false for consistency.
os.environ['FLAX_LAZY_RNG'] = 'false'
# pylint: disable=g-import-not-at-top
import types

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

from rs_gnn import data_utils
from rs_gnn import trainer

_DATA_PATH = flags.DEFINE_string('data_path', '', 'Directory of the data.')
_GCN_C_HID_DIM = flags.DEFINE_integer('gcn_c_hid_dim', 32,
                                      'GCN-C hidden dimension.')
_RSGNN_HID_DIM = flags.DEFINE_integer('rsgnn_hid_dim', 512,
                                      'RS-GNN hidden dimension.')
_SEED = flags.DEFINE_integer('seed', 42, 'Random seed.')
_RSGNN_EPOCHS = flags.DEFINE_integer('rsgnn_epochs', 2000,
                                     'Number of RS-GNN epochs.')
_GCN_C_EPOCHS = flags.DEFINE_integer('gcn_c_epochs', 1000,
                                     'Number of epochs for GCN-C.')
_NUM_REPS_MULTIPLIER = flags.DEFINE_integer(
    'num_reps_multiplier', 2, 'num_reps = num_class * num_reps_multiplier.')
_VALID_EACH = flags.DEFINE_integer('valid_each', 10, 'Validate each k epochs.')
_NUM_VALID_NODES = flags.DEFINE_integer('num_valid_nodes', 500,
                                        'Number of validation set nodes.')
_LR = flags.DEFINE_float('lr', 0.001, 'Learning rate.')
_DROP_RATE = flags.DEFINE_float('drop_rate', 0.5, 'Dropout probability.')
_GCN_C_W_DECAY = flags.DEFINE_float('gcn_c_w_decay', 5e-4,
                                    'Weight decay for the GCN.')
_LAMBDA = flags.DEFINE_float('lambda_', 0.001, 'Hyperparam for JointDGI loss.')
_DATASET = flags.DEFINE_enum('dataset', 'cora', ['cora'], '')


def create_splits(train_nodes, num_nodes):
  train_idx = np.array([False] * num_nodes)
  train_idx[train_nodes] = True
  valid_nodes = np.random.choice(np.where(np.logical_not(train_idx))[0],
                                 _NUM_VALID_NODES.value, replace=False)
  valid_idx = np.array([False] * num_nodes)
  valid_idx[valid_nodes] = True
  test_idx = np.logical_not(np.logical_or(train_idx, valid_idx))
  return types.SimpleNamespace(train=jnp.array(train_idx),
                               valid=jnp.array(valid_idx),
                               test=jnp.array(test_idx))


def get_rsgnn_flags(num_classes):
  return types.SimpleNamespace(
      hid_dim=_RSGNN_HID_DIM.value,
      epochs=_RSGNN_EPOCHS.value,
      num_classes=num_classes,
      num_reps=_NUM_REPS_MULTIPLIER.value * num_classes,
      valid_each=_VALID_EACH.value,
      lr=_LR.value,
      lambda_=_LAMBDA.value)


def get_gcn_c_flags(num_classes):
  return types.SimpleNamespace(
      hid_dim=_GCN_C_HID_DIM.value,
      epochs=_GCN_C_EPOCHS.value,
      num_classes=num_classes,
      valid_each=_VALID_EACH.value,
      lr=_LR.value,
      drop_rate=_DROP_RATE.value,
      w_decay=_GCN_C_W_DECAY.value)


def main(unused_args):
  """Runs node selector, receives selected nodes, trains GCN."""
  np.random.seed(_SEED.value)
  key = jax.random.PRNGKey(_SEED.value)
  graph, labels, num_classes = data_utils.create_jraph(_DATA_PATH.value,
                                                       _DATASET.value)
  rsgnn_flags = get_rsgnn_flags(num_classes)
  selected = trainer.train_rsgnn(rsgnn_flags, graph, key)
  key, gcn_key = jax.random.split(key)
  splits = create_splits(selected, graph.n_node[0])
  gcn_c_flags = get_gcn_c_flags(num_classes)
  gcn_accu = trainer.train_gcn(gcn_c_flags, graph, labels, gcn_key, splits)
  print(f'GCN Test Accuracy: {gcn_accu}')


if __name__ == '__main__':
  app.run(main)

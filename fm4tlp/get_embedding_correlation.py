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

r"""Script to study correlation between memory embeddings and structural/positional embeddings.

command for an example run:

python google_research/fm4tlp/get_embedding_correlation -- \
  --data=tgbl_wiki \
  --root_dir=./data \
  --output_subdir=palowitch_factored_test \
  --model_name=tgn \
  --train_group=cc-subgraph \
  --val_group=cc-subgraph \
  --experiment_name=transductive \
  --run_id=0 \
  --pos_enc_dim=4
"""

import datetime
import itertools
import os
import timeit

from absl import app
from absl import flags
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import tensorflow.compat.v1 as tf
import torch
import tqdm

from fm4tlp import model_config as model_config_lib
from fm4tlp.models import all_models
from fm4tlp.models import model_template
from fm4tlp.modules import early_stopping
from fm4tlp.utils import dataset_pyg_transfer
from fm4tlp.utils import structural_feature_helper
from fm4tlp.utils import utils


_DATA = flags.DEFINE_string(
    'data',
    None,
    help='tgbl_flight or tgbl_comment',
    required=True,
)

_TRAIN_GROUP = flags.DEFINE_string(
    'train_group',
    None,
    help='Continent or community',
    required=True,
)

_VAL_GROUP = flags.DEFINE_string(
    'val_group',
    None,
    help='Continent or community',
    required=True,
)

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    None,
    'Root directory with processed files for transfer learning',
    required=True,
)

_OUTPUT_SUBDIR = flags.DEFINE_string(
    'output_subdir',
    None,
    (
        'If it doesn\t exist already, `output_subdir` will be created under '
        '`root_dir`/`data`, and all model output and saved models will be '
        'written here.'
    ),
    required=True,
)

_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    None,
    'A model name from one of the model config pbtxt files in models/configs.',
    required=True,
)

_SEED = flags.DEFINE_integer('seed', 12345, 'Seed for random number generator.')

_RUN_ID = flags.DEFINE_integer('run_id', 0, 'Index of the run')

_EXPECT_GPU = flags.DEFINE_bool(
    'expect_gpu',
    False,
    'Used to check if CUDA is available when GPU is requested.',
)

_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name',
    None,
    'Name of the experiment. All model/results files [will use this as a'
    ' filename suffix. If unspecified, set to current timestamp at binary'
    ' start.',
    required=True,
)

_POS_ENC_DIM = flags.DEFINE_integer(
    'pos_enc_dim', 4, 'Positional encoding dimension.'
)


def main(_):

  # ==========
  # ==========
  # ==========

  # Compute current time
  current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  experiment_name = _EXPERIMENT_NAME.value or current_time

  if _EXPECT_GPU.value:
    assert torch.cuda.is_available()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Start...
  start_overall = timeit.default_timer()
  # ==========

  # data loading
  dataset_root = os.path.join(_ROOT_DIR.value, 'datasets', _DATA.value)
  train_dataset = dataset_pyg_transfer.PyGLinkPropPredDataset(
      name=_DATA.value,
      group=_TRAIN_GROUP.value,
      mode='train',
      root=dataset_root,
  )

  train_data = train_dataset.get_TemporalData()

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, _DATA.value + '_total_count.csv'), 'r'
  ) as f:
    node_count = pd.read_csv(f)
  total_nodes = node_count['num_nodes'][0]

  assert train_data.src and train_data.dst
  all_nodes = set(train_data.src.numpy()).union(set(train_data.dst.numpy()))

  G = nx.Graph()
  G.add_edges_from(zip(train_data.src.numpy(), train_data.dst.numpy()))

  # define model
  model_config = model_config_lib.get_model_config(_MODEL_NAME.value)
  model: model_template.TlpModel = getattr(
      all_models, model_config.model_class
  )(
      model_config=model_config,
      total_num_nodes=total_nodes,
      raw_message_size=train_data.msg.size(-1),
      device=device,
      learning_rate=0,
  )

  model_name = '_'.join(
      [model.model_name, _DATA.value, _TRAIN_GROUP.value, _VAL_GROUP.value]
  )

  run_directory = os.path.join(
      _ROOT_DIR.value, 'experiments', _OUTPUT_SUBDIR.value, _DATA.value
  )

  # set the seed for deterministic results...
  torch.manual_seed(_RUN_ID.value + _SEED.value)
  utils.set_random_seed(_RUN_ID.value + _SEED.value)

  # define an early stopper
  save_model_dir = os.path.join(run_directory, 'saved_models')
  save_model_id = (
      f'{model_name}_{_SEED.value}_{_RUN_ID.value}_{experiment_name}'
  )
  early_stopper = early_stopping.EarlyStopMonitor(
      save_model_dir=save_model_dir,
      save_model_id=save_model_id,
  )
  print('INFO: done setting up loading of saved models.')

  # ==================================================== Test
  # first, load the best model
  early_stopper.load_checkpoint(model)

  if model.has_memory:
    ## Load memory embeddings
    memory_embeddings = dict()
    for node in tqdm.tqdm(all_nodes):
      z = model.get_memory_embeddings(torch.tensor([node]))
      memory_embeddings[node] = z.detach().numpy()

    ## Get structural/positional embeddings
    topological_feature_dict = (
        structural_feature_helper.generate_graph_features(G)
    )
    pos_feature_dict = structural_feature_helper.lap_positional_encoding(
        G, _POS_ENC_DIM.value
    )
    pe_feature_dict = structural_feature_helper.init_positional_encoding(
        G, _POS_ENC_DIM.value
    )

    memory_distances = []
    structural_distances = []
    pos_distances = []
    pe_distances = []

    node_pairs = list(itertools.product(all_nodes, all_nodes))

    # Get pairwise distances.
    # Can be improved by chunking the node_pairs and applying difference-based
    # formula for correlation.
    # Alternatively, randomly sample N node pairs.
    for node_pair in tqdm.tqdm(node_pairs):
      node1 = node_pair[0]
      node2 = node_pair[1]
      memory_distances.append(
          np.linalg.norm(memory_embeddings[node1] - memory_embeddings[node2])
      )
      structural_distances.append(
          np.linalg.norm(
              topological_feature_dict[node1] - topological_feature_dict[node2]
          )
      )
      pos_distances.append(
          np.linalg.norm(pos_feature_dict[node1] - pos_feature_dict[node2])
      )
      pe_distances.append(
          np.linalg.norm(pe_feature_dict[node1] - pe_feature_dict[node2])
      )

    res_pearson_structural = scipy.stats.pearsonr(
        structural_distances, memory_distances
    )
    res_spearman_structural = scipy.stats.spearmanr(
        structural_distances, memory_distances
    )
    res_pearson_pos = scipy.stats.pearsonr(pos_distances, memory_distances)
    res_spearman_pos = scipy.stats.spearmanr(pos_distances, memory_distances)
    res_pearson_pe = scipy.stats.pearsonr(pe_distances, memory_distances)
    res_spearman_pe = scipy.stats.spearmanr(pe_distances, memory_distances)
    print(
        'Pearson and Spearman correlation of pairwise node distances between'
        f' memory and structural embeddings: {res_pearson_structural[0]: .4f},'
        f' {res_spearman_structural[0]: .4f}'
    )
    print(
        'Pearson and Spearman correlation of pairwise node distances between'
        f' memory and positional embeddings: {res_pearson_pos[0]: .4f},'
        f' {res_spearman_pos[0]: .4f}'
    )
    print(
        'Pearson and Spearman correlation of pairwise node distances between'
        f' memory and PE embeddings: {res_pearson_pe[0]: .4f},'
        f' {res_spearman_pe[0]: .4f}'
    )


if __name__ == '__main__':
  app.run(main)

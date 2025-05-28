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

r"""Generate and store structural features for a dataset split, with Beam.

Command for an example run:

python google_research/fm4tlp/generate_structural_features_pipeline \
  --data='tgbl_wiki;cc-subgraph,tgbl_coin;cc-subgraph,tgbl_review;cc-subgraph,tgbl_comment;cc-subgraph,tgbl_flight;AS,tgbl_flight;AF,tgbl_flight;EU' \
  --root_dir=./data \
  --only_basic_features=True \
  --structural_feature_file_tag='basic_features' \
  --bs=200 \
  --pos_enc_dim=4 \
  --aggregation_window_frac=0.01
"""

from collections.abc import Sequence
import dataclasses
import os
import pickle
import sys

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import networkx as nx
import tensorflow.compat.v1 as tf

from fm4tlp.utils import structural_feature_helper



if not any(['py.runner' in m for m in list(sys.modules.keys())]):
  beam_runner = beam.runners.DirectRunner


_DATA = flags.DEFINE_list(
    'data', None, 'Comma-separated list of datasets.', required=True
)

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    None,
    'Root directory with processed files for transfer learning',
    required=True,
)

_BATCH_SIZE = flags.DEFINE_integer('bs', 200, 'Batch size.')

_POS_ENC_DIM = flags.DEFINE_integer(
    'pos_enc_dim', 4, 'Positional encoding dimension.'
)

_AGGREGATION_WINDOW_FRAC = flags.DEFINE_float(
    'aggregation_window_frac',
    0.01,
    'Fraction of the total time interval to use for aggregation.',
)

_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', 64, 'Number of workers to use for parallel processing.'
)

_OVERWRITE_FEATURES = flags.DEFINE_bool(
    'overwrite_features', False, 'Whether to overwrite features.'
)

_COMPUTE_VAC_4 = flags.DEFINE_bool(
    'compute_vac_4', False, 'Whether to compute Vertex Automorphism Count 4.'
)

_ONLY_BASIC_FEATURES = flags.DEFINE_bool(
    'only_basic_features',
    False,
    'If true, this job will only compute the graph-level features (no'
    ' positional or automorphism features).',
)

_STRUCTURAL_FEATURE_FILE_TAG = flags.DEFINE_string(
    'structural_feature_file_tag',
    '',
    'Structural feature files are formatted like'
    ' {data}_{community}_{split}{tag_str}_structural_features_{batch_index}.pkl'
    ' where tag_str is equivalent to _{structural_feature_file_tag}.',
)


_ID_TYPES = [
    'cycle_graph',
    'path_graph',
    'complete_graph',
    'binomial_tree',
    'star_graph',
    'nonisomorphic_trees',
]

_REDUCED_ID_TYPES = [
    'cycle_graph',
    'path_graph',
    'complete_graph',
]


@dataclasses.dataclass(frozen=True)
class FeaturesSpec:
  batch_index: int = dataclasses.field(default_factory=int)
  dataset: str = dataclasses.field(default_factory=str)
  community: str = dataclasses.field(default_factory=str)
  split: str = dataclasses.field(default_factory=str)


class SaveBatchStats(beam.DoFn):
  """Beam stage to save structural features for a batch of subgraphs."""

  def __init__(
      self,
      root_dir,
      vac3,
      vac4,
      compute_vac4 = False,
      overwrite = False,
      only_basic_features = False,
      structural_feature_file_tag = '',
  ):
    self._root_dir = root_dir
    self._vac3 = vac3
    self._vac4 = vac4
    self._compute_vac4 = compute_vac4
    self._overwrite = overwrite
    self._only_basic_features = only_basic_features
    self._structural_feature_file_tag = structural_feature_file_tag

  def process(self, features_spec):
    # Set up filenames.
    dataset = features_spec.dataset
    community = features_spec.community
    split = features_spec.split
    batch_index = features_spec.batch_index
    dataset_root = os.path.join(self._root_dir, 'datasets', dataset)
    batches_root = os.path.join(dataset_root, 'structural_features_by_batch')
    filename_prefix = dataset + '_' + community + '_' + split
    subgraph_filename = filename_prefix + f'_subgraph_{batch_index}.graphml'
    structural_filename_prefix = filename_prefix
    if self._structural_feature_file_tag:
      structural_filename_prefix += '_' + self._structural_feature_file_tag
    features_filename = (
        structural_filename_prefix + f'_structural_features_{batch_index}.pkl'
    )
    dataset_counter_prefix = f'{dataset}-{community}-{split}'

    # Compute features.
    if tf.io.gfile.exists(features_filename) and not self._overwrite:
      beam.metrics.Metrics.counter(
          'ComputeFeatures', dataset_counter_prefix + '-features-already-exist'
      ).inc()
      return
    else:
      beam.metrics.Metrics.counter(
          'ComputeFeatures', dataset_counter_prefix + '-features-compute-start'
      ).inc()
      # Load subgraph.
      with tf.io.gfile.GFile(
          os.path.join(batches_root, subgraph_filename), 'rb'
      ) as f:
        subgraph = nx.read_graphml(f)

      topological_feat_dict_pos = (
          structural_feature_helper.generate_graph_features(subgraph)
      )
      batch_stats = dict()
      batch_stats['topological_feats'] = topological_feat_dict_pos
      if not self._only_basic_features:
        batch_stats['laplace_pos_embed'] = (
            structural_feature_helper.lap_positional_encoding(
                subgraph, _POS_ENC_DIM.value
            )
        )
        batch_stats['init_pos_embed'] = (
            structural_feature_helper.init_positional_encoding(
                subgraph, _POS_ENC_DIM.value
            )
        )
        batch_stats['vac3'] = self._vac3.get_all_automorphism_counts(subgraph)
        if self._compute_vac4:
          batch_stats['vac4'] = self._vac4.get_all_automorphism_counts(subgraph)
      with tf.io.gfile.GFile(
          os.path.join(batches_root, features_filename), 'wb'
      ) as f:
        pickle.dump(batch_stats, f)
      beam.metrics.Metrics.counter(
          'ComputeFeatures', dataset_counter_prefix + '-features-compute-done'
      ).inc()


def make_pipeline(
    dataset_specs,
    root_dir,
    vac3,
    vac4,
    compute_vac4 = False,
    overwrite = False,
    only_basic_features = False,
    structural_feature_file_tag = '',
):
  """Makes a Beam pipeline to compute and save structural features."""

  def pipeline(root):
    _ = (
        root
        | 'CreateDatasetSpecs' >> beam.Create(dataset_specs)
        | 'SaveBatchStats'
        >> beam.ParDo(
            SaveBatchStats(
                root_dir,
                vac3,
                vac4,
                compute_vac4,
                overwrite,
                only_basic_features,
                structural_feature_file_tag,
            )
        )
    )

  return pipeline


def main(_):

  # Make giant list of all subgraphs that need to be processed.
  dataset_specs = []
  for data_string in _DATA.value:
    try:
      dataset, community = data_string.split(';')
    except ValueError:
      raise ValueError(
          'Data string must be of the form `dataset;community`. Got'
          f' {data_string}'
      ) from None
    dataset_root = os.path.join(_ROOT_DIR.value, 'datasets', dataset)
    for split in ['train', 'val', 'test']:
      batches_root = os.path.join(dataset_root, 'structural_features_by_batch')
      if not tf.io.gfile.exists(batches_root):
        tf.io.gfile.makedirs(batches_root)
      filename_prefix = dataset + '_' + community + '_' + split
      subgraph_filenames = tf.io.gfile.glob(
          os.path.join(batches_root, filename_prefix + '_subgraph_*')
      )
      logging.info(
          'Found %d subgraphs for %s_%s_%s',
          len(subgraph_filenames),
          dataset,
          community,
          split,
      )
      for subgraph_filename in subgraph_filenames:
        batch_index = int(subgraph_filename.split('_')[-1].split('.')[0])
        dataset_specs.append(
            FeaturesSpec(batch_index, dataset, community, split)
        )

  vac3 = structural_feature_helper.VertexAutomorphismCounter(
      k=3, id_types=_REDUCED_ID_TYPES
  )
  vac4 = structural_feature_helper.VertexAutomorphismCounter(k=4)

  pipeline_result = beam_runner().run(
      make_pipeline(
          dataset_specs,
          _ROOT_DIR.value,
          vac3,
          vac4,
          _COMPUTE_VAC_4.value,
          _OVERWRITE_FEATURES.value,
          _ONLY_BASIC_FEATURES.value,
          _STRUCTURAL_FEATURE_FILE_TAG.value,
      )
  )
  pipeline_result.wait_until_finish()


if __name__ == '__main__':
  app.run(main)

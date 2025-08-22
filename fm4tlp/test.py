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

r"""Dynamic Link Prediction with a TGN model with Early Stoppin Test.

Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:

python google_research/fm4tlp/test -- \
  --data=tgbl_wiki \
  --root_dir=./data \
  --output_subdir=aychatterjee_testrun \
  --model_name=tgn_structmap \
  --train_group=cc-subgraph \
  --val_group=cc-subgraph \
  --test_group=cc-subgraph \
  --update_mem \
  --warmstart \
  --warmstart_update_model \
  --warmstart_batch_fraction=0.2 \
  --experiment_name=transductive \
  --run_id=0 \
  --noexpect_gpu \
  --num_workers=16 \
  --alsologtostderr
"""

import datetime
import os
import timeit

from absl import app
from absl import flags
import pandas as pd
import tensorflow.compat.v1 as tf
import torch
from torch_geometric import loader as torch_geo_data_loader

from fm4tlp import model_config as model_config_lib
from fm4tlp.models import all_models
from fm4tlp.models import model_template
from fm4tlp.modules import early_stopping
from fm4tlp.modules import neighbor_loader
from fm4tlp.utils import dataset_pyg_transfer
from fm4tlp.utils import evaluate
from fm4tlp.utils import train_test_helper
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

_TEST_GROUP = flags.DEFINE_string(
    'test_group',
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

_UPDATE_MEM = flags.DEFINE_boolean(
    'update_mem', True, 'Flag to update memory during test'
)

_WARMSTART = flags.DEFINE_boolean('warmstart', False, 'Flag to do warmstart')

_WARMSTART_UPDATE_MODEL = flags.DEFINE_boolean(
    'warmstart_update_model',
    False,
    'Flag to decide whether to update model parameters during warmstart',
)

_WARMSTART_BATCH_FRACTION = flags.DEFINE_float(
    'warmstart_batch_fraction', 0.2, 'Fraction of batches used in warmstart'
)

_BATCH_SIZE = flags.DEFINE_integer('bs', 200, 'Batch size.')

_K_VALUE = flags.DEFINE_integer(
    'k_value', 10, 'k_value for computing ranking metrics'
)

_SEED = flags.DEFINE_integer('seed', 12345, 'Seed for random number generator.')

_RUN_ID = flags.DEFINE_integer('run_id', 0, 'Index of the run')

_NUM_NEIGHBORS = flags.DEFINE_integer(
    'num_neighbors', 10, 'Number of neighbors for neighborhood sampling.'
)

_EXPECT_GPU = flags.DEFINE_bool(
    'expect_gpu',
    False,
    'Used to check if CUDA is available when GPU is requested.',
)

_LOSS_LOGGING_FREQUENCY = flags.DEFINE_integer(
    'loss_logging_frequency',
    1000,
    'Size of step interval at which to log loss values.',
)

_USE_XM = flags.DEFINE_bool('use_xm', False, 'Whether to use XManager.')

_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name',
    None,
    'Name of the experiment. All model/results files [will use this as a'
    ' filename suffix. If unspecified, set to current timestamp at binary'
    ' start.',
    required=False,
)

_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', 64, 'Number of workers to use for parallel processing.'
)

_LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')

_STRUCTURAL_FEATURE_FILE_TAG = flags.DEFINE_string(
    'structural_feature_file_tag',
    '',
    'Structural feature files are formatted like'
    ' {data}_{community}_{split}{tag_str}_structural_features_{batch_index}.pkl'
    ' where tag_str is equivalent to _{structural_feature_file_tag}.',
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
  unused_start_overall = timeit.default_timer()
  # ==========

  # data loading
  dataset_root = os.path.join(_ROOT_DIR.value, 'datasets', _DATA.value)
  test_dataset = dataset_pyg_transfer.PyGLinkPropPredDataset(
      name=_DATA.value,
      group=_TEST_GROUP.value,
      mode='test',
      root=dataset_root,
  )

  test_data = test_dataset.get_TemporalData()
  test_data = test_data.to(device)
  metric = 'mrr'

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, _DATA.value + '_total_count.csv'), 'r'
  ) as f:
    airport_count = pd.read_csv(f)

  # Total nodes in all communities. Used in initializing the memory dict.
  total_nodes = airport_count['num_nodes'][0]

  warmstart_test_data, test_data_residual, batches_in_warmstart = (
      train_test_helper.split_for_warmstart_batches(
          test_data,
          batch_fraction=_WARMSTART_BATCH_FRACTION.value,
          batch_size=_BATCH_SIZE.value,
      )
  )

  if _WARMSTART.value:
    warmstart_test_loader = torch_geo_data_loader.TemporalDataLoader(
        warmstart_test_data, batch_size=_BATCH_SIZE.value
    )
    min_dst_idx_warmstart, max_dst_idx_warmstart = int(
        warmstart_test_data.dst.min()
    ), int(warmstart_test_data.dst.max())
  else:
    # Do we need all these defined here?
    unused_warmstart_test_data = torch.empty(0, dtype=torch.long, device=device)
    warmstart_test_loader = None
    min_dst_idx_warmstart, max_dst_idx_warmstart = None, None

  test_loader = torch_geo_data_loader.TemporalDataLoader(
      test_data_residual, batch_size=_BATCH_SIZE.value
  )

  # neighborhood sampler
  last_neighbor_loader = neighbor_loader.LastNeighborLoader(
      total_nodes, size=_NUM_NEIGHBORS.value, device=device
  )

  model_config = model_config_lib.get_model_config(_MODEL_NAME.value)
  test_feature_dim = 0
  test_structural_features = {}
  structural_feats_list = [
      'topological_feats',
      'laplace_pos_embed',
      'init_pos_embed',
      'vac3',
  ]
  if model_config.structural_mapping_hidden_dim:
    # load structural features
    test_structural_features, test_feature_dim, _, _ = (
        utils.load_structural_features(
            dataset_root,
            _DATA.value,
            _TEST_GROUP.value,
            'test',
            _NUM_WORKERS.value,
            structural_feature_file_tag=_STRUCTURAL_FEATURE_FILE_TAG.value,
            structural_feats_list=structural_feats_list,
        )
    )

  # define model
  model: model_template.TlpModel = getattr(
      all_models, model_config.model_class
  )(
      model_config=model_config,
      total_num_nodes=total_nodes,
      raw_message_size=test_data.msg.size(-1),
      device=device,
      learning_rate=_LEARNING_RATE.value,
      structural_feature_dim=test_feature_dim,
  )

  model_name = '_'.join(
      [model.model_name, _DATA.value, _TRAIN_GROUP.value, _VAL_GROUP.value]
  )

  evaluator = evaluate.Evaluator(name=_DATA.value)
  neg_sampler = test_dataset.negative_sampler

  # for saving the results...
  run_directory = os.path.join(
      _ROOT_DIR.value, 'experiments', _OUTPUT_SUBDIR.value, _DATA.value
  )
  results_path = os.path.join(run_directory, 'results', model_name)
  if not tf.io.gfile.isdir(results_path):
    print('INFO: Create directory {}'.format(results_path))
    tf.io.gfile.makedirs(results_path)

  metrics_logger = evaluate.MetricsLogger()

  results_filename = (
      f'{results_path}/{experiment_name}_results_test_{_TEST_GROUP.value}.json'
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

  # loading the test negative samples
  test_dataset.load_test_ns()

  # Warmstart
  if _WARMSTART.value:
    warmstart_performance_lists = train_test_helper.warmstart(
        model,
        test_data,
        warmstart_test_loader,
        device,
        min_dst_idx_warmstart,
        max_dst_idx_warmstart,
        metric,
        last_neighbor_loader,
        evaluator,
        metrics_logger,
        update_model=_WARMSTART_UPDATE_MODEL.value,
        structural_feats_list=structural_feats_list,
        structural_features=test_structural_features,
    )
    print('INFO: Warmstart done.')
    warmstart_loss = pd.DataFrame()
    warmstart_loss['loss'] = warmstart_performance_lists.loss
    warmstart_loss['model_loss'] = warmstart_performance_lists.model_loss
    warmstart_loss['perf'] = warmstart_performance_lists.perf
    warmstart_loss['auc'] = warmstart_performance_lists.auc
    with tf.io.gfile.GFile(
        os.path.join(
            results_path,
            f'{experiment_name}_test_{_TEST_GROUP.value}_warmstart_loss.csv',
        ),
        'w',
    ) as f:
      warmstart_loss.to_csv(f, index=False)

  # final testing
  start_test = timeit.default_timer()
  (
      perf_metric_test,
      auc,
      test_performance_lists,
  ) = train_test_helper.test(
      model,
      device,
      evaluator,
      last_neighbor_loader,
      test_data,
      metric,
      test_loader,
      neg_sampler,
      split_mode='test',
      update_memory=_UPDATE_MEM.value,
      warmstart_batch_id=batches_in_warmstart,
      structural_feats_list=structural_feats_list,
      structural_features=test_structural_features,
  )

  print('INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ')
  print(f'\tTest: {metric}: {perf_metric_test: .4f}')
  print(f'\tTest AUC: {auc: .4f}')
  test_time = timeit.default_timer() - start_test
  print(f'\tTest: Elapsed Time (s): {test_time: .4f}')

  test_loss = pd.DataFrame()
  test_loss['loss'] = test_performance_lists.loss
  test_loss['model_loss'] = test_performance_lists.model_loss
  test_loss['perf'] = test_performance_lists.perf
  test_loss['auc'] = test_performance_lists.auc
  with tf.io.gfile.GFile(
      os.path.join(
          results_path, f'{experiment_name}_test_{_TEST_GROUP.value}_loss.csv'
      ),
      'w',
  ) as f:
    test_loss.to_csv(f, index=False)

  utils.save_results(
      {
          'model': model_name,
          'data': _DATA.value,
          'run': _RUN_ID.value,
          'seed': _SEED.value,
          f'test {metric}': perf_metric_test,
          'test auc': auc,
          'test_time': test_time,
      },
      results_filename,
  )


if __name__ == '__main__':
  app.run(main)

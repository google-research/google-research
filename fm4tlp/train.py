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

r"""Dynamic Link Prediction with a TGN model with Early Stopping Train and Validation.

Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:

python google_research/fm4tlp/train  -- \
  --data=tgbl_wiki \
  --num_epoch=1 \
  --root_dir=./data \
  --output_subdir=aychatterjee_testrun \
  --model_name=tgn_structmap \
  --train_group=cc-subgraph \
  --val_group=cc-subgraph \
  --update_mem \
  --warmstart \
  --warmstart_update_model \
  --warmstart_batch_fraction=0.2 \
  --noreset_memory \
  --noreset_nbd_loader \
  --experiment_name=transductive \
  --run_id=0 \
  --noexpect_gpu \
  --num_workers=16 \
"""

import datetime
import os
import timeit

from absl import app
from absl import flags
import pandas as pd
import tensorflow.compat.v1 as tf
import torch
from torch_geometric import data as torch_geo_data
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

_RESET_MEMORY = flags.DEFINE_boolean(
    'reset_memory', True, 'Flag to reset memory before validation'
)

_RESET_NBD_LOADER = flags.DEFINE_boolean(
    'reset_nbd_loader', True, 'Flag to reset neighbor loader before validation'
)

_UPDATE_MEM = flags.DEFINE_boolean(
    'update_mem', True, 'Flag to update memory during validation'
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

_NUM_EPOCH = flags.DEFINE_integer('num_epoch', 5, 'Number of epochs.')

_SEED = flags.DEFINE_integer('seed', 12345, 'Seed for random number generator.')

_TOLERANCE = flags.DEFINE_float(
    'tolerance', 1e-6, 'Tolerance for early stopping.'
)

_PATIENCE = flags.DEFINE_integer('patience', 5, 'Patience for early stopping.')

_RUN_ID = flags.DEFINE_integer('run_id', 0, 'Index of the run')

_NUM_NEIGHBORS = flags.DEFINE_integer(
    'num_neighbors', 10, 'Number of neighbors for neighborhood sampling.'
)

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
  if not _WARMSTART.value and not _RESET_NBD_LOADER.value:
    raise ValueError(
        'Neighbor loader needs to be reset while skipping warmstart.'
    )

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
  val_dataset = dataset_pyg_transfer.PyGLinkPropPredDataset(
      name=_DATA.value,
      group=_VAL_GROUP.value,
      mode='val',
      root=dataset_root,
  )
  train_data = train_dataset.get_TemporalData()
  train_data = train_data.to(device)
  val_data = val_dataset.get_TemporalData()
  val_data = val_data.to(device)
  metric = train_dataset.eval_metric
  data = torch_geo_data.TemporalData(
      src=torch.cat([train_data.src, val_data.src]),
      dst=torch.cat([train_data.dst, val_data.dst]),
      t=torch.cat([train_data.t, val_data.t]),
      msg=torch.cat([train_data.msg, val_data.msg]),
      y=torch.cat([train_data.y, val_data.y]),
  )

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, _DATA.value + '_total_count.csv'), 'r'
  ) as f:
    node_count = pd.read_csv(f)
  # Total nodes in all communities/continents.
  # Used in initializing the memory dict for transfer learning.
  total_nodes = node_count['num_nodes'][0]

  warmstart_val_data, val_data, batches_in_warmstart = (
      train_test_helper.split_for_warmstart_batches(
          val_data,
          batch_fraction=_WARMSTART_BATCH_FRACTION.value,
          batch_size=_BATCH_SIZE.value,
      )
  )

  if _WARMSTART.value:
    warmstart_val_loader = torch_geo_data_loader.TemporalDataLoader(
        warmstart_val_data, batch_size=_BATCH_SIZE.value
    )
    min_dst_idx_warmstart, max_dst_idx_warmstart = int(
        warmstart_val_data.dst.min()
    ), int(warmstart_val_data.dst.max())
  else:
    # Do we need all these defined here?
    warmstart_val_data = torch.empty(0, dtype=torch.long, device=device)
    warmstart_val_loader = None
    min_dst_idx_warmstart, max_dst_idx_warmstart = None, None

  train_loader = torch_geo_data_loader.TemporalDataLoader(
      train_data, batch_size=_BATCH_SIZE.value
  )
  val_loader = torch_geo_data_loader.TemporalDataLoader(
      val_data, batch_size=_BATCH_SIZE.value
  )

  # Ensure to only sample actual destination nodes as negatives.
  min_dst_idx, max_dst_idx = int(train_data.dst.min()), int(
      train_data.dst.max()
  )

  # neighborhood sampler
  last_neighbor_loader = neighbor_loader.LastNeighborLoader(
      total_nodes, size=_NUM_NEIGHBORS.value, device=device
  )

  model_config = model_config_lib.get_model_config(_MODEL_NAME.value)
  train_feature_dim = 0
  train_structural_features = {}
  val_structural_features = {}
  structural_feats_list = [
      'topological_feats',
      'laplace_pos_embed',
      'init_pos_embed',
      'vac3',
  ]
  train_feature_mean = []
  train_feature_std = []
  if model_config.structural_mapping_hidden_dim:
    # load structural features
    (
        train_structural_features,
        train_feature_dim,
        train_feature_mean,
        train_feature_std,
    ) = utils.load_structural_features(
        dataset_root,
        _DATA.value,
        _TRAIN_GROUP.value,
        'train',
        _NUM_WORKERS.value,
        structural_feature_file_tag=_STRUCTURAL_FEATURE_FILE_TAG.value,
        structural_feats_list=structural_feats_list,
    )
    val_structural_features, val_feature_dim, _, _ = (
        utils.load_structural_features(
            dataset_root,
            _DATA.value,
            _VAL_GROUP.value,
            'val',
            _NUM_WORKERS.value,
            structural_feature_file_tag=_STRUCTURAL_FEATURE_FILE_TAG.value,
            structural_feats_list=structural_feats_list,
        )
    )
    if train_feature_dim != val_feature_dim:
      raise ValueError(
          'Train and val feature dimensions are different: '
          f'{train_feature_dim} vs {val_feature_dim}'
      )

  # define model
  model: model_template.TlpModel = getattr(
      all_models, model_config.model_class
  )(
      model_config=model_config,
      total_num_nodes=total_nodes,
      raw_message_size=train_data.msg.size(-1),
      device=device,
      learning_rate=_LEARNING_RATE.value,
      structural_feature_dim=train_feature_dim,
      structural_feature_mean=train_feature_mean,
      structural_feature_std=train_feature_std,
  )

  MODEL_NAME = '_'.join(
      [model.model_name, _DATA.value, _TRAIN_GROUP.value, _VAL_GROUP.value]
  )

  print('==========================================================')
  print(
      f'=================*** {MODEL_NAME}: LinkPropPred:'
      f' {_DATA.value} ***============='
  )
  print('==========================================================')

  evaluator = evaluate.Evaluator(name=_DATA.value)
  neg_sampler = val_dataset.negative_sampler

  # for saving the results...
  run_directory = os.path.join(
      _ROOT_DIR.value, 'experiments', _OUTPUT_SUBDIR.value, _DATA.value
  )
  results_path = os.path.join(run_directory, 'results', MODEL_NAME)
  if not tf.io.gfile.isdir(results_path):
    print('INFO: Create directory {}'.format(results_path))
    tf.io.gfile.makedirs(results_path)

  start_run = timeit.default_timer()

  results_filename = f'{results_path}/{experiment_name}_results_train.json'

  # set the seed for deterministic results...
  torch.manual_seed(_RUN_ID.value + _SEED.value)
  utils.set_random_seed(_RUN_ID.value + _SEED.value)

  # define an early stopper
  save_model_dir = os.path.join(run_directory, 'saved_models')
  save_model_id = (
      f'{MODEL_NAME}_{_SEED.value}_{_RUN_ID.value}_{experiment_name}'
  )
  early_stopper = early_stopping.EarlyStopMonitor(
      save_model_dir=save_model_dir,
      save_model_id=save_model_id,
      tolerance=_TOLERANCE.value,
      patience=_PATIENCE.value,
  )
  print('INFO: done setting up early stopping.')

  # ==================================================== Train & Validation
  # loading the validation negative samples
  val_dataset.load_val_ns()
  start_train_val = timeit.default_timer()
  metrics_logger = evaluate.MetricsLogger()
  for epoch in range(1, _NUM_EPOCH.value + 1):
    # training
    start_epoch_train = timeit.default_timer()
    print(f'INFO: Epoch {epoch}.')
    loss = train_test_helper.train(
        model,
        data,
        train_loader,
        device,
        min_dst_idx,
        max_dst_idx,
        last_neighbor_loader,
        metrics_logger,
        structural_feats_list,
        train_structural_features,
    )
    print(
        f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s):'
        f' {timeit.default_timer() - start_epoch_train: .4f}'
    )

    print('INFO: Starting validation.')
    # validation
    start_val = timeit.default_timer()

    if _RESET_MEMORY.value and model.has_memory:
      model.reset_memory()

    if _RESET_NBD_LOADER.value:
      last_neighbor_loader.reset_state()  # Start with an empty graph

    # Warmstart
    if _WARMSTART.value:
      warmstart_performance_lists = train_test_helper.warmstart(
          model,
          data,
          warmstart_val_loader,
          device,
          min_dst_idx_warmstart,
          max_dst_idx_warmstart,
          metric,
          last_neighbor_loader,
          evaluator,
          metrics_logger,
          update_model=_WARMSTART_UPDATE_MODEL.value,
          structural_feats_list=structural_feats_list,
          structural_features=val_structural_features,
      )
      print('INFO: Warmstart done.')
      warmstart_loss = pd.DataFrame()
      warmstart_loss['loss'] = warmstart_performance_lists.loss
      warmstart_loss['model_loss'] = warmstart_performance_lists.model_loss
      warmstart_loss['perf'] = warmstart_performance_lists.perf
      warmstart_loss['auc'] = warmstart_performance_lists.auc
      with tf.io.gfile.GFile(
          os.path.join(
              results_path, f'{experiment_name}_val_warmstart_loss.csv'
          ),
          'w',
      ) as f:
        warmstart_loss.to_csv(f, index=False)

    perf_metric_val, auc, test_performance_lists = train_test_helper.test(
        model,
        device,
        evaluator,
        last_neighbor_loader,
        data,
        metric,
        val_loader,
        neg_sampler,
        split_mode='val',
        update_memory=_UPDATE_MEM.value,
        warmstart_batch_id=batches_in_warmstart,
        structural_feats_list=structural_feats_list,
        structural_features=val_structural_features,
    )
    print(f'\tValidation {metric}: {perf_metric_val: .4f}')
    print(f'\tValidation AUC: {auc: .4f}')
    print(
        '\tValidation: Elapsed time (s):'
        f' {timeit.default_timer() - start_val: .4f}'
    )
    val_loss = pd.DataFrame()
    val_loss['loss'] = test_performance_lists.loss
    val_loss['model_loss'] = test_performance_lists.model_loss
    val_loss['perf'] = test_performance_lists.perf
    val_loss['auc'] = test_performance_lists.auc
    with tf.io.gfile.GFile(
        os.path.join(results_path, f'{experiment_name}_val_loss.csv'), 'w'
    ) as f:
      val_loss.to_csv(f, index=False)

    # check for early stopping
    if early_stopper.step_check(perf_metric_val, model):
      break

  train_val_time = timeit.default_timer() - start_train_val
  print(f'Train & Validation: Elapsed Time (s): {train_val_time: .4f}')
  # ==================================================== Test
  # first, load the best model
  early_stopper.load_checkpoint(model)

  utils.save_results(
      {
          'model': MODEL_NAME,
          'data': _DATA.value,
          'run': _RUN_ID.value,
          'seed': _SEED.value,
          f'val {metric}': perf_metric_val,
          'auc': auc,
          'tot_train_val_time': train_val_time,
      },
      results_filename,
  )

  print(
      f'INFO: >>>>> Run: {_RUN_ID.value}, elapsed time:'
      f' {timeit.default_timer() - start_run: .4f} <<<<<'
  )
  print(
      '-------------------------------------------------------------------------------'
  )

  print(
      f'Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}'
  )
  print('==============================================================')


if __name__ == '__main__':
  app.run(main)

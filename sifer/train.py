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

"""Trainer File."""

import collections
import json
import os
import pickle
import sys
import time

from absl import app
from absl import flags
from ml_collections import config_flags
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from sifer import dataset
from sifer import learning
from sifer import params
from sifer.utils import eval_helper
from sifer.utils import misc


FLAGS = flags.FLAGS

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

_work_unit = None

config_flags.DEFINE_config_file(
    'config',
    'sifer/params.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False,
)




def main(_):

  misc.prepare_environ_for_pytorch()
  args = params.update_config(FLAGS.config)

  start_step = 0
  args.output_folder_name = f'{args.dataset}_{args.algorithm}_hparams{args.hparams_seed}_seed{args.seed}'
  if _work_unit is not None:
    args.output_folder_name += str(_work_unit.id)
  misc.prepare_folders(args)
  args.output_dir = os.path.join(args.output_dir, args.output_folder_name)
  sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
  sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

  # download data and weights
  print('downloading files ...')
  misc.download_files()
  print('downloading done')

  print('Args:')
  for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))

  misc.set_seed(args.seed)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  if args.dataset in dataset.DATASETS.keys():
    train_dataset = dataset.DATASETS[args.dataset](
        args.data_dir, 'tr', train_attr='no'
    )
  else:
    raise NotImplementedError

  num_workers = train_dataset.N_WORKERS
  input_shape = train_dataset.INPUT_SHAPE
  num_labels = train_dataset.num_labels
  num_attributes = train_dataset.num_attributes
  data_type = train_dataset.data_type
  n_steps = args.steps or train_dataset.N_STEPS
  checkpoint_freq = args.checkpoint_freq or train_dataset.CHECKPOINT_FREQ
  args.n_steps = n_steps

  if args.debug:
    n_steps = 100

  train_loader = dataset.fast_dataloader.InfiniteDataLoader(
      dataset=train_dataset,
      batch_size=args.batch_size,
      num_workers=num_workers,
  )

  split_names = ['va'] + dataset.DATASETS[args.dataset].EVAL_SPLITS
  dsets = [
      dataset.DATASETS[args.dataset](args.data_dir, split)
      for split in split_names
  ]
  eval_loaders = []
  for dset in dsets:
    eval_loaders.append(
        dataset.fast_dataloader.FastDataLoader(
            dataset=dset,
            batch_size=max(128, args.batch_size * 2),
            num_workers=num_workers,
        )
    )

  algorithm_class = learning.get_algorithm_class(args.algorithm)
  algorithm = algorithm_class(
      data_type,
      input_shape,
      num_labels,
      num_attributes,
      len(train_dataset),
      args,
      grp_sizes=train_dataset.group_sizes,
  )

  best_model_path = os.path.join(args.output_dir, 'model.best.pkl')

  algorithm.to(device)

  train_minibatches_iterator = iter(train_loader)
  checkpoint_vals = collections.defaultdict(lambda: [])
  steps_per_epoch = len(train_dataset) / args.batch_size

  def save_checkpoint(save_dict, filepath=best_model_path):
    torch.save(save_dict, filepath)

  # last_results_keys = None
  best_val_metric = 0
  is_best = False
  es_group, es_metric = args.es_metric.split(':')

  for step in range(start_step, n_steps):
    step_start_time = time.time()
    i, x, y, a = next(train_minibatches_iterator)
    minibatch_device = (i, x.to(device), y.to(device), a.to(device))
    algorithm.train()
    step_vals = algorithm.update(minibatch_device, step)
    checkpoint_vals['step_time'].append(time.time() - step_start_time)

    for key, val in step_vals.items():
      checkpoint_vals[key].append(val)

    if (step % checkpoint_freq == 0) or (step == n_steps - 1):
      results = {
          'step': step,
          'epoch': step / steps_per_epoch,
      }
      for key, val in checkpoint_vals.items():
        results[key] = np.mean(val)

      curr_metrics = {
          split: eval_helper.eval_metrics(algorithm, loader, device)
          for split, loader in zip(split_names, eval_loaders)
      }
      if curr_metrics['va'][es_group][es_metric] > best_val_metric:
        best_val_metric = curr_metrics['va'][es_group][es_metric]
        is_best = True

      for split in sorted(split_names):
        for key1 in curr_metrics[split]:
          for key2 in curr_metrics[split][key1]:
            if isinstance(curr_metrics[split][key1][key2], dict):
              for key3 in curr_metrics[split][key1][key2]:
                results[f'{split}_{key1}_{key2}_{key3}'] = curr_metrics[split][
                    key1
                ][key2][key3]
                if is_best:
                  results[f'best_{split}_{key1}_{key2}_{key3}'] = curr_metrics[
                      split
                  ][key1][key2][key3]
            else:
              results[f'{split}_{key1}_{key2}'] = curr_metrics[split][key1][
                  key2
              ]
              if is_best:
                results[f'best_{split}_{key1}_{key2}'] = curr_metrics[split][
                    key1
                ][key2]

      if is_best:
        save_dict = {
            'args': vars(args),
            'best_es_metric': best_val_metric,
            'start_step': step + 1,
            'model_dict': algorithm.state_dict(),
        }
        save_checkpoint(save_dict)
        is_best = False

      results['mem_gb'] = torch.cuda.max_memory_allocated() / (
          1024.0 * 1024.0 * 1024.0
      )
      results.update({
          'args': vars(args),
      })

      epochs_path = os.path.join(args.output_dir, 'results.json')
      with open(epochs_path, 'a') as f:
        f.write(json.dumps(results, sort_keys=True) + '\n')

      checkpoint_vals = collections.defaultdict(lambda: [])

  # load best model and get metrics on eval sets
  algorithm.load_state_dict(
      torch.load(os.path.join(args.output_dir, 'model.best.pkl'))['model_dict']
  )

  algorithm.eval()

  split_names = ['va'] + dataset.DATASETS[args.dataset].EVAL_SPLITS
  final_eval_loaders = []
  dsets = [
      dataset.DATASETS[args.dataset](args.data_dir, split)
      for split in split_names
  ]
  for dset in dsets:
    final_eval_loaders.append(
        dataset.fast_dataloader.FastDataLoader(
            dataset=dset,
            batch_size=max(128, args.batch_size * 2),
            num_workers=num_workers,
        )
    )

  final_results = {
      split: eval_helper.eval_metrics(algorithm, loader, device)
      for split, loader in zip(split_names, final_eval_loaders)
  }
  pickle.dump(
      final_results,
      open(os.path.join(args.output_dir, 'final_results.pkl'), 'wb'),
  )
  # xm_logger_fn(final_results, n_steps)
  print('\nTest accuracy (best validation checkpoint):')
  print(
      f"\tmean:\t[{final_results['te']['overall']['accuracy']:.3f}]\n"
      f"\tworst:\t[{final_results['te']['min_group']['accuracy']:.3f}]"
  )
  print('Group-wise accuracy:')
  for split in final_results.keys():
    print(
        '\t[{}] group-wise {}'.format(
            split,
            (
                np.array2string(
                    pd.DataFrame(final_results[split]['per_group'])
                    .T['accuracy']
                    .values,
                    separator=', ',
                    formatter={'float_kind': lambda x: '%.3f' % x},
                )
            ),
        )
    )

  with open(os.path.join(args.output_dir, 'done'), 'w') as f:
    f.write('done')


if __name__ == '__main__':
  flags.mark_flags_as_required(['config'])
  # g3_multiprocessing.handle_main(main)
  app.run(main)

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

"""Evaluates intervention performance of interactive bottleneck models."""

import json
import os
import pickle
from typing import Any, Dict, List, Sequence, Tuple, Union

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from interactive_cbms import enum_utils
from interactive_cbms import intervened_dataset
from interactive_cbms import network
from interactive_cbms.datasets import chexpert_dataset
from interactive_cbms.datasets import cub_dataset
from interactive_cbms.datasets import oai_dataset

# pylint: disable=g-bad-import-order

tfk = tf.keras

_BTYPE = flags.DEFINE_enum_class(
    'bottleneck_type',
    default=enum_utils.BottleneckType.INDEPENDENT,
    enum_class=enum_utils.BottleneckType,
    help='Type of bottleneck to use for intervention.')
_NON_LINEAR_CTOY = flags.DEFINE_bool(
    'non_linear_ctoy',
    default=False,
    help='Whether to use a non-linear CtoY model.')
_POLICY = flags.DEFINE_enum_class(
    'intervention_policy',
    default=enum_utils.InterventionPolicy.GLOBAL_RANDOM,
    enum_class=enum_utils.InterventionPolicy,
    help='Intervention policy to use.')
_POLICY_KWARGS = flags.DEFINE_string(
    'policy_kwargs',
    default=None,
    help=('Arguments for the chosen intervention_policy in the form of a JSON '
          'string.'))
_LOAD_POLICY_FROM_SPLIT = flags.DEFINE_enum(
    'load_policy_from_split',
    default=None,
    enum_values=['val', 'test'],
    help=('Whether to load a pickled intervention policy and, (if not None) '
          'which corresponding data split to load from. Arguments from the '
          'loaded policy overrides policy_kwargs.'))
_FORMAT = flags.DEFINE_enum_class(
    'intervention_format',
    default=enum_utils.InterventionFormat.PROBS,
    enum_class=enum_utils.InterventionFormat,
    help='The format of intervention to use.')
_INCLUDE_UNCERTAIN = flags.DEFINE_bool(
    'include_uncertain',
    default=False,
    help=('Whether to intervene on concepts that are labelled as '
          'uncertain/not visible in the image. If include_uncertain=False, the '
          'true values of uncertain concepts are not revealed in the intervened'
          ' batch/dataset even when the intervention policy requests them.'))
_DATASET = flags.DEFINE_enum_class(
    'dataset',
    default=enum_utils.Dataset.CUB,
    enum_class=enum_utils.Dataset,
    help='Dataset to use for intervention.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', default=32, help='Batch Size')
_SPLIT = flags.DEFINE_enum(
    'split',
    default='val',
    enum_values=['val', 'test'],
    help='Dataset split to test intervention on.')
_N_STEPS = flags.DEFINE_integer(
    'n_steps', default=30, help='No. of steps of intervention to run')
_LOAD_POLICY_DIR = flags.DEFINE_string(
    'load_policy_dir',
    default=None,
    help=('directory to load the intervention policy from'))
_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir',
    default='ICBM_results/',
    help='Experiment directory to save results and models')
_CHECKPOINT = flags.DEFINE_enum_class(
    'checkpoint',
    default=enum_utils.Checkpoint.TRAINLOSS,
    enum_class=enum_utils.Checkpoint,
    help='Checkpoint to load from')


def load_dataset(
    ds_name, batch_size, split,
    random_cost_seed):
  """Utility function to load the dataset.

  Args:
    ds_name: Name of the dataset to load.
    batch_size: Batch size.
    split: Data split to load. Allowed values are "val" and "test".
    random_cost_seed: Random seed for cost assignment (only used when
      ds_name="cub")

  Returns:
    ds_test: The loaded dataset split as a tf.data.Dataset object.
    ds_info: A dictionary containing dataset metadata.
  """
  if ds_name is enum_utils.Dataset.CUB:
    dataset = cub_dataset
  elif ds_name is enum_utils.Dataset.CHEXPERT:
    dataset = chexpert_dataset
  elif ds_name is enum_utils.Dataset.OAI:
    dataset = oai_dataset
  logging.info('Loading dataset %s', ds_name)
  _, ds_test, _ = dataset.load_dataset(
      batch_size=batch_size, merge_train_and_val=split == 'test')
  concept_groups = dataset.load_concept_groups()
  concept_costs = dataset.load_concept_costs(concept_groups,
                                             seed=random_cost_seed)
  ds_info = {
      'image_size': dataset.Config.image_size,
      'n_concepts': dataset.Config.n_concepts,
      'n_classes': dataset.Config.n_classes,
      'batch_size': batch_size,
      'concept_groups': concept_groups,
      'concept_costs': concept_costs,
  }
  return ds_test, ds_info


def load_compile_models(xtoc_path, ctoy_path, non_linear_ctoy,
                        ds_info):
  """Utility function for loading X->C and C->Y models.

  Args:
    xtoc_path: Path to X->C model weights.
    ctoy_path: Path to C->Y model weights.
    non_linear_ctoy: Whether to use a non-linear CtoY model.
    ds_info: A dictionary containing dataset metadata.

  Returns:
    xtoc_model: The loaded and compiled X->C tfk.Model object.
    ctoy_model: The loaded and compiled C->Y tfk.Model object.
  """
  logging.info('Loading XtoC and CtoY models from %s and %s respectively',
               xtoc_path, ctoy_path)
  xtoc_model = network.InteractiveBottleneckModel(
      arch=enum_utils.Arch.X_TO_C,
      n_concepts=ds_info['n_concepts'],
      n_classes=ds_info['n_classes'],
      non_linear_ctoy=non_linear_ctoy)
  ctoy_model = network.InteractiveBottleneckModel(
      arch=enum_utils.Arch.C_TO_Y,
      n_concepts=ds_info['n_concepts'],
      n_classes=ds_info['n_classes'],
      non_linear_ctoy=non_linear_ctoy)
  xtoc_model.build([None, *ds_info['image_size']])
  ctoy_model.build([None, ds_info['n_concepts']])

  xtoc_model.load_weights(xtoc_path)
  ctoy_model.load_weights(ctoy_path)

  xtoc_model.compile(optimizer='sgd')
  ctoy_model.compile(optimizer='sgd')
  return xtoc_model, ctoy_model


def load_intervention_policy(policy_load_path,
                             policy_kwargs):
  """Utility function for loading existing intervention policy.

  Args:
    policy_load_path: Path to existing policy to load.
    policy_kwargs: Dictionary containing policy arguments to update using the
      loaded policy.

  Returns:
    Dictionary with updated keyword arguments.
  """
  with open(policy_load_path, 'rb') as f:
    load_policy_kwargs = pickle.load(f)
  policy_kwargs.update(load_policy_kwargs)
  return policy_kwargs


def evaluate_intervention(
    policy_name,
    intervention_format,
    include_uncertain, steps, xtoc_model,
    ctoy_model, policy_ds,
    policy_ds_info, policy_kwargs
):
  """Utility function to evaluate intervention performance.

  Args:
    policy_name: Policy to use for intervention.
    intervention_format: Concept representation format to use for intervention.
    include_uncertain: Whether to intervene on concepts that are labelled as
      uncertain/not visible in the image. If include_uncertain=False, the true
      values of uncertain concepts are not revealed in the intervened
      batch/dataset even when the intervention policy requests them.
    steps: Nunber of intervention steps.
    xtoc_model: X->C tfk.Model instance.
    ctoy_model: C->Y tfk.Model instance.
    policy_ds: A validation or test dataset that an intervention policy can use
      to determine the next best concept to request true labels for.
    policy_ds_info: Dictionary containing dataset metadata.
    policy_kwargs: Keyword arguments to pass on to the intervention policy.

  Returns:
    intervention_metrics: A dictionary containing the intervention metrics.
    concepts_revealed: A list of concepts revealed at each intervention step.
  """

  logging.info('Initializing %s intervention policy', policy_name)
  intervened_ds = intervened_dataset.IntervenedDataset(
      policy=policy_name,
      intervention_format=intervention_format,
      include_uncertain=include_uncertain,
      steps=steps,
      xtoc_model=xtoc_model,
      ctoy_model=ctoy_model,
      policy_dataset=policy_ds,
      concept_groups=policy_ds_info['concept_groups'],
      **policy_kwargs)
  intervened_ds_out_signature = (
      tf.TensorSpec(
          shape=(None, *policy_ds_info['image_size']), dtype=tf.float32),
      tf.TensorSpec(
          shape=(None, policy_ds_info['n_concepts']), dtype=tf.float32),
      tf.TensorSpec(
          shape=((None, 1) if policy_ds_info['n_classes'] == 1 else (None,)),
          dtype=tf.int64))
  intervention_metrics = {metric.name: [] for metric in ctoy_model.metrics}
  concepts_revealed = []
  for step, (intervention_mask,
             next_best_concept) in enumerate(intervened_ds.policy):
    logging.info('Evaluating intervention step %d', step)
    intervened_ds_test = intervened_ds.load_dataset(
        intervention_mask, policy_ds_info['batch_size'],
        intervened_ds_out_signature)
    metric_values = ctoy_model.evaluate(intervened_ds_test, verbose=0)
    for metric, metric_value in zip(ctoy_model.metrics, metric_values):
      intervention_metrics[metric.name].append(metric_value)
    concepts_revealed.append(next_best_concept)
  return intervention_metrics, concepts_revealed


def save_policy(policy_save_path, policy_kwargs,
                intervention_metrics,
                concepts_revealed):
  """Utility function for saving intervenion policy.

  Args:
    policy_save_path: Path to save the policy at.
    policy_kwargs: A dictionary containing the policy arguments.
    intervention_metrics: A dictionary containing the intervention metrics.
    concepts_revealed: A list of concepts revealed at each intervention step.
  """
  os.makedirs(os.path.dirname(policy_save_path))
  policy_kwargs.update({
      'metrics': intervention_metrics,
      'concepts_revealed': concepts_revealed
  })
  with open(policy_save_path, 'wb') as f:
    pickle.dump(policy_kwargs, f)



def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  policy_kwargs = json.loads(_POLICY_KWARGS.value)
  policy_dir = os.path.join('{}', _DATASET.value, 'intervention_{}')
  if _POLICY.value in [enum_utils.InterventionPolicy.GLOBAL_RANDOM,
                       enum_utils.InterventionPolicy.INSTANCE_RANDOM]:
    policy_path = os.path.join(
        policy_dir,
        (f'{_BTYPE.value}-{_POLICY.value}-{policy_kwargs["seed"]}-'
         f'{_FORMAT.value}-uncertain_{_INCLUDE_UNCERTAIN.value}.pkl'))
  elif _POLICY.value is enum_utils.InterventionPolicy.COOP:
    policy_path = os.path.join(
        policy_dir,
        (f'{_BTYPE.value}-{_POLICY.value}-'
         f'{policy_kwargs["concept_metric"]}-'
         f'{policy_kwargs["label_metric_weight"]}-'
         f'{policy_kwargs["label_metric"]}-{policy_kwargs["cost_weight"]}-'
         f'{policy_kwargs["random_cost_seed"]}-{_FORMAT.value}-'
         f'uncertain_{_INCLUDE_UNCERTAIN.value}.pkl'))
  else:
    policy_path = os.path.join(
        policy_dir, (f'{_BTYPE.value}-{_POLICY.value}-{_FORMAT.value}-'
                     f'uncertain_{_INCLUDE_UNCERTAIN.value}.pkl'))

  ds_test, ds_info = load_dataset(
      _DATASET.value, _BATCH_SIZE.value, _SPLIT.value,
      policy_kwargs['random_cost_seed'])
  policy_kwargs['concept_costs'] = ds_info['concept_costs']

  if _LOAD_POLICY_FROM_SPLIT.value is not None:
    policy_load_dir = _LOAD_POLICY_DIR.value
    if policy_load_dir is None:
      policy_load_dir = _EXPERIMENT_DIR.value

    policy_load_path = policy_path.format(policy_load_dir,
                                          _LOAD_POLICY_FROM_SPLIT.value)
    policy_kwargs = load_intervention_policy(policy_load_path, policy_kwargs)

  if _BTYPE.value is enum_utils.BottleneckType.INDEPENDENT:
    xtoc_path = os.path.join(_EXPERIMENT_DIR.value, _DATASET.value,
                             'XtoC', 'sgd_lr-0.01_wd-4e-05',
                             f'checkpoint_{_CHECKPOINT.value}')
    ctoy_path = os.path.join(_EXPERIMENT_DIR.value, _DATASET.value,
                             'CtoY', 'sgd_lr-0.001_wd-5e-05',
                             f'checkpoint_{_CHECKPOINT.value}')
  elif _BTYPE.value is enum_utils.BottleneckType.JOINT_SIGMOID:
    xtoc_path = ctoy_path = os.path.join(
        _EXPERIMENT_DIR.value, _DATASET.value, 'XtoCtoY_sigmoid',
        'sgd_lr-0.01_wd-4e-05', f'checkpoint_{_CHECKPOINT.value}')
  elif _BTYPE.value is enum_utils.BottleneckType.JOINT:
    xtoc_path = ctoy_path = os.path.join(
        _EXPERIMENT_DIR.value, _DATASET.value, 'XtoCtoY',
        'sgd_lr-0.01_wd-4e-05', f'checkpoint_{_CHECKPOINT.value}')

  xtoc_model, ctoy_model = load_compile_models(
      xtoc_path, ctoy_path, _NON_LINEAR_CTOY.value, ds_info)
  logging.info('Evaluating XtoC')
  xtoc_model.evaluate(ds_test, verbose=2)

  intervention_metrics, concepts_revealed = evaluate_intervention(
      _POLICY.value, _FORMAT.value, _INCLUDE_UNCERTAIN.value, _N_STEPS.value,
      xtoc_model, ctoy_model, ds_test, ds_info, policy_kwargs)

  policy_save_path = policy_path.format(_EXPERIMENT_DIR.value, _SPLIT.value)
  save_policy(policy_save_path, policy_kwargs, intervention_metrics,
              concepts_revealed)


if __name__ == '__main__':
  app.run(main)

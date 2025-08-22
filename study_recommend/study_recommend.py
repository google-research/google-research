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

"""Launch script to launch train and/or evaluation jobs for models in STUDY."""


import functools
import json
import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import pandas as pd
from pandas.core import groupby as pd_groupby
from study_recommend import config
from study_recommend import datasource as datasource_lib
from study_recommend import inference
from study_recommend import training_loop
from study_recommend import types
from study_recommend.utils import evaluation
from study_recommend.utils import input_pipeline_utils
from study_recommend.utils import load_data

jax.config.parse_flags_with_absl()

_TRAIN_DATA_PATH = flags.DEFINE_string(
    'train_data_path',
    None,
    'Path to the file with the student-title actvity to use for training.',
)
_VALID_DATA_PATH = flags.DEFINE_string(
    'valid_data_path',
    None,
    'Path to the file with the student-title actvity to use for validation.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'Path to folder to write output files to.'
)
_LOAD_PATH = flags.DEFINE_string(
    'load_path',
    None,
    'Path for folder to load pre-trained mode from. If not supplied then it'
    ' will default to output_path.',
)
_TENSORBOARD_PATH = flags.DEFINE_string(
    'tensorboard_path',
    None,
    'Path to write tensorboard logs to. If not supplied it will default to a'
    ' subdirectory in output_path.',
)
_N_STEPS = flags.DEFINE_integer(
    'n_steps', 20_000, 'Number of steps to train for.'
)

_STUDENT_INFO_PATH = flags.DEFINE_string(
    'student_info_path',
    None,
    'Path to file with student information (grade level and school).',
)
_SCHOOL_INFO_PATH = flags.DEFINE_string(
    'school_info_path',
    None,
    'Path to file with school information (maps schools to district). '
    'Only required if using district_year grouping.',
)

VALID_CLASSROOM_GROUPINGS = ['school_year', 'district_year', 'none']
_CLASSROOM_GROUPING = flags.DEFINE_string(
    'classroom_grouping',
    'school_year',
    'How to group students into groups for inference. Valid values are'
    f' {", ".join(VALID_CLASSROOM_GROUPINGS)}.',
)

_VOCAB_SIZE = flags.DEFINE_integer(
    'vocab_size',
    2000,
    'This specifies the number of books to assign a unique token in the'
    ' modeling. All other books are lumped together as an out-of-vocabulary'
    ' token. We keep the most frequent books in the training dataset.',
)
_SEQ_LEN = flags.DEFINE_integer(
    'seq_len',
    33,
    'Maximum sequence length to process. Longer sequences will be split into'
    ' chunks.',
)
_STUDENT_CHUNK_SIZE = flags.DEFINE_integer(
    'student_chunk_size',
    None,
    'Determines the maximum number of interactions to consider per student for'
    ' STUDY models. If not supplied then it will be taken as equal to seq_len.',
)
INDIVIDUAL = 'individual'
STUDY = 'study'
AVAILABLE_MODELS = [INDIVIDUAL, STUDY]
_MODEL = flags.DEFINE_string(
    'model',
    'individual',
    'The kind of model to train. Valid values are'
    f' {", ".join(AVAILABLE_MODELS)}.',
)


_EVAL_EVERY = flags.DEFINE_integer(
    'eval_every',
    500,
    'How many timesteps between two consecutive logs of heavier evaluations of'
    ' model performance during training.',
)

_SAVE_EVERY = flags.DEFINE_integer(
    'save_every',
    1000,
    'How many timesteps between two consecute saves of progress to a checkpoint'
    ' file.',
)

_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 0.1024, 'Peak learning rate.'
)
_PER_DEVICE_BATCH_SIZE = flags.DEFINE_integer(
    'per_device_batch_size',
    2048,
    'The maximum number of samples that can fit on a single accelerator'
    ' (GPU/TPU) during training. The total batch size is scaled up by the'
    ' number of available accelerators.',
)
_N_RECOMMENDATIONS = flags.DEFINE_list(
    'n_recommendations',
    ['1', '3', '5', '10', '20'],
    'A list of values of target number of recommendations to evalute.'
    ' performance at.',
)
# Flags for multiprocessing.
_IN_MANAGED_PARALLEL_ENV = flags.DEFINE_bool(
    'in_managed_parallel_env',
    False,
    'Supplying this flag indicates that the script is launched via SLURM,'
    ' Open MPI, a TPU environment. If this flag is passed then '
    'coordinator_address and num_processed and process_id do not need to be '
    'supplied.',
)
_NUM_PROCESSES = flags.DEFINE_integer(
    'num_processes',
    1,
    'The number of training processes to be launched. We recommend one per GPU'
    ' for training. This must be set to 1 for running evaluations. Not required'
    ' if --in_managed_paralled_env is passed',
    required=False,
)
_PROCESS_ID = flags.DEFINE_integer(
    'process_id',
    None,
    'The index of this process. Not required if num_processes=1 or'
    ' --in_managed_paralled_env is passed.',
    required=False,
)
_COORDINATOR_ADDRESS = flags.DEFINE_string(
    'coordinator_address',
    None,
    'The address of the coordinator process ip:port. Not required if'
    ' num_processes=1 or --in_managed_paralled_env is passed',
    required=False,
)


_SEED = flags.DEFINE_integer('seed', 0, '')

_EXPERIMENTAL_NUM_GRADE_LEVELS = flags.DEFINE_integer(
    'experimental_num_grade_levels',
    None,
    'The value for the maximum grade level. When this flag is passed an'
    ' experimental feature that allows the model to customise recommendations'
    ' based on grade level from a complete cold start is enabled. Any students'
    ' with grade level larger than this value as well as students with grade'
    ' level unknown will be bucketed in the additional default grade level.',
)
VOCAB_FILE_NAME = 'vocab.json'

flags.register_validator(
    'model',
    lambda model: model in AVAILABLE_MODELS,
    message='Model must be one of ' + ', '.join(AVAILABLE_MODELS),
)

flags.register_validator(
    'classroom_grouping',
    lambda x: x in VALID_CLASSROOM_GROUPINGS,
    message='Must be one of ' + ', '.join(VALID_CLASSROOM_GROUPINGS),
)
flags.register_validator(
    'n_recommendations',
    lambda list_: [entry.isdigit() for entry in list_],
    message='Must be a list of comma separated integers.',
)

open_file = open
make_dirs = functools.partial(os.makedirs, exist_ok=True)
file_exists = os.path.exists


def sync_all_hosts_before_close():
  """Make Jax processes that have finished wait for the host before terminating.

  If the jax processes that are not doing evaluations terminate earlier than the
  host the host will crash upon loss of connection to the processes that
  terminate.
  """
  if jax.process_count() > 1:
    # Make sure all hosts stay up until the end of main.
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    assert x[0] == jax.device_count()


def score_and_save_recommendations(
    recommendations,
    eval_data,
    n_recommendations,
):
  """Compute hits@n scores for recommendations and write results to disc.

  Args:
    recommendations: The recommendations predicted for evaluation data
    eval_data: Grouped Dataframe with the actual validation interaction history.
    n_recommendations: A list of values of n to compute hits@n for.
  """
  logging.info('Preprocessing titles for evaluation')
  reference_titles = evaluation.interaction_df_to_title_array(eval_data)

  logging.info('Computing aggregate metrics on recommendations.')
  make_dirs(_OUTPUT_PATH.value)

  results = {}
  for n in n_recommendations:
    results[n] = evaluation.hits_at_n(reference_titles, recommendations, n)

  logging.info('Computing per student hits@n  from the recommedations ....')
  per_student_results = evaluation.compute_per_student_hits_at_n_dataframe(
      reference_titles,
      recommendations,
      n_recommendations,
      compute_non_history=True,
      compute_non_continuation=True,
  )
  per_student_results_path = os.path.join(_OUTPUT_PATH.value, 'per_student.csv')

  logging.info('Writing per students results to %s', per_student_results_path)
  with open_file(per_student_results_path, 'w') as f:
    per_student_results.to_csv(f, index=False)

  aggregates = (
      per_student_results.drop(
          types.ResultsRecordFields.STUDENT_ID.value, axis=1
      )
      .groupby(
          [
              types.ResultsRecordFields.EVAL_TYPE.value,
              types.ResultsRecordFields.N_RECOMMENDATIONS.value,
          ],
          as_index=False,
      )
      .aggregate('mean')
  )
  aggregates_path = os.path.join(_OUTPUT_PATH.value, 'aggregate.csv')

  logging.info('Writing aggregate results to %s', aggregates_path)
  with open_file(aggregates_path, 'w') as f:
    aggregates.to_csv(f, index=False)

  recommendations_path = os.path.join(
      _OUTPUT_PATH.value, 'recommendations.json'
  )

  logging.info('Writing predictions to %s', recommendations_path)
  with open_file(recommendations_path, 'w') as f:
    f.write(json.dumps(recommendations, indent=2))


def main(_):
  logging.info('flags %s', flags.FLAGS.flag_values_dict())
  if _NUM_PROCESSES.value == 1:
    logging.info('In single process environment.')
  elif _IN_MANAGED_PARALLEL_ENV.value:
    logging.info('In MPI, SLURM or TPU environment.')
    jax.distributed.initialize()
    logging.info('Called jax.distributed.initialize() with no arguments.')
  else:
    logging.info('Not in MPI, SLURM, TPU or single process environment.')
    # These flags must be supplied
    assert _COORDINATOR_ADDRESS.value is not None
    assert _NUM_PROCESSES.value is not None
    assert _PROCESS_ID.value is not None

    jax.distributed.initialize(
        coordinator_address=_COORDINATOR_ADDRESS.value,
        num_processes=_NUM_PROCESSES.value,
        process_id=_PROCESS_ID.value,
        local_device_ids=_PROCESS_ID.value,
    )
    logging.info(
        'Called jax.distributed.initialize() with coordinator_address=%s,'
        ' num_processes=%d, process_id=%d, local_device_ids=%d.',
        _COORDINATOR_ADDRESS.value,
        _NUM_PROCESSES.value,
        _PROCESS_ID.value,
        _PROCESS_ID.value,
    )
  logging.info('jax.device_count() = %d', jax.device_count())

  logging.info('len(jax.local_devices()) = %d', len(jax.local_devices()))

  logging.info('Loading student info ...')
  with open_file(_STUDENT_INFO_PATH.value, 'r') as f:
    student_info = pd.read_csv(f)

  if _SCHOOL_INFO_PATH.value is not None:
    logging.info('Loading school info ...')
    with open_file(_SCHOOL_INFO_PATH.value, 'r') as f:
      school_info = pd.read_csv(f)
  else:
    school_info = None

  logging.info('Loading training data ...')
  train_data = load_data.load_student_activity_files(_TRAIN_DATA_PATH.value)

  logging.info('Loading validation data ...')
  valid_data = load_data.load_student_activity_files(_VALID_DATA_PATH.value)

  # Try to load vocab if has already been created. Otherwise it will be created
  # when preprocessing training data.
  load_path = _LOAD_PATH.value or _OUTPUT_PATH.value
  vocab_path = os.path.join(load_path, VOCAB_FILE_NAME)
  if file_exists(vocab_path):
    logging.info('Loading vocab from %s', vocab_path)
    with open_file(vocab_path, 'r') as f:
      vocab = input_pipeline_utils.Vocabulary().deserialize(f)
      vocab_size = None
      new_vocab_created = False
  else:
    vocab = None
    vocab_size = _VOCAB_SIZE.value
    new_vocab_created = True

  logging.info('Preprocessing training data ...')
  # These fields are required by all mdoels
  fields = (types.ModelInputFields.TITLES, types.ModelInputFields.STUDENT_IDS)
  # Now add fields required by specific models.
  if _MODEL.value == INDIVIDUAL:
    fields += (types.ModelInputFields.INPUT_POSITIONS,)
  elif _MODEL.value == STUDY:
    fields += (types.ModelInputFields.TIMESTAMPS,)
  if _EXPERIMENTAL_NUM_GRADE_LEVELS.value is not None:
    fields += (types.ModelInputFields.GRADE_LEVELS,)

  train_datasource, vocab = (
      datasource_lib.ClassroomGroupedDataSource.datasource_from_grouped_activity_dataframe(
          train_data,
          student_info,
          seq_len=_SEQ_LEN.value,
          fields=fields,
          student_chunk_len=_STUDENT_CHUNK_SIZE.value,
          vocab=vocab,
          vocab_size=vocab_size,
          classroom=_CLASSROOM_GROUPING.value,
          school_info=school_info,
          with_replacement=True,
          max_grade_level=_EXPERIMENTAL_NUM_GRADE_LEVELS.value,
      )
  )

  # Write vocab to disk if newly created.
  if new_vocab_created and jax.process_index() == 0:
    vocab_path = os.path.join(_OUTPUT_PATH.value, VOCAB_FILE_NAME)
    logging.info('Writing vocab to %s', vocab_path)
    make_dirs(os.path.dirname(vocab_path))
    with open_file(vocab_path, 'w') as f:
      vocab.serialize(f)

  logging.info('Preprocessing validation data ...')
  valid_datasource, _ = (
      datasource_lib.ClassroomGroupedDataSource.datasource_from_grouped_activity_dataframe(
          valid_data,
          student_info,
          seq_len=_SEQ_LEN.value,
          fields=fields,
          student_chunk_len=_STUDENT_CHUNK_SIZE.value,
          vocab=vocab,
          classroom=_CLASSROOM_GROUPING.value,
          school_info=school_info,
          with_replacement=True,
          max_grade_level=_EXPERIMENTAL_NUM_GRADE_LEVELS.value,
      )
  )

  cfg = config.get_config(
      vocab,
      _PER_DEVICE_BATCH_SIZE.value,
      seq_len=_SEQ_LEN.value,
      working_dir=_OUTPUT_PATH.value,
      tensorboard_dir=_TENSORBOARD_PATH.value,
  )

  cfg.num_train_steps = _N_STEPS.value

  def modify(attr_dict, **kwargs):
    assert all(key in attr_dict for key in kwargs)
    attr_dict.update(kwargs)
    return attr_dict

  cfg = modify(
      cfg,
      num_train_steps=_N_STEPS.value,
      model_class=_MODEL.value,
      chunk_length=_STUDENT_CHUNK_SIZE.value,
      seed=_SEED.value,
      learning_rate=_LEARNING_RATE.value,
      checkpoint_every_steps=_SAVE_EVERY.value,
      eval_every_steps=_EVAL_EVERY.value,
  )

  if _EXPERIMENTAL_NUM_GRADE_LEVELS.value is not None:
    # We have embeddings for grade levels 1 through n plus an additional
    # catch all embedding for students with missing or invalid grade levels.
    # This catch all grade level is assigned value 0.
    num_grade_level_embeddings = _EXPERIMENTAL_NUM_GRADE_LEVELS.value + 1
    cfg = modify(cfg, num_grade_levels=num_grade_level_embeddings)

  logging.info('Loading or initialising model ...')
  loaded_model = training_loop.init_or_load(
      cfg, train_datasource, _LOAD_PATH.value
  )

  state, _, eval_cfg, model_class = loaded_model
  if state.step < _N_STEPS.value:
    logging.info('Calling training loop ...')
    state, eval_cfg = training_loop.training_loop(
        cfg, train_datasource, valid_datasource, preloaded=loaded_model
    )
  else:
    logging.info(
        'No training steps required, skipping training. Restored from step %s',
        state.step,
    )

  logging.info('resetting validation datasource')
  valid_datasource.re_initialize_sampler(
      with_replacement=False,
      with_automatic_reset=False,
      ordered_within_student=True,
  )

  n_recommedations = [int(i) for i in _N_RECOMMENDATIONS.value]
  logging.info(
      'Running inference to produce recommendations of validation dataset ...'
  )
  recommendations = inference.recommend_from_datasource(
      eval_cfg,
      model_class,
      state.params,
      valid_datasource,
      n_recommendations=max(n_recommedations),
      vocab=vocab,
      per_device_batch_size=_PER_DEVICE_BATCH_SIZE.value,
  )

  is_host = jax.process_index() == 0
  if is_host:
    # Only the hosts will do evaluations and write results to disk
    # results to disk.
    score_and_save_recommendations(
        recommendations, valid_data, n_recommedations
    )

  logging.info('Waiting for other hosts to finish.')
  sync_all_hosts_before_close()
  logging.info('Done.')


if __name__ == '__main__':
  app.run(main)

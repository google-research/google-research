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

"""Various constants needed in the solution."""
import utils


class DistributeModeChoices(utils.FlagChoices):
  mirroredstrategy = "MirroredStrategy"
  none = "None"
  onedevicestrategy = "OneDeviceStrategy"
  split_vertically = "split_vertically"
  split_and_data_parallel = "split_vertically_data_parallel"
  tpustrategy = "TPUStrategy"


STRATEGIES = frozenset([
    DistributeModeChoices.onedevicestrategy,
    DistributeModeChoices.tpustrategy,
    DistributeModeChoices.mirroredstrategy
])


DATA_PARALLEL_DMC = frozenset([
    DistributeModeChoices.tpustrategy,
    DistributeModeChoices.mirroredstrategy,
    DistributeModeChoices.split_and_data_parallel
])


# Pure Data Parallel Strategies means that they don't do Model Parallelism.
# This is (was) for when we also used tf.distribute.TPUMPStrategy
PURE_DATA_PARALLEL_STRATEGIES = frozenset([
    DistributeModeChoices.tpustrategy,
    DistributeModeChoices.mirroredstrategy,
])


DATA_PARALLEL_STRATEGIES = frozenset([
    DistributeModeChoices.tpustrategy,
    DistributeModeChoices.mirroredstrategy,
])


# Taken from internal documentation
COMPUTATION_SHAPE_FROM_NUM_CORES = {
    1: [1, 1, 1, 1],
    2: [1, 1, 1, 2],
    4: [1, 2, 1, 2],
    8: [2, 2, 1, 2],
    16: [4, 2, 1, 2],
}


class ApproachTypeChoices(utils.FlagChoices):
  naked_lm = "naked_lm"
  lm_and_realm = "lm_and_realm"
  cached_realm = "cached_realm"
  cached_pretok = "cached_pretok"


class ModelTypeChoices(utils.FlagChoices):
  distilgpt2 = "distilgpt2"
  gpt2_xl = "gpt2-xl"
  gpt2_large = "gpt2-large"
  gpt2_medium = "gpt2-medium"
  gpt2 = "gpt2"


RETRIEVAL_APPROACH_TYPES = frozenset([
    ApproachTypeChoices.lm_and_realm,
    ApproachTypeChoices.cached_realm,
    ApproachTypeChoices.cached_pretok,
])


# There will eventually be more, like generate.
class TaskChoices(utils.FlagChoices):
  train = "train"


class DatasetNameChoices(utils.FlagChoices):
  kilt_eli5 = "kilt_eli5"


PPL_MASK_ID = -100
RAGGED_PADDING_ID = -1


class CTH5Fields(utils.FlagChoices):
  distances = "distances"
  gpt2_question_ids_inputs = "gpt2_question_ids"
  gpt2_answer_ids_inputs = "gpt2_answer_ids"
  gpt2_retrieved_ids = "gpt2_retrieved_ids"

  # Deprecated:
  gpt2_question_ids_labels = "gpt2_question_labels"
  gpt2_answer_ids_labels = "gpt2_answer_labels"


class SplitChoices(utils.FlagChoices):
  train = "train"
  eval = "eval"
  test = "test"


class DatasetTypeChoices(utils.FlagChoices):
  tfr = "tfr"
  hdf5 = "hdf5"

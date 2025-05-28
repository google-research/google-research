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

"""T5 modified tasks that accept different sentence pieces.

NOTE:
To create a new task, we assume that there is gin configuration that points to
a sentencepiece model: seqio.SentencePieceVocabulary.sentencepiece_model_file.
This is the sentencepiece that is used for the task.
"""
import functools
from absl import logging
import gin
import seqio
import t5.data
from t5.data import preprocessors
from t5.data.glue_utils import get_glue_metric
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
import t5.data.mixtures
import t5.data.tasks
import tensorflow_datasets as tfds

MixtureRegistry = seqio.MixtureRegistry
TaskRegistry = seqio.TaskRegistry
TfdsTask = t5.data.TfdsTask

DEFAULT_EXTRA_IDS = 100


def get_custom_output_features(add_eos=True, extra_ids=DEFAULT_EXTRA_IDS):
  """Construct output features with custom vocabs."""
  sentence_piece_model_path = gin.query_parameter(
      "seqio.SentencePieceVocabulary.sentencepiece_model_file")

  custom_vocab = seqio.SentencePieceVocabulary(sentence_piece_model_path,
                                               extra_ids)
  return {
      "inputs":
          seqio.Feature(
              vocabulary=custom_vocab, add_eos=add_eos, required=False),
      "targets":
          seqio.Feature(vocabulary=custom_vocab, add_eos=add_eos)
  }


# C4 c4_v220_span_corruption data with a custom sentencepiece.
TaskRegistry.add(
    "c4_v220_span_corruption_custom_sp",
    source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=get_custom_output_features(),
    metric_fns=[])

# =================================== GLUE =====================================
# The glue tasks with custom sentencepiece (for t5).

for b in tfds.text.glue.Glue.builder_configs.values():
  name = "glue_%s_v002_custom_sp" % b.name
  logging.info("Registering glue task %s", name)

  seqio.TaskRegistry.add(
      name,
      source=seqio.TfdsDataSource(
          tfds_name="glue/%s:1.0.0" % b.name,
          splits=["test"] if b.name == "ax" else None),
      preprocessors=[
          get_glue_text_preprocessor(b),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=get_glue_metric(b.name),
      output_features=get_custom_output_features(),
      postprocess_fn=get_glue_postprocess_fn(b))


def get_glue_weight_mapping():
  """Prepares mapping of glue mixtures for t5 with custom sp or byt5 models."""
  add_on = "_custom_sp"

  # The same weight distribution as in standard tasks.
  base_weighted_task = {
      "glue_cola_v002": 8_551.,
      "glue_sst2_v002": 67_349.,
      "glue_mrpc_v002": 3_668.,
      "glue_qqp_v002": 363_849.,
      "glue_stsb_v002": 5_749.,
      "glue_mnli_v002": 392_702.,
      "glue_qnli_v002": 104_743.,
      "glue_rte_v002": 2_490.,
      "glue_mnli_mismatched_v002": 0.,
      "glue_mnli_matched_v002": 0.,
      "glue_ax_v002": 0.,
  }

  res = {}
  for k, v in base_weighted_task.items():
    res[k + add_on] = v

  return res


# Glue mixtures for fine tuning that use the same weights as the standard
# glue_v002_proportional but have different vocabs.
seqio.MixtureRegistry.add("glue_v002_proportional_custom_sp",
                          list(get_glue_weight_mapping().items()))

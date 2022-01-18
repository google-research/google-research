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

"""AT5 tasks."""

import functools

import seqio
from t5.data import postprocessors as t5_postprocessors
from t5.data import preprocessors as t5_preprocessors

from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
from t5.data.glue_utils import get_super_glue_metric
from t5.evaluation import metrics as t5_metrics

import tensorflow_datasets as tfds
import t5_closed_book_qa.t5_cbqa.preprocessors as t5_cbqa_preprocessors

TaskRegistry = seqio.TaskRegistry

EN_VOCAB_SPM_PATH = "gs://t5-data/vocabs/cc_en.32000/sentencepiece.model"
WMT14_CUSTOM_SPM_PATH = "gs://t5-data/vocabs/wmt_ende.37000/spm.model"


WMT14_VOCAB_EXTRA_100 = seqio.SentencePieceVocabulary(
    WMT14_CUSTOM_SPM_PATH, extra_ids=100)
EN_VOCAB_EXTRA_100 = seqio.SentencePieceVocabulary(
    EN_VOCAB_SPM_PATH, extra_ids=100)

EN_VOCAB_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=EN_VOCAB_EXTRA_100, add_eos=True),
    "targets": seqio.Feature(vocabulary=EN_VOCAB_EXTRA_100, add_eos=True)
}

#================================ English only vocab ===========================
for version in ("2.2.0", "2.3.0", "2.3.1"):
  TaskRegistry.add(
      "c4_v{}_unsupervised_en32k".format(version.replace(".", "")),
      source=seqio.TfdsDataSource(tfds_name="c4/en:{}".format(version)),
      preprocessors=[
          functools.partial(
              t5_preprocessors.rekey,
              key_map={
                  "inputs": None,
                  "targets": "text"
              }),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          t5_preprocessors.unsupervised,
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=EN_VOCAB_OUTPUT_FEATURES,
      metric_fns=[])


#================================ XSUM =========================================
TaskRegistry.add(
    "xsum_v110",
    source=seqio.TfdsDataSource(tfds_name="xsum:1.1.0"),
    preprocessors=[
        functools.partial(
            t5_preprocessors.summarize,
            article_key="document",
            summary_key="summary"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[t5_metrics.rouge],
    output_features=EN_VOCAB_OUTPUT_FEATURES,
)

#============================ SuperGLUE English Vocab===========================
for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
  # We use a simplified version of WSC, defined below
  if "wsc" in b.name:
    continue
  if b.name == "axb":
    text_preprocessor = [
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                "premise": "sentence1",
                "hypothesis": "sentence2",
                "label": "label",
                "idx": "idx",
            }),
        get_glue_text_preprocessor(b)
    ]
  else:
    text_preprocessor = [get_glue_text_preprocessor(b)]
  TaskRegistry.add(
      "super_glue_%s_v102_envocab" % b.name,
      source=seqio.TfdsDataSource(
          tfds_name="super_glue/%s:1.0.2" % b.name,
          splits=["test"] if b.name in ["axb", "axg"] else None),
      preprocessors=text_preprocessor + [
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=get_super_glue_metric(b.name),
      output_features=EN_VOCAB_OUTPUT_FEATURES,
      postprocess_fn=get_glue_postprocess_fn(b))

# ======================== Definite Pronoun Resolution =========================
TaskRegistry.add(
    "dpr_v001_simple_envocab",
    source=seqio.TfdsDataSource(tfds_name="definite_pronoun_resolution:1.1.0"),
    preprocessors=[
        t5_preprocessors.definite_pronoun_resolution_simple,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[t5_metrics.accuracy],
    output_features=EN_VOCAB_OUTPUT_FEATURES)

# =================================== WSC ======================================
TaskRegistry.add(
    "super_glue_wsc_v102_simple_train_envocab",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/wsc.fixed:1.0.2", splits=["train"]),
    preprocessors=[
        functools.partial(
            t5_preprocessors.wsc_simple, correct_referent_only=True),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[],
    output_features=EN_VOCAB_OUTPUT_FEATURES)
TaskRegistry.add(
    "super_glue_wsc_v102_simple_eval_envocab",
    source=seqio.TfdsDataSource(
        tfds_name="super_glue/wsc.fixed:1.0.2", splits=["validation", "test"]),
    preprocessors=[
        functools.partial(
            t5_preprocessors.wsc_simple, correct_referent_only=False),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5_postprocessors.wsc_simple,
    metric_fns=[t5_metrics.accuracy],
    output_features=EN_VOCAB_OUTPUT_FEATURES)

# ============================ Web Questions ===================================

# This task uses 10% of the train split for validation.
TaskRegistry.add(
    "web_questions_open_envocab",
    source=seqio.TfdsDataSource(
        tfds_name="web_questions:1.0.0",
        splits={
            "train": "train[:3417]",  # ~90%, matches numbers used by ORQA
            "validation": "train[3417:]",  # ~10%, matches numbers used by ORQA
            "test": "test"
        }),
    preprocessors=[
        t5_cbqa_preprocessors.web_questions_open,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5_postprocessors.qa,
    metric_fns=[t5_metrics.squad],
    output_features=EN_VOCAB_OUTPUT_FEATURES)

# This tasks trains on the full train split.
TaskRegistry.add(
    "web_questions_open_test_envocab",
    source=seqio.TfdsDataSource(
        tfds_name="web_questions:1.0.0",
        splits={
            "train": "train",
            "validation": "test",
        }),
    preprocessors=[
        t5_cbqa_preprocessors.web_questions_open,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5_postprocessors.qa,
    metric_fns=[t5_metrics.squad],
    output_features=EN_VOCAB_OUTPUT_FEATURES)

# WMT14 en-de t2t task with the custom vocabulary for the original Transformer
# paper relication experiments.
# This is internal because the vocabulary only resides in CNS for now.
b = tfds.translate.wmt_t2t.WmtT2tTranslate.builder_configs["de-en"]

wmt14_t2t_output_features = {
    "inputs": seqio.Feature(vocabulary=WMT14_VOCAB_EXTRA_100, add_eos=True),
    "targets": seqio.Feature(vocabulary=WMT14_VOCAB_EXTRA_100, add_eos=True)
}
TaskRegistry.add(
    "wmt_t2t_ende_v003_vocab_37000",
    source=seqio.TfdsDataSource(tfds_name="wmt_t2t_translate/de-en:1.0.0"),
    preprocessors=[
        functools.partial(
            t5_preprocessors.translate,
            source_language=b.language_pair[1],
            target_language=b.language_pair[0],
        ),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[t5_metrics.bleu],
    output_features=wmt14_t2t_output_features)

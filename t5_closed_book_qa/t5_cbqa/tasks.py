# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""T5 CBQA tasks."""
import functools

from . import metrics
from . import postprocessors
from . import preprocessors

import seqio
from t5.data import postprocessors as t5_postprocessors
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics

MixtureRegistry = seqio.MixtureRegistry
TaskRegistry = seqio.TaskRegistry

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
NQ_TRAIN_SPLIT_START = 7830
NQ_TRAIN_SPLIT_END = 79168
NQO_TRAIN_SPLIT_END = 79168
WQ_TRAIN_SPLIT_END = 3417
TQA_TRAIN_SPLIT_END = 78785


DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH),
            add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH),
            add_eos=True)
}


# ========================== Natural Questions =================================

# Natural Questions open domain variant that most closely matches the official
# evaluation procedure.
# The model is trained to predict all ground-truth answers
# and is only considered correct if it predicts all answers for any one of the
# annotators. As in the official evaluation, we consider questions with fewer
# than two non-null annotations unanswerable (given the context) but because we
# cannot predict unanswerability without the context, we only compute the recall
# metric. Further, because our model does not have access to the oracle context,
# we also normalize predicted and ground-truth answers when comparing them.

# This task uses a portion of the train set for validation.
TaskRegistry.add(
    "natural_questions_nocontext",
    source=seqio.TfdsDataSource(
        tfds_name="natural_questions:0.0.2",
        splits={
            "train": f"train[{NQ_TRAIN_SPLIT_START}:{NQ_TRAIN_SPLIT_END}]",
            "validation": f"train[:{NQ_TRAIN_SPLIT_START}]",
            "test": "validation"
        }),
    preprocessors=[
        preprocessors.natural_questions_nocontext,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=postprocessors.natural_questions,
    metric_fns=[
        functools.partial(
            metrics.natural_questions,
            # Train set does not contain multiple annotations.
            non_null_threshold=1)
    ])
# This task uses full train split and reports metrics on the NQ validation split
# (which is the test set in the open domain setting).
TaskRegistry.add(
    "natural_questions_nocontext_test",
    source=seqio.TfdsDataSource(tfds_name="natural_questions:0.0.2"),
    preprocessors=[
        preprocessors.natural_questions_nocontext,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=postprocessors.natural_questions,
    metric_fns=[metrics.natural_questions])


# The standard open domain variant of Natural Questions, where:
# 1) the model is only ever trained to output a single answer;
# 2) if a question has multiple answers, it is trained to predict the first;
# 3) any questions with  answers longer than five tokens are ignored;
# 4) answers are normalized before being compared;

# This task uses a portion of the train split for validation.
TaskRegistry.add(
    "natural_questions_open",
    source=seqio.TfdsDataSource(
        tfds_name="natural_questions_open:1.0.0",
        splits={
            # ~90%, matches numbers used by ORQA
            "train": f"train[:{NQO_TRAIN_SPLIT_END}]",
            # ~10%, matches numbers used by ORQA
            "validation": f"train[{NQO_TRAIN_SPLIT_END}:]",
            "test": "validation"
        }),
    preprocessors=[
        preprocessors.natural_questions_open,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=t5_postprocessors.qa,
    metric_fns=[t5_metrics.squad])
# This is a slight variant of the previous task that selects a random answer
# when multiple are provided instead of using the first.
TaskRegistry.add(
    "natural_questions_open_randanswer",
    source=seqio.TfdsDataSource(
        tfds_name="natural_questions_open:1.0.0",
        splits={
            "train": f"train[:{NQO_TRAIN_SPLIT_END}]",
            "validation": f"train[{NQO_TRAIN_SPLIT_END}:]",
            "test": "validation"
        }),
    preprocessors=[
        preprocessors.natural_questions_open,
        preprocessors.sample_answer,
        seqio.preprocessors.tokenize,
        # Do not cache - ensures we are sampling different answers.
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=t5_postprocessors.qa,
    metric_fns=[t5_metrics.squad])
# This task uses full train split and reports metrics on the NQ validation split
# (which is the test set in the open domain setting).
TaskRegistry.add(
    "natural_questions_open_test",
    source=seqio.TfdsDataSource(tfds_name="natural_questions_open:1.0.0"),
    preprocessors=[
        preprocessors.natural_questions_open,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=t5_postprocessors.qa,
    metric_fns=[t5_metrics.squad])

# ============================ Web Questions ===================================

# This task uses 10% of the train split for validation.
TaskRegistry.add(
    "web_questions_open",
    source=seqio.TfdsDataSource(
        tfds_name="web_questions:1.0.0",
        splits={
            # ~90%, matches numbers used by ORQA
            "train": f"train[:{WQ_TRAIN_SPLIT_END}]",
            # ~10%, matches numbers used by ORQA
            "validation": f"train[{WQ_TRAIN_SPLIT_END}:]",
            "test": "test"
        }),
    preprocessors=[
        preprocessors.web_questions_open,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=t5_postprocessors.qa,
    metric_fns=[t5_metrics.squad],
)

# This tasks trains on the full train split.
TaskRegistry.add(
    "web_questions_open_test",
    source=seqio.TfdsDataSource(
        tfds_name="web_questions:1.0.0",
        splits={
            "train": "train",
            "validation": "test",
        }),
    preprocessors=[
        preprocessors.web_questions_open,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=t5_postprocessors.qa,
    metric_fns=[t5_metrics.squad],
)

# =============================== Trivia QA ====================================

TaskRegistry.add(
    "trivia_qa_open",
    source=seqio.TfdsDataSource(
        tfds_name="trivia_qa/unfiltered.nocontext:1.1.0",
        splits={
            # ~90%, matches numbers used by ORQA
            "train": f"train[:{TQA_TRAIN_SPLIT_END}]",
            # ~10%, matches numbers used by ORQA
            "validation": f"train[{TQA_TRAIN_SPLIT_END}:]",
            "test": "validation"
        }),
    preprocessors=[
        preprocessors.trivia_qa_open,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=t5_postprocessors.qa,
    metric_fns=[t5_metrics.trivia_qa])

# This tasks trains on combined train and validation splits.
TaskRegistry.add(
    "trivia_qa_open_test",
    source=seqio.TfdsDataSource(
        tfds_name="trivia_qa/unfiltered.nocontext:1.1.0",
        splits={
            "train": "train+validation",
            "test": "test"
        }),
    preprocessors=[
        preprocessors.trivia_qa_open,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=t5_postprocessors.qa,
    metric_fns=[t5_metrics.trivia_qa])


# ============================= CBQA Mixtures ==================================
# This mixture is to be used for hyperparameter tuning. Training happens on
# validation sets (if available) or subsplits of the train set. Evaluation
# happens on the validation (or heldout portion of the train) split.
MixtureRegistry.add(
    "closed_book_qa",
    [
        "trivia_qa_open",
        "natural_questions_open",
        "web_questions_open"
    ],
    default_rate=seqio.mixing_rate_num_examples
)

# This mixture is to be used at test time. Training happens on the combined
# train and validation splits and evaluation happens on the test split.
MixtureRegistry.add(
    "closed_book_qa_test",
    [
        "trivia_qa_open_test",
        "natural_questions_open_test",
        "web_questions_open_test"
    ],
    default_rate=seqio.mixing_rate_num_examples
)

# ========================= Salient Span Masking ===============================

TaskRegistry.add(
    "salient_span_masked_wikipedia",
    source=seqio.TfdsDataSource(
        tfds_name="salient_span_wikipedia/sentences:1.0.0"),
    preprocessors=[
        preprocessors.mask_salient_spans,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])

TaskRegistry.add(
    "span_corrupted_wikipedia",
    source=seqio.TfdsDataSource(
        tfds_name="salient_span_wikipedia/sentences:1.0.0"),
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5_preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])

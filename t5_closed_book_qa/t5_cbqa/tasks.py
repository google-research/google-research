# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

import t5.data
from t5.data import postprocessors as t5_postprocessors
from t5.evaluation import metrics as t5_metrics

MixtureRegistry = t5.data.MixtureRegistry
TaskRegistry = t5.data.TaskRegistry
TfdsTask = t5.data.TfdsTask

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS

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
    TfdsTask,
    tfds_name="natural_questions:0.0.2",
    splits={
        "train": "train[7830:]",
        "validation": "train[:7830]",
        "test": "validation"
    },
    text_preprocessor=preprocessors.natural_questions_nocontext,
    postprocess_fn=postprocessors.natural_questions,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
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
    TfdsTask,
    tfds_name="natural_questions:0.0.2",
    text_preprocessor=preprocessors.natural_questions_nocontext,
    postprocess_fn=postprocessors.natural_questions,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[metrics.natural_questions])


# The standard open domain variant of Natural Questions, where:
# 1) the model is only ever trained to output a single answer;
# 2) if a question has multiple answers, it is trained to predict the first;
# 3) any questions with  answers longer than five tokens are ignored;
# 4) answers are normalized before being compared;

# This task uses a portion of the train split for validation.
TaskRegistry.add(
    "natural_questions_open",
    TfdsTask,
    tfds_name="natural_questions:0.0.2",
    splits={
        "train": "train[7830:]",
        "validation": "train[:7830]",
        "test": "validation"
    },
    text_preprocessor=preprocessors.natural_questions_open,
    postprocess_fn=t5_postprocessors.qa,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[t5_metrics.squad])
# This is a slight variant of the previous task that selects a random answer
# when multiple are provided instead of using the first.
TaskRegistry.add(
    "natural_questions_open_randanswer",
    TfdsTask,
    tfds_name="natural_questions:0.0.2",
    splits={
        "train": "train[7830:]",
        "validation": "train[:7830]",
        "test": "validation"
    },
    text_preprocessor=functools.partial(
        preprocessors.natural_questions_open,
        sample_answer=True
    ),
    supports_caching=False,  # Ensures we are sampling different answers.
    postprocess_fn=t5_postprocessors.qa,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[t5_metrics.squad])
# This task uses full train split and reports metrics on the NQ validation split
# (which is the test set in the open domain setting).
TaskRegistry.add(
    "natural_questions_open_test",
    TfdsTask,
    tfds_name="natural_questions:0.0.2",
    text_preprocessor=preprocessors.natural_questions_open,
    postprocess_fn=t5_postprocessors.qa,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[t5_metrics.squad])

# ============================ Web Questions ===================================

# This task uses 10% of the train split for validation.
TaskRegistry.add(
    "web_questions_open",
    TfdsTask,
    tfds_name="web_questions:1.0.0",
    splits={
        "train": "train[10%:]",
        "validation": "train[:10%]",
        "test": "test"
    },
    text_preprocessor=[preprocessors.web_questions_open],
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    postprocess_fn=t5_postprocessors.qa,
    metric_fns=[t5_metrics.squad],
)

# This tasks trains on the full train split.
TaskRegistry.add(
    "web_questions_open_test",
    TfdsTask,
    tfds_name="web_questions:1.0.0",
    splits={
        "train": "train",
        "validation": "test",
    },
    text_preprocessor=[preprocessors.web_questions_open],
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    postprocess_fn=t5_postprocessors.qa,
    metric_fns=[t5_metrics.squad],
)

# =============================== Trivia QA ====================================

TaskRegistry.add(
    "trivia_qa_open",
    TfdsTask,
    tfds_name="trivia_qa/unfiltered.nocontext:1.1.0",
    text_preprocessor=preprocessors.trivia_qa_open,
    postprocess_fn=t5_postprocessors.qa,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[t5_metrics.trivia_qa])

# This tasks trains on combined train and validation splits.
TaskRegistry.add(
    "trivia_qa_open_test",
    TfdsTask,
    tfds_name="trivia_qa/unfiltered.nocontext:1.1.0",
    text_preprocessor=preprocessors.trivia_qa_open,
    splits={
        "train": "train+validation",
        "test": "test"
    },
    postprocess_fn=t5_postprocessors.qa,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[t5_metrics.trivia_qa])


# ============================= CBQA Mixtures ==================================
MixtureRegistry.add(
    "closed_book_qa",
    [
        ("trivia_qa_open", 87622),
        ("natural_questions_open", 85666),
        ("web_questions_open", 3400)
    ]
)

MixtureRegistry.add(
    "closed_book_qa_test",
    [
        ("trivia_qa_open_test", 98935),
        ("natural_questions_open_test", 87925),
        ("web_questions_open_test", 3778)
    ]
)

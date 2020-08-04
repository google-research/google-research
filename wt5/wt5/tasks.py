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

"""WT5 tasks."""
import functools

from . import metrics
from . import postprocessors
from . import preprocessors

import t5.data
from t5.data import postprocessors as t5_postprocessors
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics
import tensorflow_datasets as tfds


TaskRegistry = t5.data.TaskRegistry
TfdsTask = t5.data.TfdsTask

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS

# ======================== CoS-E Corpus Task ==================================
TaskRegistry.add(
    "cos_e_v001",
    TfdsTask,
    tfds_name="cos_e:0.0.1",
    text_preprocessor=preprocessors.cos_e,
    postprocess_fn=postprocessors.abstractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[metrics.esnli_metric])

# CoS-E with no explanations, and modified prefixes like e-SNLI.
TaskRegistry.add(
    "cos_e_v001_0_expln_like_esnli",
    TfdsTask,
    tfds_name="cos_e:0.0.1",
    text_preprocessor=functools.partial(
        preprocessors.cos_e, prefix="nli", question_prefix="premise:",
        drop_explanations=True),
    postprocess_fn=postprocessors.abstractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[metrics.esnli_metric])

n_cos_e_explanations = [5000, 2000, 1000, 500, 200, 100]
for n in n_cos_e_explanations:
  TaskRegistry.add(
      "cos_e_explanations_take{}_v001".format(n),
      t5.data.TfdsTask,
      tfds_name="cos_e:0.0.1",
      splits={"train": "train[0:{}]".format(n)},
      text_preprocessor=[preprocessors.cos_e],
      postprocess_fn=postprocessors.abstractive_explanations,
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      metric_fns=[])
  # Skip n in train.
  TaskRegistry.add(
      "cos_e_labels_skip{}_v001".format(n),
      t5.data.TfdsTask,
      tfds_name="cos_e:0.0.1",
      splits={"train": "train[{}:]".format(n)},
      text_preprocessor=functools.partial(
          preprocessors.cos_e, prefix="cos_e", drop_explanations=True),
      postprocess_fn=postprocessors.abstractive_explanations,
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      metric_fns=[])

# Note: cos_e has a validation set (we use the dev set for validation), but no
# test set.
TaskRegistry.add(
    "cos_e_eval_v001",
    TfdsTask,
    tfds_name="cos_e:0.0.1",
    text_preprocessor=preprocessors.cos_e,
    postprocess_fn=postprocessors.abstractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    splits=["validation"],
    metric_fns=[metrics.esnli_metric])

# ============== Zero Shot Transfer Tasks for eSNLI and CoS-E ==================

# Note: cos_e has a validation set (we use the dev set for validation), but no
# test set.
# CoS-E evaluation, with modified prefixes like e-SNLI.
TaskRegistry.add(
    "cos_e_eval_v001_like_esnli",
    TfdsTask,
    tfds_name="cos_e:0.0.1",
    text_preprocessor=functools.partial(
        preprocessors.cos_e, prefix="explain nli", question_prefix="premise:"),
    postprocess_fn=postprocessors.abstractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    splits=["validation"],
    metric_fns=[metrics.esnli_metric])

TaskRegistry.add(
    "esnli_v002_with_choices",
    TfdsTask,
    tfds_name="esnli:0.0.2",
    text_preprocessor=functools.partial(preprocessors.esnli,
                                        add_choices=True),
    postprocess_fn=postprocessors.abstractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[metrics.esnli_metric])

# e-SNLI with no explanations.
TaskRegistry.add(
    "esnli_v002_0_expln_with_choices",
    TfdsTask,
    tfds_name="esnli:0.0.2",
    text_preprocessor=functools.partial(
        preprocessors.esnli, prefix="nli", drop_explanations=True,
        add_choices=True),
    postprocess_fn=postprocessors.abstractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[metrics.esnli_metric])

# ======================== e-SNLI Corpus Task ==================================
TaskRegistry.add(
    "esnli_v002",
    TfdsTask,
    tfds_name="esnli:0.0.2",
    text_preprocessor=preprocessors.esnli,
    postprocess_fn=postprocessors.abstractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[metrics.esnli_metric])

# e-SNLI with no explanations.
TaskRegistry.add(
    "esnli_v002_0_expln",
    TfdsTask,
    tfds_name="esnli:0.0.2",
    text_preprocessor=functools.partial(
        preprocessors.esnli, prefix="nli", drop_explanations=True),
    postprocess_fn=postprocessors.abstractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[metrics.esnli_metric])

TaskRegistry.add(
    "esnli_eval_v002",
    TfdsTask,
    tfds_name="esnli:0.0.2",
    text_preprocessor=preprocessors.esnli,
    postprocess_fn=postprocessors.abstractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    splits=["validation", "test"],
    metric_fns=[metrics.esnli_metric])

n_esnli_explanations = [50000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100]
for n in n_esnli_explanations:
  # Take n in train.
  TaskRegistry.add(
      "esnli_explanations_take{}_v002".format(n),
      t5.data.TfdsTask,
      tfds_name="esnli:0.0.2",
      splits={"train": "train[0:{}]".format(n)},
      text_preprocessor=[preprocessors.esnli],
      postprocess_fn=postprocessors.abstractive_explanations,
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      metric_fns=[])
  # Skip n in train.
  TaskRegistry.add(
      "esnli_labels_skip{}_v002".format(n),
      t5.data.TfdsTask,
      tfds_name="esnli:0.0.2",
      splits={"train": "train[{}:]".format(n)},
      text_preprocessor=functools.partial(
          preprocessors.esnli, prefix="nli", drop_explanations=True),
      postprocess_fn=postprocessors.abstractive_explanations,
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      metric_fns=[])

mnli_config = tfds.text.glue.Glue.builder_configs["mnli"]
# pylint: disable=protected-access
TaskRegistry.add(
    "mnli_v002",
    TfdsTask,
    tfds_name="glue/mnli:1.0.0",
    text_preprocessor=functools.partial(
        t5_preprocessors.glue,
        benchmark_name="nli",
        label_names=mnli_config.label_classes),
    metric_fns=t5.data.tasks.GLUE_METRICS["mnli"],
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    postprocess_fn=t5.data.tasks._get_glue_postprocess_fn(mnli_config),
)
for mnli_eval_set in ("matched", "mismatched"):
  TaskRegistry.add(
      "mnli_explain_eval_%s_v002" % mnli_eval_set,
      TfdsTask,
      tfds_name="glue/mnli_%s:1.0.0" % mnli_eval_set,
      text_preprocessor=functools.partial(
          t5_preprocessors.glue,
          benchmark_name="explain nli",
          label_names=mnli_config.label_classes),
      metric_fns=[metrics.esnli_metric],
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      postprocess_fn=postprocessors.abstractive_explanations,
  )
# pylint: enable=protected-access

# ======================== Movie Rationales ======================
TaskRegistry.add(
    "movie_rationales_v010",
    TfdsTask,
    tfds_name="movie_rationales:0.1.0",
    text_preprocessor=preprocessors.extractive_explanations,
    postprocess_fn=postprocessors.extractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[metrics.extractive_explanations_metric])

TaskRegistry.add(
    "movie_rationales_v010_no_expl",
    TfdsTask,
    tfds_name="movie_rationales:0.1.0",
    text_preprocessor=functools.partial(
        preprocessors.extractive_explanations,
        drop_explanations=True,
        prefix="sentiment"),
    postprocess_fn=postprocessors.extractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[])

n_movie_explanations = [1000, 500, 200, 100]
for n in n_movie_explanations:
  # Take n in train.
  TaskRegistry.add(
      "movie_rationales_explanations_take{}_v010".format(n),
      t5.data.TfdsTask,
      tfds_name="movie_rationales:0.1.0",
      splits={"train": "train[0:{}]".format(n)},
      text_preprocessor=preprocessors.extractive_explanations,
      postprocess_fn=postprocessors.extractive_explanations,
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      metric_fns=[])
  # Skip n in train.
  TaskRegistry.add(
      "movie_rationales_labels_skip{}_v010".format(n),
      t5.data.TfdsTask,
      tfds_name="movie_rationales:0.1.0",
      splits={"train": "train[{}:]".format(n)},
      text_preprocessor=functools.partial(
          preprocessors.extractive_explanations, drop_explanations=True),
      postprocess_fn=postprocessors.extractive_explanations,
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      metric_fns=[])

TaskRegistry.add(
    "movie_rationales_eval_v010",
    TfdsTask,
    tfds_name="movie_rationales:0.1.0",
    text_preprocessor=preprocessors.extractive_explanations,
    postprocess_fn=postprocessors.extractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    splits=["validation", "test"],
    metric_fns=[metrics.extractive_explanations_metric])

# ======================= IMDB Movie Reviews =====================
TaskRegistry.add(
    "imdb_reviews_v100",
    TfdsTask,
    tfds_name="imdb_reviews:1.0.0",
    text_preprocessor=preprocessors.imdb_reviews,
    postprocess_fn=functools.partial(
        t5_postprocessors.string_label_to_class_id,
        label_classes=["negative", "positive"]),
    splits=["train", "test"],
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[t5_metrics.accuracy])

TaskRegistry.add(
    "imdb_reviews_eval_v100",
    TfdsTask,
    tfds_name="imdb_reviews:1.0.0",
    text_preprocessor=functools.partial(
        preprocessors.imdb_reviews, prefix="explain sentiment"),
    postprocess_fn=postprocessors.extractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[metrics.extractive_explanations_metric],
    splits=["test"],
)

# ======================== Amazon Reviews ======================
amazon_review_categories = [
    b.name for b in tfds.structured.AmazonUSReviews.builder_configs.values()]

for c in amazon_review_categories:
  TaskRegistry.add(
      "amazon_reviews_{}_v010".format(c.lower()),
      TfdsTask,
      tfds_name="amazon_us_reviews/{}:0.1.0".format(c),
      text_preprocessor=preprocessors.amazon_reviews,
      postprocess_fn=functools.partial(
          t5_postprocessors.string_label_to_class_id,
          label_classes=["negative", "positive"]),
      splits={"train": "train[10%:]", "validation": "train[:10%]"},
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      metric_fns=[t5_metrics.accuracy])

  TaskRegistry.add(
      "amazon_reviews_{}_eval_v010".format(c.lower()),
      TfdsTask,
      tfds_name="amazon_us_reviews/{}:0.1.0".format(c),
      text_preprocessor=functools.partial(
          preprocessors.amazon_reviews, prefix="explain sentiment"),
      postprocess_fn=postprocessors.extractive_explanations,
      splits={"validation": "train[:10%]"},
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      metric_fns=[metrics.extractive_explanations_metric])

# ======================== Eraser MultiRC ======================
TaskRegistry.add(
    "eraser_multi_rc_v011",
    TfdsTask,
    tfds_name="eraser_multi_rc:0.1.1",
    text_preprocessor=preprocessors.eraser_multi_rc,
    postprocess_fn=postprocessors.extractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    metric_fns=[metrics.extractive_explanations_metric])

n_multi_rc_explanations = [10000, 5000, 2000, 1000, 500, 200, 100]
for n in n_multi_rc_explanations:
  # Take n in train.
  TaskRegistry.add(
      "eraser_multi_rc_explanations_take{}_v011".format(n),
      t5.data.TfdsTask,
      tfds_name="eraser_multi_rc:0.1.1",
      splits={"train": "train[0:{}]".format(n)},
      text_preprocessor=preprocessors.eraser_multi_rc,
      postprocess_fn=postprocessors.extractive_explanations,
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      metric_fns=[])
  # Skip n in train.
  TaskRegistry.add(
      "eraser_multi_rc_labels_skip{}_v011".format(n),
      t5.data.TfdsTask,
      tfds_name="eraser_multi_rc:0.1.1",
      splits={"train": "train[{}:]".format(n)},
      text_preprocessor=functools.partial(
          preprocessors.eraser_multi_rc, drop_explanations=True),
      postprocess_fn=postprocessors.extractive_explanations,
      sentencepiece_model_path=DEFAULT_SPM_PATH,
      metric_fns=[])

TaskRegistry.add(
    "eraser_multi_rc_eval_v011",
    TfdsTask,
    tfds_name="eraser_multi_rc:0.1.1",
    text_preprocessor=preprocessors.eraser_multi_rc,
    postprocess_fn=postprocessors.extractive_explanations,
    sentencepiece_model_path=DEFAULT_SPM_PATH,
    splits=["validation", "test"],
    metric_fns=[metrics.extractive_explanations_metric])

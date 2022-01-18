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

"""WT5 tasks."""
import functools

from . import metrics
from . import postprocessors
from . import preprocessors
import seqio
from t5.data import get_default_vocabulary
from t5.data import glue_utils
from t5.data import postprocessors as t5_postprocessors
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics
import tensorflow_datasets as tfds


TaskRegistry = seqio.TaskRegistry

DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=True),
    "targets":
        seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=True)
}

# ======================== CoS-E Corpus Task ==================================
TaskRegistry.add(
    "cos_e_v001",
    source=seqio.TfdsDataSource(tfds_name="cos_e:0.0.1"),
    preprocessors=[
        preprocessors.cos_e,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.abstractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.esnli_metric])

# CoS-E with no explanations, and modified prefixes like e-SNLI.
TaskRegistry.add(
    "cos_e_v001_0_expln_like_esnli",
    source=seqio.TfdsDataSource(tfds_name="cos_e:0.0.1"),
    preprocessors=[
        functools.partial(
            preprocessors.cos_e,
            prefix="nli",
            question_prefix="premise:",
            drop_explanations=True),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.abstractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.esnli_metric])

n_cos_e_explanations = [5000, 2000, 1000, 500, 200, 100]
for n in n_cos_e_explanations:
  TaskRegistry.add(
      "cos_e_explanations_take{}_v001".format(n),
      source=seqio.TfdsDataSource(
          tfds_name="cos_e:0.0.1", splits={"train": "train[0:{}]".format(n)}),
      preprocessors=[
          preprocessors.cos_e,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=postprocessors.abstractive_explanations,
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])
  # Skip n in train.
  TaskRegistry.add(
      "cos_e_labels_skip{}_v001".format(n),
      source=seqio.TfdsDataSource(
          tfds_name="cos_e:0.0.1", splits={"train": "train[{}:]".format(n)}),
      preprocessors=[
          functools.partial(
              preprocessors.cos_e, prefix="cos_e", drop_explanations=True),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=postprocessors.abstractive_explanations,
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])

# Note: cos_e has a validation set (we use the dev set for validation), but no
# test set.
TaskRegistry.add(
    "cos_e_eval_v001",
    source=seqio.TfdsDataSource(tfds_name="cos_e:0.0.1", splits=["validation"]),
    preprocessors=[
        preprocessors.cos_e,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.abstractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.esnli_metric])

# ============== Zero Shot Transfer Tasks for eSNLI and CoS-E ==================

# Note: cos_e has a validation set (we use the dev set for validation), but no
# test set.
# CoS-E evaluation, with modified prefixes like e-SNLI.
TaskRegistry.add(
    "cos_e_eval_v001_like_esnli",
    source=seqio.TfdsDataSource(tfds_name="cos_e:0.0.1", splits=["validation"]),
    preprocessors=[
        functools.partial(
            preprocessors.cos_e,
            prefix="explain nli",
            question_prefix="premise:"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.abstractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.esnli_metric])

TaskRegistry.add(
    "esnli_v010_with_choices",
    source=seqio.TfdsDataSource(tfds_name="esnli:0.1.0"),
    preprocessors=[
        functools.partial(preprocessors.esnli, add_choices=True),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.abstractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.esnli_metric])

# e-SNLI with no explanations.
TaskRegistry.add(
    "esnli_v010_0_expln_with_choices",
    source=seqio.TfdsDataSource(tfds_name="esnli:0.1.0"),
    preprocessors=[
        functools.partial(
            preprocessors.esnli,
            prefix="nli",
            drop_explanations=True,
            add_choices=True),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.abstractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.esnli_metric])

# ======================== e-SNLI Corpus Task ==================================
TaskRegistry.add(
    "esnli_v010",
    source=seqio.TfdsDataSource(tfds_name="esnli:0.1.0"),
    preprocessors=[
        preprocessors.esnli,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.abstractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.esnli_metric])

# e-SNLI with no explanations.
TaskRegistry.add(
    "esnli_v010_0_expln",
    source=seqio.TfdsDataSource(tfds_name="esnli:0.1.0"),
    preprocessors=[
        functools.partial(
            preprocessors.esnli, prefix="nli", drop_explanations=True),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.abstractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.esnli_metric])

TaskRegistry.add(
    "esnli_eval_v010",
    source=seqio.TfdsDataSource(
        tfds_name="esnli:0.1.0", splits=["validation", "test"]),
    preprocessors=[
        preprocessors.esnli,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.abstractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.esnli_metric])

n_esnli_explanations = [50000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100]
for n in n_esnli_explanations:
  # Take n in train.
  TaskRegistry.add(
      "esnli_explanations_take{}_v010".format(n),
      source=seqio.TfdsDataSource(
          tfds_name="esnli:0.1.0", splits={"train": "train[0:{}]".format(n)}),
      preprocessors=[
          preprocessors.esnli,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=postprocessors.abstractive_explanations,
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])
  # Skip n in train.
  TaskRegistry.add(
      "esnli_labels_skip{}_v010".format(n),
      source=seqio.TfdsDataSource(
          tfds_name="esnli:0.1.0", splits={"train": "train[{}:]".format(n)}),
      preprocessors=[
          functools.partial(
              preprocessors.esnli, prefix="nli", drop_explanations=True),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=postprocessors.abstractive_explanations,
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])

mnli_config = tfds.text.glue.Glue.builder_configs["mnli"]
# pylint: disable=protected-access
TaskRegistry.add(
    "mnli_v002",
    source=seqio.TfdsDataSource(tfds_name="glue/mnli:1.0.0"),
    preprocessors=[
        functools.partial(
            t5_preprocessors.glue,
            benchmark_name="nli",
            label_names=mnli_config.label_classes),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=glue_utils.GLUE_METRICS["mnli"],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=glue_utils.get_glue_postprocess_fn(mnli_config),
)
for mnli_eval_set in ("matched", "mismatched"):
  TaskRegistry.add(
      "mnli_explain_eval_%s_v002" % mnli_eval_set,
      source=seqio.TfdsDataSource(tfds_name="glue/mnli_%s:1.0.0" %
                                  mnli_eval_set),
      preprocessors=[
          functools.partial(
              t5_preprocessors.glue,
              benchmark_name="explain nli",
              label_names=mnli_config.label_classes),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=[metrics.esnli_metric],
      output_features=DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=postprocessors.abstractive_explanations,
  )
# pylint: enable=protected-access

# ======================== Movie Rationales ======================
TaskRegistry.add(
    "movie_rationales_v010",
    source=seqio.TfdsDataSource(tfds_name="movie_rationales:0.1.0"),
    preprocessors=[
        preprocessors.extractive_explanations,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.extractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.extractive_explanations_metric])

TaskRegistry.add(
    "movie_rationales_v010_no_expl",
    source=seqio.TfdsDataSource(tfds_name="movie_rationales:0.1.0"),
    preprocessors=[
        functools.partial(
            preprocessors.extractive_explanations,
            drop_explanations=True,
            prefix="sentiment"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.extractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])

n_movie_explanations = [1000, 500, 200, 100]
for n in n_movie_explanations:
  # Take n in train.
  TaskRegistry.add(
      "movie_rationales_explanations_take{}_v010".format(n),
      source=seqio.TfdsDataSource(
          tfds_name="movie_rationales:0.1.0",
          splits={"train": "train[0:{}]".format(n)}),
      preprocessors=[
          preprocessors.extractive_explanations,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=postprocessors.extractive_explanations,
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])
  # Skip n in train.
  TaskRegistry.add(
      "movie_rationales_labels_skip{}_v010".format(n),
      source=seqio.TfdsDataSource(
          tfds_name="movie_rationales:0.1.0",
          splits={"train": "train[{}:]".format(n)}),
      preprocessors=[
          functools.partial(
              preprocessors.extractive_explanations, drop_explanations=True),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=postprocessors.extractive_explanations,
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])

TaskRegistry.add(
    "movie_rationales_eval_v010",
    source=seqio.TfdsDataSource(
        tfds_name="movie_rationales:0.1.0", splits=["validation", "test"]),
    preprocessors=[
        preprocessors.extractive_explanations,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.extractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.extractive_explanations_metric])

# ======================= IMDB Movie Reviews =====================
TaskRegistry.add(
    "imdb_reviews_v100",
    source=seqio.TfdsDataSource(
        tfds_name="imdb_reviews:1.0.0", splits=["train", "test"]),
    preprocessors=[
        preprocessors.imdb_reviews,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=functools.partial(
        t5_postprocessors.string_label_to_class_id,
        label_classes=["negative", "positive"]),
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[t5_metrics.accuracy])

TaskRegistry.add(
    "imdb_reviews_eval_v100",
    source=seqio.TfdsDataSource(
        tfds_name="imdb_reviews:1.0.0", splits=["test"]),
    preprocessors=[
        functools.partial(
            preprocessors.imdb_reviews, prefix="explain sentiment"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.extractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.extractive_explanations_metric],
)

# ======================== Amazon Reviews ======================
amazon_review_categories = [
    b.name for b in tfds.structured.AmazonUSReviews.builder_configs.values()]

for c in amazon_review_categories:
  TaskRegistry.add(
      "amazon_reviews_{}_v010".format(c.lower()),
      source=seqio.TfdsDataSource(
          tfds_name="amazon_us_reviews/{}:0.1.0".format(c),
          splits={
              "train": "train[10%:]",
              "validation": "train[:10%]"
          }),
      preprocessors=[
          preprocessors.amazon_reviews,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=functools.partial(
          t5_postprocessors.string_label_to_class_id,
          label_classes=["negative", "positive"]),
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[t5_metrics.accuracy])

  TaskRegistry.add(
      "amazon_reviews_{}_eval_v010".format(c.lower()),
      source=seqio.TfdsDataSource(
          tfds_name="amazon_us_reviews/{}:0.1.0".format(c),
          splits={"validation": "train[:10%]"}),
      preprocessors=[
          functools.partial(
              preprocessors.amazon_reviews, prefix="explain sentiment"),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=postprocessors.extractive_explanations,
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[metrics.extractive_explanations_metric])

# ======================== Eraser MultiRC ======================
TaskRegistry.add(
    "eraser_multi_rc_v011",
    source=seqio.TfdsDataSource(tfds_name="eraser_multi_rc:0.1.1"),
    preprocessors=[
        preprocessors.eraser_multi_rc,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.extractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.extractive_explanations_metric])

n_multi_rc_explanations = [10000, 5000, 2000, 1000, 500, 200, 100]
for n in n_multi_rc_explanations:
  # Take n in train.
  TaskRegistry.add(
      "eraser_multi_rc_explanations_take{}_v011".format(n),
      source=seqio.TfdsDataSource(
          tfds_name="eraser_multi_rc:0.1.1",
          splits={"train": "train[0:{}]".format(n)}),
      preprocessors=[
          preprocessors.eraser_multi_rc,
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=postprocessors.extractive_explanations,
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])
  # Skip n in train.
  TaskRegistry.add(
      "eraser_multi_rc_labels_skip{}_v011".format(n),
      source=seqio.TfdsDataSource(
          tfds_name="eraser_multi_rc:0.1.1",
          splits={"train": "train[{}:]".format(n)}),
      preprocessors=[
          functools.partial(
              preprocessors.eraser_multi_rc, drop_explanations=True),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      postprocess_fn=postprocessors.extractive_explanations,
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])

TaskRegistry.add(
    "eraser_multi_rc_eval_v011",
    source=seqio.TfdsDataSource(
        tfds_name="eraser_multi_rc:0.1.1", splits=["validation", "test"]),
    preprocessors=[
        preprocessors.eraser_multi_rc,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=postprocessors.extractive_explanations,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.extractive_explanations_metric])

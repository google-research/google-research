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

"""WT5 mixtures."""
import functools
from . import tasks

import t5

MixtureRegistry = t5.data.MixtureRegistry
TaskRegistry = t5.data.TaskRegistry


def _rate_num_movies(task, scale=1.0):
  del task
  return scale * 125000000.0


def _rate_num_input_examples(task):
  if "train" in task.splits:
    return float(task.num_input_examples("train"))
  elif "validation" in task.splits:
    return float(task.num_input_examples("validation"))
  else:
    raise ValueError("Task %s does not have a train or validation split." % (
        task.name))

# -------------------------- CoS-E --------------------------------------------
COS_E_SIZE = 9741

for n in tasks.n_cos_e_explanations:
  cos_e_n_explanations_tasks = [
      ("cos_e_explanations_take{}_v001".format(n), n),
      ("cos_e_labels_skip{}_v001".format(n), COS_E_SIZE-n),
      ("cos_e_eval_v001", COS_E_SIZE),
  ]

  MixtureRegistry.add(
      "cos_e_{}_explanations".format(n),
      cos_e_n_explanations_tasks
  )

# -------------------------- eSNLI --------------------------------------------
ESNLI_SIZE = 549367

for n in tasks.n_esnli_explanations:
  esnli_n_explanations_tasks = [
      ("esnli_explanations_take{}_v010".format(n), n),
      ("esnli_labels_skip{}_v010".format(n), ESNLI_SIZE-n),
      ("esnli_eval_v010", ESNLI_SIZE),
  ]
  MixtureRegistry.add(
      "esnli_{}_explanations".format(n),
      esnli_n_explanations_tasks
  )
  MixtureRegistry.add(
      "esnli_mnli_{}_explanations".format(n),
      esnli_n_explanations_tasks + [
          ("mnli_v002", 392702),
          ("mnli_explain_eval_matched_v002", 9815),
          ("mnli_explain_eval_mismatched_v002", 9815),
      ]
  )

MixtureRegistry.add(
    "esnli_mnli_all_explanations",
    ["esnli_v010", "mnli_v002",
     "mnli_explain_eval_matched_v002", "mnli_explain_eval_mismatched_v002"],
    default_rate=_rate_num_input_examples,
)

MixtureRegistry.add(
    "imdb_reviews_movie_rationales",
    ["imdb_reviews_v100", "movie_rationales_v010", "imdb_reviews_eval_v100"],
    default_rate=_rate_num_input_examples,
)

# ------------------------- eSNLI to CoS-E explanation Transfer ---------------
MixtureRegistry.add(
    "esnli_cos_e_transfer",
    [("esnli_v010", ESNLI_SIZE),
     ("esnli_v010_0_expln", ESNLI_SIZE),
     ("cos_e_v001_0_expln_like_esnli", COS_E_SIZE),
     ("cos_e_eval_v001_like_esnli", 1221)],
    default_rate=_rate_num_input_examples,
)

MixtureRegistry.add(
    "esnli_cos_e_transfer_choices",
    [("esnli_v010_with_choices", ESNLI_SIZE),
     ("esnli_v010_0_expln_with_choices", ESNLI_SIZE),
     ("cos_e_v001_0_expln_like_esnli", COS_E_SIZE),
     ("cos_e_eval_v001_like_esnli", 1221)],
    default_rate=_rate_num_input_examples,
)

for n in tasks.n_movie_explanations:
  movie_size = 1600
  movie_rationales_n_explanations_tasks = [
      ("movie_rationales_explanations_take{}_v010".format(n), n),
      ("movie_rationales_labels_skip{}_v010".format(n), movie_size-n),
      ("movie_rationales_eval_v010", movie_size),
  ]
  MixtureRegistry.add(
      "movie_rationales_{}_explanations".format(n),
      movie_rationales_n_explanations_tasks
  )
# ----------------------------- Amazon Reviews ---------------------------------
amazon_reviews_train_tasks = []
amazon_reviews_eval_tasks = []
for c in tasks.amazon_review_categories:
  amazon_reviews_train_tasks.append("amazon_reviews_{}_v010".format(c.lower()))
  amazon_reviews_eval_tasks.append(
      "amazon_reviews_{}_eval_v010".format(c.lower()))

MixtureRegistry.add("amazon_reviews", amazon_reviews_train_tasks,
                    default_rate=_rate_num_input_examples)

MixtureRegistry.add(
    "amazon_reviews_movie_rationales",
    amazon_reviews_train_tasks + amazon_reviews_eval_tasks +
    ["movie_rationales_v010"],
    default_rate=_rate_num_input_examples,
)

MixtureRegistry.add(
    "amazon_books_movies_equal", [
        "amazon_reviews_books_v1_00_v010", "movie_rationales_v010",
        "amazon_reviews_books_v1_00_eval_v010"
    ],
    default_rate=1.0)

for factor in [10, 20, 30, 40]:
  amazon_reviews_tasks = [
      (t, _rate_num_input_examples) for t in amazon_reviews_train_tasks
  ]

  # Running eval only on a few categories because otherwise evaluation takes a
  # very long time.
  amazon_reviews_tasks.append(
      ("amazon_reviews_books_v1_00_eval_v010", _rate_num_input_examples))
  amazon_reviews_tasks.append(
      ("amazon_reviews_electronics_v1_00_eval_v010", _rate_num_input_examples))
  amazon_reviews_tasks.append(
      ("amazon_reviews_apparel_v1_00_eval_v010", _rate_num_input_examples))

  amazon_reviews_tasks.append(
      ("movie_rationales_v010",
       functools.partial(_rate_num_movies, scale=factor)))

  MixtureRegistry.add("amazon_reviews_movies_r{}".format(int(factor)),
                      amazon_reviews_tasks)

  # Remove the movie_rationales_v010 scaled at factor, for transfer mixture.
  amazon_reviews_tasks.pop()
  amazon_reviews_tasks.append(
      ("movie_rationales_v010",
       functools.partial(_rate_num_movies, scale=(factor/2))))

  amazon_reviews_tasks.append(
      ("movie_rationales_v010_no_expl",
       functools.partial(_rate_num_movies, scale=(factor/2))))

  MixtureRegistry.add("amazon_reviews_movies_transfer_r{}".format(int(factor)),
                      amazon_reviews_tasks)

for n in tasks.n_multi_rc_explanations:
  multi_rc_size = 24029
  eraser_multi_rc_n_explanations_tasks = [
      ("eraser_multi_rc_explanations_take{}_v011".format(n), n),
      ("eraser_multi_rc_labels_skip{}_v011".format(n), multi_rc_size-n),
      ("eraser_multi_rc_eval_v011", multi_rc_size),
  ]
  MixtureRegistry.add(
      "eraser_multi_rc_rationales_{}_explanations".format(n),
      eraser_multi_rc_n_explanations_tasks
  )

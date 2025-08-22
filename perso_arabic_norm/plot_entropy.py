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

r"""Prepare a plot of entropy/perplexity vs. ngram order.

Example:
--------
  python plot_entropy.py \
    --results_dir data/ngrams/results/pure_baselines \
    --language_list ks,pnb \
    --output_plots_dir /tmp

Dependencies:
-------------
  absl
  pandas
  seaborn
"""

from typing import Sequence

import collections
import logging
import os
import pathlib

from absl import app
from absl import flags

import pandas as pd
import seaborn as sns

import utils

flags.DEFINE_string(
    "results_dir", "",
    "Directory containing individual tab-separated (tsv) files containing the "
    "results for a particular language and ngram order.")

flags.DEFINE_list(
    "language_list", [],
    "List of strings containing language codes that are assumed to be prefixes "
    "of the tsv files containing the results.")

flags.DEFINE_string(
    "output_plots_dir", "",
    "Specifies the directory that will contain the plots.")

flags.DEFINE_boolean(
    "use_perplexities", False,
    "Use perplexities instead of entropies.")

flags.DEFINE_boolean(
    "word_models", False,
    "The models are word- rather than character-based.")

FLAGS = flags.FLAGS


def _load_entropies(file_path):
  """Loads perplexities from a given file and returns them as entropies.

  Returns the tuple consisting of the ngram order and the corresponding
  entropy array.
  """
  order = utils.ngram_order_from_filename(file_path)
  return order, utils.read_entropies(file_path,
                                     as_ppl=FLAGS.use_perplexities)


def _process_dir(directory):
  """Processes all the results files in the supplied directory."""
  pathlist = pathlib.Path(directory).rglob("*.tsv")
  results_per_language = collections.defaultdict(list)
  for path in pathlist:
    file_path = str(path)
    filename = os.path.basename(file_path)
    for language in FLAGS.language_list:
      if filename.startswith(language):
        order, entropies = _load_entropies(file_path)
        results_per_language[language].append((order, list(entropies)))
        break

  # Sort the results by order.
  for language in results_per_language:
    results_per_language[language].sort(key=lambda x: x[0])
  return results_per_language


def _build_dataframe(results_per_language):
  """Processes dictionary of all languages/orders/entropies into dataframe."""
  col_languages = []
  col_orders = []
  col_entropies = []
  for language in sorted(results_per_language.keys()):
    for order, entropies in results_per_language[language]:
      for entropy in entropies:
        col_languages.append(language)
        col_orders.append(order)
        col_entropies.append(entropy)
  return pd.DataFrame(data = {
      "language" : col_languages,
      "order" : col_orders,
      "entropy" : col_entropies,
  })


def _plot_ylabel():
  """Returns y-label for the plot."""
  return "perplexity" if FLAGS.use_perplexities else (
      "entropy (bits/%s)" % ("word" if FLAGS.word_models else "char"))


def _plot_filename_suffix():
  """Returns suffix for the plot file name."""
  return "ppl" if FLAGS.use_perplexities else "ent"


def _plot_individual(df, language):
  """Plots individual language."""
  ax = sns.lineplot(x="order", y="entropy",
                    data=df[df["language"] == language])
  ax.set(ylabel=_plot_ylabel())
  fig = ax.get_figure()
  plot_file = os.path.join(FLAGS.output_plots_dir,
                           language + "_%s.pdf" % _plot_filename_suffix())
  logging.info(f"Saving {plot_file} ...")
  fig.savefig(plot_file, bbox_inches="tight", dpi=600)
  fig.clear()


def _plot_all(df):
  """Plots all languages on a single plot."""
  ax = sns.relplot(x="order", y="entropy", hue="language", style="language",
                   kind="line", facet_kws={"legend_out": True}, markers=True,
                   data=df)
  ax.set(ylabel=_plot_ylabel())
  fig = ax.figure
  plot_file = os.path.join(FLAGS.output_plots_dir,
                           "all_%s.pdf" % _plot_filename_suffix())
  logging.info(f"Saving {plot_file} ...")
  fig.savefig(plot_file, bbox_inches="tight", dpi=600)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not FLAGS.results_dir:
    raise app.UsageError("Specify --results_dir [DIR]!")
  if not FLAGS.language_list:
    raise app.UsageError("Specify --language_list [LANG1],...")
  if not FLAGS.output_plots_dir:
    raise app.UsageError("Specify --output_plots_dir [DIR]!")

  if not os.path.isdir(FLAGS.results_dir):
    raise ValueError(f"Directory {FLAGS.results_dir} is missing")
  results_per_language = _process_dir(FLAGS.results_dir)
  df = _build_dataframe(results_per_language)

  # Save dataframe to tsv file.
  output_tsv_file = os.path.join(FLAGS.output_plots_dir, "report.tsv")
  logging.info("Saving the report to %s ...", output_tsv_file)
  df.to_csv(output_tsv_file, sep="\t", index=None, encoding="utf-8")

  # Save plots.
  for language in FLAGS.language_list:
    _plot_individual(df, language)
  _plot_all(df)


if __name__ == "__main__":
  app.run(main)

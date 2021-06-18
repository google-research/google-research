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

r"""Main tool for training and evaluating the neural model.

The neural model is meant for approximating the neural measures of logography
(denoted $S$ in the respective paper).

Please see the `README.md` file in this directory for the examples of how to
run this tool.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import flags

import tensorflow.compat.v1 as tf  # tf

import homophonous_logography.neural.corpus as data
import homophonous_logography.neural.eval as evaluate
import homophonous_logography.neural.model as model_lib
import homophonous_logography.neural.utils as utils

flags.DEFINE_bool("train", True,
                  "Train the model.")

flags.DEFINE_bool("eval", False,
                  "Evaluate the model.")

flags.DEFINE_bool("show_plots", False, "Whether or not to display plots.")

flags.DEFINE_bool("deviation_only_for_correct", True,
                  "In evaluation, compute the deviation only for the correctly "
                  "predicted outputs.")

flags.DEFINE_bool("report_type_stats", False,
                  "Average S value over types, not tokens")

flags.DEFINE_list("languages", ["middle_persian"],
                  "Comma-separated list of language names.")

flags.DEFINE_string("languages_file", "",
                    ("Specifies the list of languages as a newline-separated "
                     "list in a file, rather than in a command line."))

flags.DEFINE_string("direction", "written",
                    "Direction: written (p2g) or pronounced (g2p).")

flags.DEFINE_string("datasets_dir", None,
                    ("By default, the datasets are fetched remotely. If "
                     "specified, the datasets will be read from the specified "
                     "directory instead."))

flags.DEFINE_integer("num_epochs", 40,
                     "Number of training epochs.")

flags.DEFINE_integer("batch_size", 256,
                     "Training batch size.")

flags.DEFINE_string("model_dir", "/tmp/logo-models",
                    ("Directory containing training artifacts (checkpoints and "
                     "so on)."))

flags.DEFINE_integer("max_sentence_len", -1,
                     "If set, the maximum sentence length (in tokens) for "
                     "training and evaluation: overrides any internal "
                     "language-specific settings.")

flags.DEFINE_integer("ntest", -1,
                     "Number of test examples to evaluate "
                     "(defaults to whole set)")

flags.DEFINE_integer("window", -1,
                     "If set to a number greater than 0, predict target "
                     "from a window of input words.  "
                     "Note that this includes a space token.")

FLAGS = flags.FLAGS

tf.enable_eager_execution()

# Note that with setting N_TEST for Finnish to 400 or more we get a NaN result
# for the ratio: presumably there is one bad test item somewhere ?
# Check out np.nansum().

_PRINT_PREDICTIONS = True
_COMPUTE_DEVIATION = True
_DEVIATION_MASK_SIGMA = 0.3
_SIMPLE_SKEW = True
_PRINT_ATTENTION = False
_LOWER = False  # Middle Persian only.
_FIGSIZE = (30, 75)


# These were determined via trial and error to be reasonable limits in the
# Colab. These can be overridden here by setting --max_sentence_len.
_LSPEC_LENGTHS = {
    "english": 30,
    "french": 30,
    "chinese": 30,
    "chinese-cangjie": 30,
    "chinese-tok": 20,
    "chinese-tok-cangjie": 17,  # 20, but this takes way too long.
    "finnish": 20,
    "hebrew": 20,
    "japanese": 37,
    "japanese-cangjie": 37,
    "korean-jamo": 15,
    "middle_persian": 60
}


def _get_corpus_and_model(language):
  """Returns the tuple with the corpus and model for the given language."""
  # Read the corpus.
  try:
    if FLAGS.max_sentence_len > -1:
      max_sen_len = FLAGS.max_sentence_len
    else:
      max_sen_len = _LSPEC_LENGTHS[language]
  except KeyError:
    max_sen_len = 15
  corpus = data.Corpus(data.read_corpus(
      language, lower=_LOWER, max_length=max_sen_len,
      datasets_dir=FLAGS.datasets_dir))
  input_symbols = (corpus.pronounce_symbol_table
                   if FLAGS.direction == "written"
                   else corpus.written_symbol_table)
  output_symbols = (corpus.written_symbol_table
                    if FLAGS.direction == "written"
                    else corpus.pronounce_symbol_table)

  # Make model.
  if FLAGS.direction == "written":
    short_direction = "p2g"
  else:
    short_direction = "g2p"
  model = model_lib.Seq2SeqModel(
      batch_size=FLAGS.batch_size,
      input_symbols=input_symbols,
      output_symbols=output_symbols,
      model_dir=FLAGS.model_dir,
      name="{}_{}".format(language, short_direction))

  return corpus, model


def _test_language(language, corpus, model,
                   print_predictions=False,
                   show_plots=False,
                   compute_deviation=True,
                   deviation_only_for_correct=True,
                   simple_skew=False):
  """Runs model evaluation."""
  # Create test log that also redirects to stdout.
  stdout_file = sys.stdout
  logfile = os.path.join(model.checkpoint_dir, "eval.log")
  print("Test log: {}".format(logfile))
  sys.stdout = utils.DualLogger(logfile)

  print("Window size: {}".format(FLAGS.window))
  print("# test examples: {}".format(FLAGS.ntest))
  test_examples = data.test_examples(corpus, FLAGS.direction,
                                     window=FLAGS.window)
  indices = data.random_test_indices(test_examples, k=FLAGS.ntest)
  tot, cor, rat, nrat = evaluate.eval_and_plot(
      model, test_examples, indices,
      show_plots=show_plots,
      print_predictions=print_predictions,
      print_attention=_PRINT_ATTENTION,
      compute_deviation=compute_deviation,
      deviation_mask_sigma=_DEVIATION_MASK_SIGMA,
      deviation_only_for_correct=deviation_only_for_correct,
      simple_skew=simple_skew,
      report_type_stats=FLAGS.report_type_stats,
      figsize=_FIGSIZE)
  print("*" * 80)
  print("Language: {}".format(language))
  print("Total non-trivial predictions: {}".format(tot))
  print("Accuracy: {}".format(cor))
  print("Ratio: {}".format(rat))
  print("tf.reduce_max'ed normalized ratio: {}".format(nrat))

  # Restore stdout.
  sys.stdout = stdout_file


def _train_and_test(language):
  """Depending on the flags trains and tests a single language."""
  corpus, model = _get_corpus_and_model(language)

  if FLAGS.train:
    print("{}: Training. Checkpoints can be found in {} ...".format(
        language, model.checkpoint_dir))
    # TODO(agutkin,rws): Note, it looks like the training does not get restarted
    # from the last saved checkpoint. It probably makes sense to fix this.
    model.train(corpus, epochs=FLAGS.num_epochs,
                direction=FLAGS.direction, window=FLAGS.window)

  # Evaluate the model.
  if FLAGS.eval:
    print("{}: Evaluating from checkpoint in \"{}\" ...".format(
        language, model.checkpoint_dir))
    _test_language(language, corpus, model,
                   show_plots=FLAGS.show_plots,
                   print_predictions=_PRINT_PREDICTIONS,
                   compute_deviation=_COMPUTE_DEVIATION,
                   deviation_only_for_correct=FLAGS.deviation_only_for_correct,
                   simple_skew=_SIMPLE_SKEW)


def _language_list():
  """Returns configured list of languages."""
  if FLAGS.languages_file:
    languages = []
    with open(FLAGS.languages_file, "r", encoding="utf-8") as f:
      lines = [line.strip() for line in f.read().split("\n")]
      for line in lines:
        if line.startswith("#"):
          continue
        languages.append(line)
  else:
    languages = FLAGS.languages
  return languages


def main(unused_args):
  # Check flags.
  if not FLAGS.train and not FLAGS.eval:
    print("Enable either --train or --eval or both!")
    return
  if not FLAGS.model_dir:
    print("Model directory cannot be empty!")
    return
  if not FLAGS.languages and not FLAGS.languages_file:
    print("No languages specified with --languages!")
    return

  # Train and test. At least one mode of the two has to be enabled.
  languages = _language_list()
  print("=== Languages: {}".format(" ".join(languages)))
  for language in languages:
    _train_and_test(language)


if __name__ == "__main__":
  tf.app.run()

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

"""Locally decodes test set given a checkpoint."""

import csv
import collections
import glob
import math
import os
import sys

# pylint: disable=unused-import
import feature_neighborhood_flags
from google.protobuf import text_format

from lingvo import compat as tf
from lingvo import model_registry
from lingvo.core import cluster_factory
from lingvo.core import py_utils

from tensorflow.core.protobuf import saver_pb2  # pylint: disable=g-direct-tensorflow-import

tf.flags.DEFINE_integer(
    "ckpt_limit", -1, "If set, find largest checkpoint with "
    "stamp no larger than this.")
tf.flags.DEFINE_string(
    "ckpt", None, "Path to checkpoint, or path to directory "
    "containing checkpoints if ckpt_limit is set.")
tf.flags.DEFINE_string(
    "neighbor_attention_file", None,
    "If not None, use the neighbor data and write "
    "out the neighbor attention tensors to this file.")
tf.flags.DEFINE_string("model", None, "Name of the model to use to decode.")
tf.flags.DEFINE_string("decode_dir", None,
                       "Path to output directory for decoding.")
tf.flags.DEFINE_integer("beam_size", 8, "Beam search beam size.")
tf.flags.DEFINE_integer(
    "num_samples", 0,
    "Number of examples to decode. Keep the default if you need to process "
    "all the samples in the dataset but you don't know the size in advance.")
tf.flags.DEFINE_boolean(
    "inference", False,
    "In inference mode the ground truth labels are not known. Instead of "
    "computing the accuracy metrics and such, simply generate a two-column "
    "TSV file containing cognate IDs and corresponding cognate hypotheses.")

FLAGS = tf.flags.FLAGS

# Disabling eager execution is required to keep the old
# `tf.compat.v1.train.Saver` API working. Ideally a better solution will be
# to rewrite this using newer recommended `tf.train.Checkpoint` API.
tf.compat.v1.disable_eager_execution()


def get_dataset_info():
  """Returns the number of records and all languages in the dataset."""
  # Note: this is a compat version because we are not executing in eager mode.
  _, file_path = py_utils.RecordFormatFromFilePattern(
      FLAGS.feature_neighborhood_test_path)
  iterator = tf.compat.v1.data.make_one_shot_iterator(
      tf.data.TFRecordDataset(file_path))
  next_batch = iterator.get_next()
  num_records = 0
  with tf.compat.v1.Session() as sess:
    try:
      while True:
        sess.run(next_batch)
        num_records = num_records + 1
    except tf.errors.OutOfRangeError:
      pass
  return num_records


class FeatureNeighborhoodModelDecoder:
  """Simple decoder for FeatureNeighborhoodModel."""

  def __init__(self):
    # TODO(llion): Find a more sensible fix.
    cluster_factory.SetRequireSequentialInputOrder(True).__enter__()
    checkpoint_glob = sorted(
        glob.glob(os.path.join(FLAGS.ckpt, "ckpt-*.index")))
    if FLAGS.ckpt_limit == -1:
      self._ckpt = checkpoint_glob[-1].replace(".index", "")
    else:
      last_ckpt = None
      for idx in checkpoint_glob:
        ckpt_base = idx.replace(".index", "")
        value = int(ckpt_base.split("/")[-1].replace("ckpt-", ""))
        if value > FLAGS.ckpt_limit:
          break
        last_ckpt = ckpt_base
      assert last_ckpt is not None
      self._ckpt = last_ckpt
    self._decode_path = FLAGS.decode_dir or os.path.dirname(FLAGS.ckpt)
    sys.stderr.write("Using checkpoint: {}\n".format(self._ckpt))

  def _get_model(self):
    """Fetch model for decoding."""
    name = FLAGS.model
    self.is_transformer = "Transformer" in FLAGS.model
    p = model_registry.GetParams(
        "feature_neighborhood_model_config." + name,
        "Train")
    p.is_inference = True
    p.input.file_pattern = FLAGS.feature_neighborhood_test_path
    # Send it round twice so we can get the last few examples.
    p.input.repeat_count = 2
    p.cluster.require_sequential_input_order = True
    if self.is_transformer:
      p.task.beam_size = FLAGS.beam_size
    mdl = p.Instantiate()
    mdl.ConstructFPropGraph()
    return mdl.GetTask()

  # pylint: disable=missing-function-docstring
  def decode(self):
    tf.logging.info("Getting dataset info ...")
    num_records = get_dataset_info()
    tf.logging.info("Found %d records.", num_records)
    if FLAGS.num_samples == 0:
      FLAGS.num_samples = num_records
    assert FLAGS.num_samples != 0, "Number of examples to decode must be set."

    if "golden" in FLAGS.feature_neighborhood_test_path:
      decode_prefix = "golden_"
    else:
      decode_prefix = ""

    mdl = self._get_model()
    output_attention = mdl.params.use_neighbors
    if output_attention:
      attention_path = FLAGS.neighbor_attention_file
      if not attention_path:
        attention_path = os.path.join(FLAGS.decode_dir,
                                      "neighbor_attention.txt")
      neighbor_stream = open(attention_path, "w", encoding="utf8")

    svr = tf.train.Saver(
        sharded=True,
        max_to_keep=3,
        keep_checkpoint_every_n_hours=0.5,
        pad_step_number=True,
        write_version=saver_pb2.SaverDef.V2)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      def run_eval(sess, mdl):
        return sess.run(
            [mdl.loss, mdl.per_example_tensors, mdl.prediction_values])

      def strip_at_s_boundary(decoded):
        result = []
        for d in decoded:
          if d == "</s>":
            break
          result.append(d)
        return result

      def collect_neighbor_attention_tensors(prediction_values, batch_size):
        return [
            prediction_values.neighbor_attention[b, :]
            for b in range(batch_size)
        ]

      def decode_strings(array, symbol_table, joiner=""):
        if not symbol_table:
          return list(array)
        else:
          return clean_string(
              joiner.join(
                  strip_at_s_boundary(symbol_table.find(s) for s in array)))

      def collect_batch_data(prediction_values, b):
        neighbor_spellings = []
        neighbor_pronunciations = []
        output_joiner = " " if FLAGS.split_output_on_space else ""
        for n in range(mdl.params.max_neighbors):
          neighbor_spellings.append(
              decode_strings(prediction_values.batch.neighbor_spellings[b][n],
                             mdl.params.input_symbols))
        for n in range(mdl.params.max_neighbors):
          neighbor_pronunciations.append(
              decode_strings(
                  prediction_values.batch.neighbor_pronunciations[b][n],
                  mdl.params.output_symbols, output_joiner))
        return (neighbor_spellings, neighbor_pronunciations)

      # TODO(rws):
      #
      # In the next two nrow is max_pronunciation_len and ncol is
      # max_neighbors. That means that each neighbor's contribution is going
      # down the columns. compute_neighbor_activations processes this in the
      # correct order, so perhaps we should also do the transpose on the R
      # matrix.
      def create_r_matrix(array, nrow, ncol):
        assert array.shape == (nrow, ncol)
        values = []
        for row in range(nrow):
          for col in range(ncol):
            values.append(str(array[row, col]))
        tmpl = "matrix(c({}), byrow=TRUE, ncol={})"
        return tmpl.format(", ".join(values), ncol)

      def compute_neighbor_activations(array, nrow, num_real_neighbors):
        svalues = []
        if num_real_neighbors == 0:
          return svalues
        assert array.shape == (nrow, mdl.params.max_neighbors)
        # Hack to make sure we don't attempt to take the max of a zero
        # array, since numpy gets unhappy if you do that.
        values = array.sum(axis=0)[:num_real_neighbors]
        max_activation = values.max()
        for v in values:
          if math.isclose(v, max_activation):
            svalues.append("={:.5f}".format(v))
          else:
            svalues.append(" {:.5f}".format(v))
        return svalues

      def clean_string(s):
        return s.replace("</s>", "").replace("<spc>", " ").strip()

      decode_path = os.path.join(
          self._decode_path,
          decode_prefix + "decode_{}.txt".format(FLAGS.ckpt_limit))
      decode_file = open(decode_path, "w", encoding="utf8")

      def _print(output):
        print(output)
        decode_file.write(output + "\n")

      _print("Results from: {}\n".format(self._ckpt))
      _print("Decoding to:" + self._decode_path)
      if output_attention:
        _print("Decoding Attention to: " + attention_path)

      svr.restore(sess, self._ckpt)
      total = 0
      correct = 0
      lang_total = collections.defaultdict(int)
      lang_correct = collections.defaultdict(int)
      num_samples = FLAGS.num_samples
      output_joiner = " " if FLAGS.split_output_on_space else ""
      hypotheses = []
      while True:
        try:
          _, per_example_tensors, prediction_values = run_eval(sess, mdl)
          mdl.ProcessFPropResults(sess, 0, None, None)
          strings = mdl.print_input_and_output_tensors(per_example_tensors)
          batch_size = len(strings.inp)
          if output_attention and not self.is_transformer:
            neighbor_attention_tensors = collect_neighbor_attention_tensors(
                prediction_values, batch_size)
          for i in range(batch_size):
            _print("*" * 80)
            input_string = clean_string("".join(strings.inp[i]))
            _print(" inp: {}".format(input_string))
            ref = clean_string(output_joiner.join(strings.ref[i]))
            hyp = clean_string(output_joiner.join(
                strip_at_s_boundary(strings.hyp[i])))
            _print(" ref: {}".format(ref))
            tag = "*"
            if ref == hyp:
              correct += 1
              lang_correct[input_string] += 1
              tag = " "
            _print("{}hyp: {}".format(tag, hyp))
            hypotheses.append((strings.cognate_id[i], input_string, hyp))
            if "ave_entropy" in per_example_tensors:
              _print(" ent: {}".format(per_example_tensors["ave_entropy"][i]))
            if "ref_ave_entropy" in per_example_tensors:
              _print(" ref_ent: {}".format(
                  per_example_tensors["ref_ave_entropy"][i]))
            if "beam_scores" in per_example_tensors:
              _print(" beam: {}".format(
                  list(per_example_tensors["beam_scores"][i])))
            total += 1
            lang_total[input_string] += 1
            if mdl.params.use_neighbors:
              neighbor_spellings, neighbor_pronunciations = collect_batch_data(
                  prediction_values, i)
              # Compute the actual number of real neighbors:
              num_real_neighbors = 0
              for n in range(mdl.params.max_neighbors):
                if "<pad>" in neighbor_spellings[n]:
                  break
                num_real_neighbors += 1
              if output_attention:
                if self.is_transformer:
                  # (max_pronunciation_len, max_spelling_len + max_neighbors*2)
                  neighbor_stream.write(
                      "neighbor_attention_tensor:\n{}\n{}\n".format(
                          input_string,
                          per_example_tensors["attention"][i].tolist()))
                  # .to_list so that we get command and all the numbers
                  neighbor_activations = [1] * len(neighbor_spellings)
                  ave_att = (per_example_tensors["attention"][i].mean(axis=0))
                  max_spelling_len = mdl.params.max_spelling_len
                  max_neighbors = mdl.params.max_neighbors
                  # Add the attentions to the neighbour spelling and pron only
                  ave_att = ave_att[max_spelling_len:]
                  neighbor_activations = (
                      ave_att[0:max_neighbors] +
                      ave_att[max_neighbors:2 * max_neighbors])
                else:
                  neighbor_activations = compute_neighbor_activations(
                      neighbor_attention_tensors[i],
                      mdl.params.max_pronunciation_len, num_real_neighbors)
                  neighbor_stream.write(
                      "neighbor_attention_tensor:\n{}\n{}\n".format(
                          input_string,
                          create_r_matrix(neighbor_attention_tensors[i],
                                          mdl.params.max_pronunciation_len,
                                          mdl.params.max_neighbors)))
              else:
                neighbor_activations = [0] * len(neighbor_spellings)

              for n in range(num_real_neighbors):
                _print("{}\t{}\t{}".format(
                    neighbor_activations[n], neighbor_spellings[n],
                    neighbor_pronunciations[n]))

            _print(str(total))
            if total == num_samples:
              raise tf.errors.OutOfRangeError(None, None, "I'm done")
        # The InvalidArgumentError will get hit if we have a trailing
        # smaller batch size.
        except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError):
          break

      if not FLAGS.inference:
        accuracy = correct / total
        _print("Correct: {}/{}".format(correct, total))
        _print("Accuracy = {:.6f} (Error Rate = {:.6f})".format(
            accuracy, 1.0 - accuracy))
        # Per-language accuracies.
        if len(lang_correct) > 1:
          _print("----")
          for lang in sorted(lang_correct.keys()):
            lang_accuracy = lang_correct[lang] / lang_total[lang]
            _print("%s\tCorrect: %d/%d" % (lang, lang_correct[lang],
                                          lang_total[lang]))
            _print("{}\tError Rate: {:.6f}".format(lang, 1.0 - lang_accuracy))

      decode_file.close()
      if output_attention:
        neighbor_stream.close()

      # Also output to separate file so that we don't have to scroll to the
      # bottom of a long file.
      if not FLAGS.inference:
        decode_path = os.path.join(
            self._decode_path,
            decode_prefix + "results_{}.txt".format(FLAGS.ckpt_limit))
        with open(decode_path, "w", encoding="utf8") as f:
          f.write("Results from: {}\n".format(self._ckpt))
          f.write("Correct: {}/{}\n".format(correct, total))
          f.write("Accuracy = {:.6f} (Error Rate = {:.6f})\n".format(
              accuracy, 1.0 - accuracy))

      # Dump the results in SIGTYP results TSV format.
      languages = sorted(list(set([language for _, language, _ in hypotheses])))
      decode_path = os.path.join(
          self._decode_path,
          decode_prefix + "results_{}.tsv".format(FLAGS.ckpt_limit))
      with open(decode_path, "w", encoding="utf8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["COGID"] + languages)
        for cog_id, lang, hyp in hypotheses:
          row = [cog_id]
          lang_id = languages.index(lang)
          for i in range(len(languages)):
            if i == lang_id:
              row.append(hyp)
            else:
              row.append(None)
          writer.writerow(row)


def main(unused_argv):
  decoder = FeatureNeighborhoodModelDecoder()
  with cluster_factory.SetEval(mode=True):
    decoder.decode()


if __name__ == "__main__":
  tf.flags.mark_flag_as_required("ckpt")
  tf.flags.mark_flag_as_required("feature_neighborhood_test_path")
  tf.flags.mark_flag_as_required("input_symbols")
  tf.flags.mark_flag_as_required("output_symbols")
  tf.app.run(main)

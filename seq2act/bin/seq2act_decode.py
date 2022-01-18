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

"""seq2act decoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow.compat.v1 as tf
from seq2act.models import input as input_utils
from seq2act.models import seq2act_estimator
from seq2act.models import seq2act_model
from seq2act.utils import decode_utils

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("beam_size", 1, "beam size")
flags.DEFINE_string("problem", "android_howto", "problem")
flags.DEFINE_string("data_files", "", "data_files")
flags.DEFINE_string("checkpoint_path", "", "checkpoint_path")
flags.DEFINE_string("output_dir", "", "output_dir")
flags.DEFINE_integer("decode_batch_size", 1, "decode_batch_size")


def get_input(hparams, data_files):
  """Get the input."""
  if FLAGS.problem == "pixel_help":
    data_source = input_utils.DataSource.PIXEL_HELP
  elif FLAGS.problem == "android_howto":
    data_source = input_utils.DataSource.ANDROID_HOWTO
  elif FLAGS.problem == "rico_sca":
    data_source = input_utils.DataSource.RICO_SCA
  else:
    raise ValueError("Unrecognized test: %s" % FLAGS.problem)
  tf.logging.info("Testing data_source=%s data_files=%s" % (
      FLAGS.problem, data_files))
  dataset = input_utils.input_fn(
      data_files,
      FLAGS.decode_batch_size,
      repeat=1,
      data_source=data_source,
      max_range=hparams.max_span,
      max_dom_pos=hparams.max_dom_pos,
      max_pixel_pos=(
          hparams.max_pixel_pos),
      load_extra=True,
      load_dom_dist=(hparams.screen_encoder == "gcn"))
  iterator = tf.data.make_one_shot_iterator(dataset)
  features = iterator.get_next()
  return features


def generate_action_mask(features):
  """Computes the decode mask from "task" and "verb_refs"."""
  eos_positions = tf.to_int32(tf.expand_dims(
      tf.where(tf.equal(features["task"], 1))[:, 1], 1))
  decode_mask = tf.cumsum(tf.to_int32(
      tf.logical_and(
          tf.equal(features["verb_refs"][:, :, 0], eos_positions),
          tf.equal(features["verb_refs"][:, :, 1], eos_positions + 1))),
                          axis=-1)
  decode_mask = tf.sequence_mask(
      tf.reduce_sum(tf.to_int32(tf.less(decode_mask, 1)), -1),
      maxlen=tf.shape(decode_mask)[1])
  return decode_mask


def _decode_common(hparams):
  """Common graph for decoding."""
  features = get_input(hparams, FLAGS.data_files)
  decode_features = {}
  for key in features:
    if key.endswith("_refs"):
      continue
    decode_features[key] = features[key]
  _, _, _, references = seq2act_model.compute_logits(
      features, hparams, mode=tf.estimator.ModeKeys.EVAL)
  decode_utils.decode_n_step(seq2act_model.compute_logits,
                             decode_features, references["areas"],
                             hparams, n=20,
                             beam_size=FLAGS.beam_size)
  decode_mask = generate_action_mask(decode_features)
  return decode_features, decode_mask, features


def to_string(name, seq):
  steps = []
  for step in seq:
    steps.append(",".join(map(str, step)))
  return name + " - ".join(steps)


def ref_acc_to_string_list(task_seqs, ref_seqs, masks):
  """Convert a seqs of refs to strings."""
  cra = 0.
  pra = 0.
  string_list = []
  for task, seq, mask in zip(task_seqs, ref_seqs, masks):
    # Assuming batch_size = 1
    string_list.append(task)
    string_list.append(to_string("gt_seq", seq["gt_seq"][0]))
    string_list.append(to_string("pred_seq", seq["pred_seq"][0][mask[0]]))
    string_list.append(
        "complete_seq_acc: " + str(
            seq["complete_seq_acc"]) + " partial_seq_acc: " + str(
                seq["partial_seq_acc"]))
    cra += seq["complete_seq_acc"]
    pra += seq["partial_seq_acc"]
  mcra = cra / len(ref_seqs)
  mpra = pra / len(ref_seqs)
  string_list.append("mean_complete_seq_acc: " + str(mcra) +(
      "mean_partial_seq_acc: " + str(mpra)))
  return string_list


def save(task_seqs, seqs, masks, tag):
  string_list = ref_acc_to_string_list(task_seqs, seqs, masks)
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  with tf.gfile.GFile(os.path.join(FLAGS.output_dir, "decodes." + tag),
                      mode="w") as f:
    for item in string_list:
      print(item)
      f.write(str(item))
      f.write("\n")


def decode_fn(hparams):
  """The main function."""
  decode_dict, decode_mask, label_dict = _decode_common(hparams)
  if FLAGS.problem != "android_howto":
    decode_dict["input_refs"] = decode_utils.unify_input_ref(
        decode_dict["verbs"], decode_dict["input_refs"])
  print_ops = []
  for key in ["raw_task", "verbs", "objects",
              "verb_refs", "obj_refs", "input_refs"]:
    print_ops.append(tf.print(key, tf.shape(decode_dict[key]), decode_dict[key],
                              label_dict[key], "decode_mask", decode_mask,
                              summarize=100))
  acc_metrics = decode_utils.compute_seq_metrics(
      label_dict, decode_dict, mask=None)
  saver = tf.train.Saver()
  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    tf.logging.info("Restoring from the latest checkpoint: %s" %
                    (latest_checkpoint))
    saver.restore(session, latest_checkpoint)
    task_seqs = []
    ref_seqs = []
    act_seqs = []
    mask_seqs = []
    try:
      i = 0
      while True:
        tf.logging.info("Example %d" % i)
        task, acc, mask, label, decode = session.run([
            decode_dict["raw_task"], acc_metrics, decode_mask,
            label_dict, decode_dict
        ])
        ref_seq = {}
        ref_seq["gt_seq"] = np.concatenate([
            label["verb_refs"], label["obj_refs"], label["input_refs"]],
                                           axis=-1)
        ref_seq["pred_seq"] = np.concatenate([
            decode["verb_refs"], decode["obj_refs"], decode["input_refs"]],
                                             axis=-1)
        ref_seq["complete_seq_acc"] = acc["complete_refs_acc"]
        ref_seq["partial_seq_acc"] = acc["partial_refs_acc"]
        act_seq = {}
        act_seq["gt_seq"] = np.concatenate([
            np.expand_dims(label["verbs"], 2),
            np.expand_dims(label["objects"], 2),
            label["input_refs"]], axis=-1)
        act_seq["pred_seq"] = np.concatenate([
            np.expand_dims(decode["verbs"], 2),
            np.expand_dims(decode["objects"], 2),
            decode["input_refs"]], axis=-1)
        act_seq["complete_seq_acc"] = acc["complete_acts_acc"]
        act_seq["partial_seq_acc"] = acc["partial_acts_acc"]
        print("task", task)
        print("ref_seq", ref_seq)
        print("act_seq", act_seq)
        print("mask", mask)
        task_seqs.append(task)
        ref_seqs.append(ref_seq)
        act_seqs.append(act_seq)
        mask_seqs.append(mask)
        i += 1
    except tf.errors.OutOfRangeError:
      pass
    save(task_seqs, ref_seqs, mask_seqs, "joint_refs")
    save(task_seqs, act_seqs, mask_seqs, "joint_act")


def main(_):
  hparams = seq2act_estimator.load_hparams(FLAGS.checkpoint_path)
  hparams.set_hparam("batch_size", FLAGS.decode_batch_size)
  decode_fn(hparams)

if __name__ == "__main__":
  tf.app.run()

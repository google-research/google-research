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

"""Evaluate predictions JSON file, w.r.t. ground truth file."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import json
import os
import numpy as np
import tensorflow as tf

from schema_guided_dst import metrics

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "prediction_dir", None,
    "Directory in which all JSON files combined are predictions of the"
    " evaluation set on a single model checkpoint. We evaluate these JSON files"
    " by DSTC8 metrics.")
flags.DEFINE_string(
    "dstc8_data_dir", None,
    "Directory for the downloaded DSTC8 data, which contains the dialogue files"
    " and schema files of all datasets (train, dev, test)")
flags.DEFINE_enum("eval_set", None, ["train", "dev", "test"],
                  "Dataset split for evaluation.")
flags.DEFINE_string(
    "output_metric_file", None,
    "Single JSON output file containing aggregated evaluation metrics results"
    " for all predictions files in FLAGS.prediction_dir.")
flags.DEFINE_boolean(
    "joint_acc_across_turn", False,
    "Whether to compute joint accuracy across turn instead of across service. "
    "Should be set to True when conducting multiwoz style evaluation.")
flags.DEFINE_boolean(
    "use_fuzzy_match", True,
    "Whether to use fuzzy string matching when comparing non-categorical slot "
    "values. Should be set to False when conducting multiwoz style evaluation.")

ALL_SERVICES = "#ALL_SERVICES"
SEEN_SERVICES = "#SEEN_SERVICES"
UNSEEN_SERVICES = "#UNSEEN_SERVICES"

# Name of the file containing all predictions and their corresponding frame
# metrics.
PER_FRAME_OUTPUT_FILENAME = "dialogues_and_metrics.json"


def get_service_set(schema_path):
  """Get the set of all services present in a schema."""
  service_set = set()
  with tf.gfile.GFile(schema_path) as f:
    schema = json.load(f)
    for service in schema:
      service_set.add(service["service_name"])
  return service_set


def get_in_domain_services(schema_path_1, schema_path_2):
  """Get the set of common services between two schemas."""
  return get_service_set(schema_path_1) & get_service_set(schema_path_2)


def get_dataset_as_dict(file_path_patterns):
  """Read the DSTC8 json dialog data as dictionary with dialog ID as keys."""
  dataset_dict = {}
  if isinstance(file_path_patterns, list):
    list_fp = file_path_patterns
  else:
    list_fp = sorted(tf.gfile.Glob(file_path_patterns))
  for fp in list_fp:
    if PER_FRAME_OUTPUT_FILENAME in fp:
      continue
    tf.logging.info("Loading file: %s", fp)
    with tf.gfile.GFile(fp) as f:
      data = json.load(f)
      if isinstance(data, list):
        for dial in data:
          dataset_dict[dial["dialogue_id"]] = dial
      elif isinstance(data, dict):
        dataset_dict.update(data)
  return dataset_dict


def get_metrics(dataset_ref, dataset_hyp, service_schemas, in_domain_services):
  """Calculate the DSTC8 metrics.

  Args:
    dataset_ref: The ground truth dataset represented as a dict mapping dialogue
      id to the corresponding dialogue.
    dataset_hyp: The predictions in the same format as `dataset_ref`.
    service_schemas: A dict mapping service name to the schema for the service.
    in_domain_services: The set of services which are present in the training
      set.

  Returns:
    A dict mapping a metric collection name to a dict containing the values
    for various metrics. Each metric collection aggregates the metrics across
    a specific set of frames in the dialogues.
  """
  # Metrics can be aggregated in various ways, eg over all dialogues, only for
  # dialogues containing unseen services or for dialogues corresponding to a
  # single service. This aggregation is done through metric_collections, which
  # is a dict mapping a collection name to a dict, which maps a metric to a list
  # of values for that metric. Each value in this list is the value taken by
  # the metric on a frame.
  metric_collections = collections.defaultdict(
      lambda: collections.defaultdict(list))

  # Ensure the dialogs in dataset_hyp also occur in dataset_ref.
  assert set(dataset_hyp.keys()).issubset(set(dataset_ref.keys()))
  tf.logging.info("len(dataset_hyp)=%d, len(dataset_ref)=%d", len(dataset_hyp),
                  len(dataset_ref))

  # Store metrics for every frame for debugging.
  per_frame_metric = {}
  for dial_id, dial_hyp in dataset_hyp.items():
    dial_ref = dataset_ref[dial_id]

    if set(dial_ref["services"]) != set(dial_hyp["services"]):
      raise ValueError(
          "Set of services present in ground truth and predictions don't match "
          "for dialogue with id {}".format(dial_id))
    joint_metrics = [
        metrics.JOINT_GOAL_ACCURACY, metrics.JOINT_CAT_ACCURACY,
        metrics.JOINT_NONCAT_ACCURACY
    ]
    for turn_id, (turn_ref, turn_hyp) in enumerate(
        zip(dial_ref["turns"], dial_hyp["turns"])):
      metric_collections_per_turn = collections.defaultdict(
          lambda: collections.defaultdict(lambda: 1.0))
      if turn_ref["speaker"] != turn_hyp["speaker"]:
        raise ValueError(
            "Speakers don't match in dialogue with id {}".format(dial_id))

      # Skip system turns because metrics are only computed for user turns.
      if turn_ref["speaker"] != "USER":
        continue

      if turn_ref["utterance"] != turn_hyp["utterance"]:
        tf.logging.info("Ref utt: %s", turn_ref["utterance"])
        tf.logging.info("Hyp utt: %s", turn_hyp["utterance"])
        raise ValueError(
            "Utterances don't match for dialogue with id {}".format(dial_id))

      hyp_frames_by_service = {
          frame["service"]: frame for frame in turn_hyp["frames"]
      }

      # Calculate metrics for each frame in each user turn.
      for frame_ref in turn_ref["frames"]:
        service_name = frame_ref["service"]
        if service_name not in hyp_frames_by_service:
          raise ValueError(
              "Frame for service {} not found in dialogue with id {}".format(
                  service_name, dial_id))
        service = service_schemas[service_name]
        frame_hyp = hyp_frames_by_service[service_name]

        active_intent_acc = metrics.get_active_intent_accuracy(
            frame_ref, frame_hyp)
        slot_tagging_f1_scores = metrics.get_slot_tagging_f1(
            frame_ref, frame_hyp, turn_ref["utterance"], service)
        requested_slots_f1_scores = metrics.get_requested_slots_f1(
            frame_ref, frame_hyp)
        goal_accuracy_dict = metrics.get_average_and_joint_goal_accuracy(
            frame_ref, frame_hyp, service, FLAGS.use_fuzzy_match)

        frame_metric = {
            metrics.ACTIVE_INTENT_ACCURACY:
                active_intent_acc,
            metrics.REQUESTED_SLOTS_F1:
                requested_slots_f1_scores.f1,
            metrics.REQUESTED_SLOTS_PRECISION:
                requested_slots_f1_scores.precision,
            metrics.REQUESTED_SLOTS_RECALL:
                requested_slots_f1_scores.recall
        }
        if slot_tagging_f1_scores is not None:
          frame_metric[metrics.SLOT_TAGGING_F1] = slot_tagging_f1_scores.f1
          frame_metric[metrics.SLOT_TAGGING_PRECISION] = (
              slot_tagging_f1_scores.precision)
          frame_metric[
              metrics.SLOT_TAGGING_RECALL] = slot_tagging_f1_scores.recall
        frame_metric.update(goal_accuracy_dict)

        frame_id = "{:s}-{:03d}-{:s}".format(dial_id, turn_id,
                                             frame_hyp["service"])
        per_frame_metric[frame_id] = frame_metric
        # Add the frame-level metric result back to dialogues.
        frame_hyp["metrics"] = frame_metric

        # Get the domain name of the service.
        domain_name = frame_hyp["service"].split("_")[0]
        domain_keys = [ALL_SERVICES, frame_hyp["service"], domain_name]
        if frame_hyp["service"] in in_domain_services:
          domain_keys.append(SEEN_SERVICES)
        else:
          domain_keys.append(UNSEEN_SERVICES)
        for domain_key in domain_keys:
          for metric_key, metric_value in frame_metric.items():
            if metric_value != metrics.NAN_VAL:
              if FLAGS.joint_acc_across_turn and metric_key in joint_metrics:
                metric_collections_per_turn[domain_key][
                    metric_key] *= metric_value
              else:
                metric_collections[domain_key][metric_key].append(metric_value)
      if FLAGS.joint_acc_across_turn:
        # Conduct multiwoz style evaluation that computes joint goal accuracy
        # across all the slot values of all the domains for each turn.
        for domain_key in metric_collections_per_turn:
          for metric_key, metric_value in metric_collections_per_turn[
              domain_key].items():
            metric_collections[domain_key][metric_key].append(metric_value)
  all_metric_aggregate = {}
  for domain_key, domain_metric_vals in metric_collections.items():
    domain_metric_aggregate = {}
    for metric_key, value_list in domain_metric_vals.items():
      if value_list:
        # Metrics are macro-averaged across all frames.
        domain_metric_aggregate[metric_key] = float(np.mean(value_list))
      else:
        domain_metric_aggregate[metric_key] = metrics.NAN_VAL
    all_metric_aggregate[domain_key] = domain_metric_aggregate
  return all_metric_aggregate, per_frame_metric


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  in_domain_services = get_in_domain_services(
      os.path.join(FLAGS.dstc8_data_dir, FLAGS.eval_set, "schema.json"),
      os.path.join(FLAGS.dstc8_data_dir, "train", "schema.json"))
  with tf.io.gfile.GFile(
      os.path.join(FLAGS.dstc8_data_dir, FLAGS.eval_set, "schema.json")) as f:
    eval_services = {}
    list_services = json.load(f)
    for service in list_services:
      eval_services[service["service_name"]] = service

  dataset_ref = get_dataset_as_dict(
      os.path.join(FLAGS.dstc8_data_dir, FLAGS.eval_set, "dialogues_*.json"))
  dataset_hyp = get_dataset_as_dict(
      os.path.join(FLAGS.prediction_dir, "*.json"))

  all_metric_aggregate, _ = get_metrics(dataset_ref, dataset_hyp, eval_services,
                                        in_domain_services)
  tf.logging.info("Dialog metrics: %s", str(all_metric_aggregate[ALL_SERVICES]))

  # Write the aggregated metrics values.
  with tf.gfile.GFile(FLAGS.output_metric_file, "w") as f:
    json.dump(
        all_metric_aggregate,
        f,
        indent=2,
        separators=(",", ": "),
        sort_keys=True)
  # Write the per-frame metrics values with the corrresponding dialogue frames.
  with tf.gfile.GFile(
      os.path.join(FLAGS.prediction_dir, PER_FRAME_OUTPUT_FILENAME), "w") as f:
    json.dump(dataset_hyp, f, indent=2, separators=(",", ": "))


if __name__ == "__main__":
  flags.mark_flag_as_required("prediction_dir")
  flags.mark_flag_as_required("dstc8_data_dir")
  flags.mark_flag_as_required("eval_set")
  flags.mark_flag_as_required("output_metric_file")
  tf.compat.v1.app.run(main)

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

"""Evaluates DrFact results on OpenCSR Dataset."""

import collections
import json

from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


def scoring_choice(choice2found_concepts, method="top"):
  """Computes the score of each choice and return the best one."""
  scores = {}
  for choice, found_concepts in choice2found_concepts.items():
    if method == "top":
      scores[choice] = float(found_concepts[0][1])  # 0 is the top tuple
    elif method == "avg":
      scores[choice] = float(np.mean([fc[1] for fc in found_concepts]))
  return scores


def opencsr_eval_fn(dataset, results, name_map, output_prediction_file,
                    paragraphs, **kwargs):
  """Computes evaluation metrics for OpenCSRDataset.

  Args:
    dataset: An object of type OpenCSRDataset.
    results: A list of result dicts from running estimator.predict.
    name_map: A mapping from prediction indices to text strings.
    output_prediction_file: File to store predictions to.
    paragraphs: All facts in a dict.
    **kwargs: Variable keyword arguments.

  Returns:
    metrics: A dict mapping metric names to values.
  """
  del kwargs

  #   # Collect ground truth answers.
  gt_correct = {ex.qas_id: ex.correct_choice for ex in dataset.examples}
  gt_choices = {ex.qas_id: ex.choice2concepts for ex in dataset.examples}
  gt_ques = {ex.qas_id: ex.question_text for ex in dataset.examples}

  keep_num = 30
  num_correct_top = 0
  num_correct_avg = 0

  layer_weights = np.zeros_like(results[0]["layer_probs"])
  num_ent_layers = layer_weights.shape[0]
  tf.logging.info("layer_weights.shape: %s", str(layer_weights.shape))
  all_predictions = {}
  for rid, result in enumerate(results):
    qas_id = result["qas_ids"].decode("utf-8")
    preds = result["top_idx"]
    scores = result["top_vals"]
    all_predictions[qas_id] = collections.OrderedDict()
    correct_choice = gt_correct[qas_id]
    choice2concepts = gt_choices[qas_id]  # choice2concepts dict
    pred_list = []
    concept2choice = {}  # backtrack
    for choice, concepts in choice2concepts.items():
      for concept in concepts:
        assert concept not in concept2choice
        concept2choice[concept] = choice
    choice2found_concepts = collections.defaultdict(list)
    found_concepts = set()
    if rid % 100 == 0:
      tf.logging.info("Processed %d results in hotpot_eval_fn", rid)
      # tf.logging.info("found_choices: %d ", len(found_choices))
      # tf.logging.info("len(preds): %d ", len(preds))
    for i, pred in enumerate(preds):
      pred_concept = name_map[str(pred)]
      if pred_concept in concept2choice:
        found_concepts.add(pred_concept)
        pred_choi = concept2choice[pred_concept]
        choice2found_concepts[pred_choi].append(
            (pred_concept, float(scores[i])))
      if len(pred_list) <= keep_num:
        if float(scores[i]) > 0:
          pred_list.append((pred_concept, float(scores[i])))
      if len(found_concepts) == len(concept2choice):
        # Early stop when we found all choice-concepts.
        break
    choice2score_top = scoring_choice(choice2found_concepts, method="top")
    choice2score_avg = scoring_choice(choice2found_concepts, method="avg")
    choice2score_top_sorted = sorted(
        [(k, float(v)) for k, v in choice2score_top.items()],
        key=lambda x: x[1],
        reverse=True)
    choice2score_avg_sorted = sorted(
        [(k, float(v)) for k, v in choice2score_avg.items()],
        key=lambda x: x[1],
        reverse=True)
    if choice2score_top_sorted[0][0] == correct_choice:
      num_correct_top += 1
    if choice2score_avg_sorted[0][0] == correct_choice:
      num_correct_avg += 1
    all_predictions[qas_id]["question"] = gt_ques[qas_id]
    all_predictions[qas_id]["correct_choice"] = correct_choice
    all_predictions[qas_id]["choice2found_concepts"] = choice2found_concepts
    all_predictions[qas_id]["choice2score"] = {
        "top": choice2score_top,
        "avg": choice2score_avg
    }
    if FLAGS.model_type == "drfact":
      # Qry Ents
      qry_ents = [int(v) for v in result["qry_ents"] if v >= 0]
      qry_ents_scores = [float(v) for v in result["qry_ent_scores"] if v >= 0]
      all_predictions[qas_id]["qry_ents"] = [
          (eid, name_map[str(eid)], score)
          for eid, score in zip(qry_ents, qry_ents_scores)
      ]
      # Facts
      for hop_id in range(4):
        hop_key = "layer_%d_fact_ids" % hop_id
        if hop_key not in result:
          continue
        cur_facts = [int(v) for v in result[hop_key]]
        cur_fact_scores = [
            float(v) for v in result[hop_key.replace("ids", "scs")]
        ]
        all_predictions[qas_id][hop_key] = [
            (fid, " ".join(paragraphs[fid]).replace(" ##", ""), score)
            for fid, score in zip(cur_facts, cur_fact_scores)
            if score > 0
        ]
    # Non-accuracy stats
    layer_weights += result["layer_probs"]
    layer_entities = {i: [] for i in range(num_ent_layers)}
    all_predictions[qas_id]["layer_pred"] = collections.OrderedDict()
    for i in range(num_ent_layers):
      layer_entities[i] = result["layer_%d_ent" % i][:keep_num]
      all_predictions[qas_id]["layer_pred"]["layer_%d" % i] = [
          name_map[str(ee)] for ee in layer_entities[i] if ee > 0
      ]
    all_predictions[qas_id]["top_%d_predictions" % keep_num] = pred_list

  metric = dict()
  metric["num_correct_top"] = num_correct_top
  metric["num_correct_avg"] = num_correct_avg
  metric["num_examples"] = len(results)
  metric["accuracy_top"] = num_correct_top / len(results)
  metric["accuracy_avg"] = num_correct_avg / len(results)
  metric["accuracy"] = metric["accuracy_top"]
  # Non-accuracy analysis
  for i in range(num_ent_layers):  # hop_id
    metric["analysis/layer_weight_%d" %
           i] = layer_weights[i] / len(all_predictions)
    # metric["analysis/num_entities_%d" %
    #        i] = num_layer_entities[i] / len(all_predictions)
    # metric["analysis/num_new_entities_%d" %
    #        i] = num_new_entities[i] / len(all_predictions)

  results = dict(all_predictions=all_predictions, metric=metric)
  with tf.gfile.Open(output_prediction_file, "w") as gfo:
    gfo.write(json.dumps(metric) + "\n")
    gfo.write("\n".join([
        json.dumps(dict(qas_id=k, predictions=v))
        for k, v in all_predictions.items()
    ]))
    gfo.write("\n")
  return metric

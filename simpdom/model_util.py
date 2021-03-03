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

"""The data parsing and evaluation functions for the models."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools
import hashlib
import itertools
import json
import os
import random
import sys

from absl import flags
import tensorflow.compat.v1 as tf
from simpdom import constants
from simpdom import process_domtree_data

FLAGS = flags.FLAGS
tf.set_random_seed(42)


def page_level_constraint(result_path, vertical, source_website, target_website,
                          domtree_data_path):
  """Takes the top highest prediction for empty field by ranking raw scores."""
  with tf.gfile.Open(
      os.path.join(
          result_path,
          "{}/{}-results/score/{}.preds.txt".format(vertical, source_website,
                                                    target_website)), "r") as f:
    lines = [line.strip() for line in f.readlines()]
  with tf.gfile.Open(
      os.path.join(domtree_data_path, "{}.vocab.tags.txt".format(vertical)),
      "r") as f:
    tags = [line.strip() for line in f.readlines()]
  site_field_truth_exist = dict()
  page_field_max = dict()
  page_field_pred_count = dict()
  for line in lines:
    items = line.split("\t")
    assert len(items) >= 5, items
    html_path = items[0]
    truth = items[3]
    pred = items[4]
    if pred != "none":
      if pred not in page_field_pred_count:
        page_field_pred_count[pred] = 0
      page_field_pred_count[pred] += 1
      continue
    raw_scores = [float(x) for x in items[5].split(",")]
    assert len(raw_scores) == len(tags)
    site_field_truth_exist[truth] = True
    for index, score in enumerate(raw_scores):
      if html_path not in page_field_max:
        page_field_max[html_path] = {}
      if tags[index] not in page_field_max[
          html_path] or score >= page_field_max[html_path][tags[index]]:
        page_field_max[html_path][tags[index]] = score
  print(page_field_pred_count, file=sys.stderr)
  voted_lines = []
  for line in lines:
    items = line.split("\t")
    assert len(items) >= 5, items
    html_path = items[0]
    raw_scores = [float(x) for x in items[5].split(",")]
    pred = items[4]
    for index, tag in enumerate(tags):
      if tag in site_field_truth_exist and tag not in page_field_pred_count:
        if pred != "none":
          continue
        if raw_scores[index] >= page_field_max[html_path][tags[index]] - (1e-3):
          items[4] = tag
    voted_lines.append("\t".join(items))
  with tf.gfile.Open(
      os.path.join(
          result_path, "{}/{}-results/score/{}.preds.constrained.txt".format(
              vertical, source_website, target_website)), "w") as f:
    f.write("\n".join(voted_lines))
  return site_level_voting(
      result_path, vertical, source_website, target_website, constrained=True)


def site_level_voting(result_path,
                      vertical,
                      source_website,
                      target_website,
                      constrained=False):
  """Adds the majority voting for the predictions."""

  input_path = os.path.join(
      result_path,
      "{}/{}-results/score/{}.preds.txt".format(vertical, source_website,
                                                target_website))
  output_path = os.path.join(
      result_path,
      "{}/{}-results/score/{}.preds.voted.txt".format(vertical, source_website,
                                                      target_website))
  if constrained:
    input_path = input_path.replace("preds.txt", "preds.constrained.txt")
    output_path = output_path.replace("preds.voted.txt",
                                      "preds.constrained.voted.txt")
  with tf.gfile.Open(input_path, "r") as f:
    lines = [line.strip() for line in f.readlines()]

  field_xpath_freq_dict = dict()

  for line in lines:
    items = line.split("\t")
    assert len(items) >= 5, items
    xpath = items[1]
    pred = items[4]
    if pred == "none":
      continue
    if pred not in field_xpath_freq_dict:
      field_xpath_freq_dict[pred] = dict()
    if xpath not in field_xpath_freq_dict[pred]:
      field_xpath_freq_dict[pred][xpath] = 0
    field_xpath_freq_dict[pred][xpath] += 1

  most_frequent_xpaths = dict()  #  Site level voting.
  for field, xpth_freq in field_xpath_freq_dict.items():
    frequent_xpath = sorted(
        xpth_freq.items(), key=lambda kv: kv[1], reverse=True)[0][0]  # Top 1.
    most_frequent_xpaths[field] = frequent_xpath

  voted_lines = []
  for line in lines:
    items = line.split("\t")
    assert len(items) >= 5, items
    xpath = items[1]
    flag = "none"
    for field, most_freq_xpath in most_frequent_xpaths.items():
      if xpath == most_freq_xpath:
        flag = field
    if items[4] == "none" and flag != "none":
      items[4] = flag
    voted_lines.append("\t".join(items))
  with tf.gfile.Open(output_path, "w") as f:
    f.write("\n".join(voted_lines))
  return page_hits_level_metric(
      result_path,
      vertical,
      source_website,
      target_website,
      voted=True,
      constrained=constrained)


def page_hits_level_metric(result_path,
                           vertical,
                           source_website,
                           target_website,
                           voted=False,
                           constrained=False):
  """Evaluates the hit level prediction result with precision/recall/f1."""

  file_suffix = ".constrained" if constrained else ""
  file_suffix += ".voted" if voted else ""

  input_path = os.path.join(result_path,
                            ("{}/{}-results/score/{}.preds" + file_suffix +
                             ".txt").format(vertical, source_website,
                                            target_website))
  output_path = os.path.join(result_path, ("{}/{}-results/score/{}.metric.hit" +
                                           file_suffix + ".txt").format(
                                               vertical, source_website,
                                               target_website))

  with tf.gfile.Open(input_path, "r") as f:
    lines = [line.strip() for line in f.readlines()]

  evaluation_dict = dict()

  for line in lines:
    items = line.split("\t")
    assert len(items) >= 5, items
    html_path = items[0]
    text = items[2]
    truth = items[3]
    pred = items[4]
    if truth not in evaluation_dict and truth != "none":
      evaluation_dict[truth] = dict()
    if pred not in evaluation_dict and pred != "none":
      evaluation_dict[pred] = dict()
    if truth != "none":
      if html_path not in evaluation_dict[truth]:
        evaluation_dict[truth][html_path] = {"truth": set(), "pred": set()}
      evaluation_dict[truth][html_path]["truth"].add(text)
    if pred != "none":
      if html_path not in evaluation_dict[pred]:
        evaluation_dict[pred][html_path] = {"truth": set(), "pred": set()}
      evaluation_dict[pred][html_path]["pred"].add(text)
  metric_str = "tag, num_truth, num_pred, precision, recall, f1\n"
  for tag in evaluation_dict:
    num_html_pages_with_truth = 0
    num_html_pages_with_pred = 0
    num_html_pages_with_correct = 0
    for html_path in evaluation_dict[tag]:
      result = evaluation_dict[tag][html_path]
      if result["truth"]:
        num_html_pages_with_truth += 1
      if result["pred"]:
        num_html_pages_with_pred += 1
      if result["truth"] & result["pred"]:
        num_html_pages_with_correct += 1

    precision = num_html_pages_with_correct / (
        num_html_pages_with_pred + 0.000001)
    recall = num_html_pages_with_correct / (
        num_html_pages_with_truth + 0.000001)
    f1 = 2 * (precision * recall) / (precision + recall + 0.000001)
    metric_str += "%s, %d, %d, %.2f, %.2f, %.2f\n" % (
        tag, num_html_pages_with_truth, num_html_pages_with_pred, precision,
        recall, f1)
  with tf.gfile.Open(output_path, "w") as f:
    f.write(metric_str)
    print(f.name, file=sys.stderr)
  print(metric_str, file=sys.stderr)
  return metric_str


# Below are the functions for generating data for joint-extraction models.


def joint_generator_fn(json_file, goldmine_file, vertical, mode="all"):
  """Generates an iteratable hook for the input json file."""
  random_seed = int(hashlib.sha256(json_file.encode("utf-8")).hexdigest(),
                    16) % 10**4 + 42
  random.seed(random_seed)

  with tf.gfile.Open(json_file,
                     "r") as f_node_data, tf.gfile.Open(goldmine_file,
                                                        "r") as f_goldmine_data:
    node_data = json.load(f_node_data)
    goldmine_data = json.load(f_goldmine_data)
    print("Reading file:", json_file, file=sys.stderr)
    print("Reading file:", goldmine_file, file=sys.stderr)
    start = 0
    end = None
    len_data = len(node_data["features"])
    if mode == "train":
      end = int(len_data * 1.0)
    elif mode == "test":
      start = int(len_data * 0.9)
    elif mode == "all":
      pass
    html_path = ""
    print("randint list of %s with random seed %d: " % (json_file, random_seed))
    # Assert page numbers are equal.
    for page, goldmine_features in zip(node_data["features"][start:end],
                                       goldmine_data[start:end]):
      # Assert node numbers are equal.
      node_list = []
      html_path = page[0]["html_path"]
      for node_index, node in enumerate(page):
        if node["text"]:
          if node["label"] == "none" and mode != "all":
            # Make the data to be balanced.
            rand_int = random.randint(0, 100000)
            if rand_int >= FLAGS.none_cutoff:
              continue
          node["goldmine_feats"] = list(
              set(goldmine_features[node_index].get("goldmine", {})))
          node_list.append(node)
      parsed_data = joint_parse_fn(node_list, html_path, vertical)
      if parsed_data:
        yield parsed_data


def split_xpath(xpath):
  """Gets the leaf type from the xpath."""
  split_tags = []
  for tag in xpath.split("/"):
    if tag.find("[") >= 0:
      tag = tag[:tag.find("[")]
    if tag.strip():
      split_tags.append(tag.strip())
  return split_tags


def parse_xpath(xpath_list):
  """Parses the xpath for LSTM encoding."""
  node_xpath_list = []
  for xpath in xpath_list:
    node_xpath_unit_seq = split_xpath(xpath)
    node_xpath_list.append([u.encode() for u in node_xpath_unit_seq])
  node_xpath_len_list = [len(xpath) for xpath in node_xpath_list]
  max_xpath_len = max(node_xpath_len_list, default=0)
  if not max_xpath_len:
    return None
  node_xpath_list = [
      xpath + [b"<pad>"] * (max_xpath_len - len(xpath))
      for xpath in node_xpath_list
  ]
  return node_xpath_list, node_xpath_len_list


def joint_parse_fn(node_list, html_path, vertical):
  """Encodes the input data in a padded format for TensorFlow."""
  words_list = []
  leaf_type_list = []
  goldmine_feature_list = []
  prev_text_words_list = []
  partner_words_list = []
  friends_words_list = []
  friends_var_list = []
  friends_fix_list = []
  friend_has_label = []
  tag_list = []
  html_path = html_path.encode()
  xpath_list = []
  xpath_list_not_encode = []
  chars_list = []
  chars_len_list = []
  max_chars_number = 0
  max_friends_fix = 0
  max_friends_var = 0
  max_gm_feat = 8
  padded_attributes = [l.encode() for l in constants.ATTRIBUTES_PAD[vertical]]
  attributes_plus_none = [
      l.encode() for l in constants.ATTRIBUTES_PLUS_NONE[vertical]
  ]

  for node in node_list:
    words = [w.encode() for w in node["text"][:FLAGS.max_len_text]]
    words_list.append(words)
    gm_feats = node["goldmine_feats"] + ["<pad>"] * (
        max_gm_feat - len(node["goldmine_feats"]))
    assert len(gm_feats) == max_gm_feat
    goldmine_feature_list.append([gf.encode() for gf in gm_feats])
    prev_text_words = [
        w.encode()
        for w in list(itertools.chain.from_iterable(node["prev_text"]))
        [-FLAGS.max_len_prev_text:]
    ]
    prev_text_words_list.append(prev_text_words)

    if FLAGS.circle_features:
      partner_words = [
          w.encode()
          for w in list(itertools.chain.from_iterable(node["partner_text"]))
      ]
      partner_words_list.append(partner_words)

      node["friends_text"] = node["friends_text"][:FLAGS.max_len_friend_nodes]
      friends_words = [
          w.encode()
          for w in list(itertools.chain.from_iterable(node["friends_text"]))
      ]
      friends_var_words = [
          w.encode() for w in list(
              itertools.chain.from_iterable(node["friends_var_text"]))
      ]
      friend_all_words = friends_words + friends_var_words
      friend_all_words = [w.lower() for w in friend_all_words]
      has_label = any([
          w.encode() in friend_all_words for w in constants.ATTRIBUTES[vertical]
      ])
      friend_has_label.append(1.0 if has_label else 0.0)

      friends_words_list.append(friend_all_words[:FLAGS.max_len_prev_text])
      max_friends_var = max(len(node["friends_var_text"]), max_friends_var, 1)
      max_friends_fix = max(len(node["friends_text"]), max_friends_fix, 1)
      friends_var_list.append([[w.encode()
                                for w in f][:FLAGS.max_len_text]
                               for f in node["friends_var_text"]])
      friends_fix_list.append([[w.encode()
                                for w in f][:FLAGS.max_len_text]
                               for f in node["friends_text"]])

    tag_list.append(node["label"].encode())
    xpath_list.append(node["xpath"].encode())
    xpath_list_not_encode.append(node["xpath"])
    leaf_type_list.append(
        process_domtree_data.get_leaf_type(node["xpath"]).encode())
    # Chars
    chars = [[c.encode() for c in w] for w in node["text"][:FLAGS.max_len_text]]
    chars_lens = [len(wc) for wc in chars]
    chars_list.append(chars)
    chars_len_list.append(chars_lens)
    max_chars_number = max(max_chars_number, max(chars_lens))
  max_words_number = max([len(words) for words in words_list], default=0)
  # Padding the char list and char len list.
  padded_chars_list = []
  padded_chars_len_list = []
  for node_index in range(len(chars_list)):
    padded_word_len_list = []
    padded_word_chars = []
    for word_index in range(max_words_number):
      if word_index < len(chars_list[node_index]):
        padded_chars = chars_list[node_index][word_index] + [b"<pad>"] * (
            max_chars_number - len(chars_list[node_index][word_index]))
        padded_word_chars.append(padded_chars)
        padded_word_len_list.append(len(chars_list[node_index][word_index]))
      else:
        padded_word_chars.append([b"<pad>"] * max_chars_number)
        padded_word_len_list.append(0)
      assert len(padded_chars) == max_chars_number
    padded_chars_len_list.append(padded_word_len_list)
    assert len(padded_word_chars) == max_words_number
    padded_chars_list.append(padded_word_chars)
  assert len(padded_chars_list) == len(node_list)

  words_len_list = [len(words) for words in words_list]
  max_words_len = max(words_len_list, default=0)
  prev_text_words_len_list = [len(words) for words in prev_text_words_list]
  max_prev_words_len = max(prev_text_words_len_list, default=0)
  if max_words_len == 0 or max_prev_words_len == 0:
    return None
  words_list = [
      words + [b"<pad>"] * (max_words_len - len(words)) for words in words_list
  ]

  prev_text_words_list = [
      words + [b"<pad>"] * (max_prev_words_len - len(words))
      for words in prev_text_words_list
  ]

  # Parse the xpath for node classification.
  node_xpath_list, node_xpath_len_list = parse_xpath(xpath_list_not_encode)
  # Parse the relative position for node classification.
  max_pos = len(node_xpath_list)
  position_list = [int(float(x) / max_pos * 100) for x in list(range(max_pos))]
  position_list = [str(x).encode() for x in position_list]

  # Parse the friend circle features if required.
  if FLAGS.circle_features:
    friends_words_len_list = [len(words) for words in friends_words_list]
    partner_words_len_list = [len(words) for words in partner_words_list]
    max_friends_words_len = max(friends_words_len_list, default=0)
    max_partner_words_len = max(partner_words_len_list, default=0)
    # friends_words_list/partner_words_list: [num_node, num_word].
    friends_words_list = [
        words + [b"<pad>"] * (max_friends_words_len - len(words))
        for words in friends_words_list
    ]
    partner_words_list = [
        words + [b"<pad>"] * (max_partner_words_len - len(words))
        for words in partner_words_list
    ]

    # friends_var_list/friends_fix_list: [num_node, num_friend, num_word].
    max_friends_var_len = max(
        [len(x) for x in list(itertools.chain.from_iterable(friends_var_list))],
        default=1)
    max_friends_fix_len = max(
        [len(x) for x in list(itertools.chain.from_iterable(friends_fix_list))],
        default=1)
    for i, node in enumerate(friends_var_list):
      for j, friend in enumerate(node):
        friends_var_list[i][j] = friend + [b"<pad>"] * (
            max_friends_var_len - len(friend))
      friends_var_list[i] += [[b"<pad>"] * max_friends_var_len] * (
          max_friends_var - len(friends_var_list[i]))
    for i, node in enumerate(friends_fix_list):
      for j, friend in enumerate(node):
        friends_fix_list[i][j] = friend + [b"<pad>"] * (
            max_friends_fix_len - len(friend))
      friends_fix_list[i] += [[b"<pad>"] * max_friends_fix_len] * (
          max_friends_fix - len(friends_fix_list[i]))

    return ((len(node_list)), (friend_has_label), (words_list, words_len_list),
            (prev_text_words_list,
             prev_text_words_len_list), (padded_chars_list,
                                         padded_chars_len_list),
            (partner_words_list,
             partner_words_len_list), (friends_words_list,
                                       friends_words_len_list),
            (friends_fix_list, friends_var_list), (leaf_type_list,
                                                   goldmine_feature_list),
            (html_path, xpath_list), (node_xpath_list, node_xpath_len_list),
            (padded_attributes,
             attributes_plus_none), (position_list)), tag_list
    # padded_attributes, attributes_plus_none are both needed for friend circle
    # since padded_attributes is used for semantic similarity computation while
    # attributes_plus_none is used for binary scorer.
  else:
    return ((len(node_list)), (words_list, words_len_list),
            (prev_text_words_list, prev_text_words_len_list),
            (padded_chars_list, padded_chars_len_list), (leaf_type_list,
                                                         goldmine_feature_list),
            (html_path, xpath_list), (node_xpath_list, node_xpath_len_list),
            (padded_attributes), (position_list)), tag_list


def joint_input_fn(json_file,
                   goldmine_file,
                   vertical,
                   params=None,
                   shuffle_and_repeat=False,
                   mode="all"):
  """Produces the tf.dataset as the input to a Tensorflow model function."""
  print("joint_input_fn", file=sys.stderr)
  params = params if params is not None else {}
  shapes = (
      (
          (),  # len_node_list
          ([None, None], [None]),  # (words_list, nwords_list)
          ([None, None], [None]),
          # (prev_text_words_list, n_prev_text_words_list)
          ([None, None, None], [None, None]),  #  (chars_list, chars_len_list)
          ([None], [None, None]),  # (leaf_type, goldmine_feats_list)
          ((), [None]),  # (html_path, xpath_list)
          ([None, None], [None]),  # (node_xpath_list, node_xpath_len_list)
          ([len(constants.ATTRIBUTES_PAD[vertical])]),  # (padded_attributes)
          ([None])  # (Position_list)
      ),
      [None],  # tags
  )
  types = (((tf.int32), (tf.string, tf.int32), (tf.string, tf.int32),
            (tf.string, tf.int32), (tf.string, tf.string), (tf.string,
                                                            tf.string),
            (tf.string, tf.int32), (tf.string), (tf.string)), tf.string)
  defaults = (((0), ("<pad>", 0), ("<pad>", 0), ("<pad>", 0),
               ("<pad>", "<pad>"), ("pad_html_path", "pad_xpath"), ("<pad>", 0),
               ("<pad>"), ("<pad>"))), "none"

  if FLAGS.circle_features:
    shapes = (
        (
            (),  # len_node_list
            ([None]),  # (friend_has_label)
            ([None, None], [None]),  # (words_list, nwords_list)
            ([None, None], [None]),
            # (prev_text_words_list, n_prev_text_words_list)
            ([None, None, None], [None, None]),  #  (chars_list, chars_len_list)
            ([None, None], [None]),  # (partner_words, n_partner_words)
            ([None, None], [None]),  # (friends_words, n_friends_words)
            ([None, None, None], [None, None,
                                  None]),  # (friends_fix, friends_var)
            ([None], [None, None]),  # (leaf_type, goldmine_feats_list)
            ((), [None]),  # (html_path, xpath_list)
            ([None, None], [None]),  # (node_xpath_list, node_xpath_len_list)
            ([len(constants.ATTRIBUTES_PAD[vertical])],
             [len(constants.ATTRIBUTES_PLUS_NONE[vertical])
             ]),  # (padded_attributes, attributes_plus_none)
            ([None])  # (Position_list)
        ),
        [None],  # tags
    )
    types = (((tf.int32), (tf.float32), (tf.string, tf.int32),
              (tf.string, tf.int32), (tf.string, tf.int32),
              (tf.string, tf.int32), (tf.string, tf.int32),
              (tf.string, tf.string), (tf.string, tf.string),
              (tf.string, tf.string), (tf.string, tf.int32),
              (tf.string, tf.string), (tf.string)), tf.string)
    defaults = (((0), (0.0), ("<pad>", 0), ("<pad>", 0), ("<pad>", 0),
                 ("<pad>", 0), ("<pad>", 0), ("<pad>", "<pad>"),
                 ("<pad>", "<pad>"), ("pad_html_path", "pad_xpath"),
                 ("<pad>", 0), ("<pad>", "<pad>"), ("<pad>"))), "none"

  dataset = tf.data.Dataset.from_generator(
      functools.partial(joint_generator_fn, json_file, goldmine_file, vertical,
                        mode),
      output_shapes=shapes,
      output_types=types)

  if shuffle_and_repeat:
    dataset = dataset.shuffle(params["buffer"]).repeat(params["epochs"])

  dataset = (
      dataset.padded_batch(params.get("batch_size", 50), shapes,
                           defaults).prefetch(1))
  return dataset

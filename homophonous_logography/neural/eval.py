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

"""Miscellaneous evaluation APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import tensorflow.compat.v1 as tf  # tf

import homophonous_logography.neural.corpus as data_lib

CH_FONT = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def convolve2d(x, y):
  """2D convolution, using FFT.

  Defines the ratio of deviation from the ideal grapheme/phoneme mapping.
  Source:
    https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy

  Args:
    x: numpy array
    y: numpy array

  Returns:
    Numpy array containing the result of convolution.
  """
  fr = np.fft.fft2(x)
  fr2 = np.fft.fft2(np.flipud(np.fliplr(y)))
  m, n = fr.shape
  cc = np.real(np.fft.ifft2(fr * fr2))
  cc = np.roll(cc, -int(m/2) + 1, axis=0)
  cc = np.roll(cc, -int(n/2) + 1, axis=1)
  return cc


def compute_skew_from_ideal(attention_matrix, sentence, predicted, sigma=0.3):
  """Computes skew from ideal grapheme/phoneme mapping.

  Args:
    attention_matrix: observed attention matrix
    sentence: preprocessed input sentence
    predicted: preprocessed predicted spelling/pronunciation
    sigma: standard deviation for computing gaussian convolution mask

  Returns:
    A four-tuple consisting of:
      weight_mask: Weight mask.
      filtered: Attention matrix filtered by the weight mask.
      ratio: Ration of filtered to non-filtered (estimate of a neural measure).
      reduced_ratio: Same as above, but reduced but elements reduced prior
        to computation.
  """
  sentence = list(sentence)
  predicted = list(predicted)
  left_target = -1
  right_target = -1
  for i in range(len(sentence)):
    if sentence[i] == "<":
      left_target = i + 1
    elif sentence[i] == ">":
      right_target = i
  assert left_target != -1 and right_target != -1
  input_length = attention_matrix.shape[1]
  output_length = attention_matrix.shape[0]
  predicted_length = len(predicted)
  ideal_phonographic_matrix = np.zeros([output_length, input_length])
  j = 0
  for i in range(left_target, right_target):
    ideal_phonographic_matrix[j, i] = 1
    if j < predicted_length:
      j += 1
  while j < predicted_length:
    ideal_phonographic_matrix[j, i] = 1
    j += 1
  # https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-79.php
  x, y = np.meshgrid(np.linspace(-1, 1, input_length),
                     np.linspace(-1, 1, output_length))
  d = np.sqrt(x * x + y * y)
  mu = 0.0
  convolution_mask = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))
  weight_mask = 1 - convolve2d(ideal_phonographic_matrix, convolution_mask)
  mn, mx = np.amin(weight_mask), np.amax(weight_mask)
  weight_mask = (weight_mask - mn) * 1 / (mx - mn)
  filtered = attention_matrix * weight_mask
  ratio = np.sum(filtered) / np.sum(attention_matrix)
  reduced_ratio = (np.sum(tf.reduce_max(filtered, axis=0)) /
                   np.sum(tf.reduce_max(attention_matrix, axis=0)))
  return weight_mask, filtered, ratio, reduced_ratio


def compute_simple_skew(attention_matrix, sentence, predicted):
  """Computes simple skew from ideal grapheme/phoneme mapping.

  Args:
    attention_matrix: observed attention matrix
    sentence: preprocessed input sentence
    predicted: preprocessed predicted spelling/pronunciation

  Returns:
    A four-tuple consisting of:
      weight_mask: Weight mask.
      filtered: Attention matrix filtered by the weight mask.
      ratio: Ration of filtered to non-filtered (estimate of a neural measure).
      reduced_ratio: Same as above, but reduced but elements reduced prior
        to computation.
  """
  sentence = list(sentence)
  predicted = list(predicted)
  left_target = -1
  right_target = -1
  for i in range(len(sentence)):
    if sentence[i] == "<":
      left_target = i + 1
    elif sentence[i] == ">":
      right_target = i
  assert left_target != -1 and right_target != -1
  input_length = attention_matrix.shape[1]
  output_length = attention_matrix.shape[0]
  predicted_length = len(predicted)
  mask = np.ones([output_length, input_length])
  for j in range(left_target, right_target):
    for i in range(predicted_length):
      mask[i, j] = 0.0
  filtered = attention_matrix * mask
  ratio = np.sum(filtered) / np.sum(attention_matrix)
  reduced_ratio = (np.sum(tf.reduce_max(filtered, axis=0)) /
                   np.sum(tf.reduce_max(attention_matrix, axis=0)))
  return mask, filtered, ratio, reduced_ratio


def plot_attention(matrix, sentence, predicted,
                   figsize=(10, 10), showticks=True,
                   cmap="plasma"):
  """Generates attention matrix plot."""
  sentence = list(sentence)
  predicted = list(predicted)
  fig = plt.figure(figsize=figsize, dpi=400)
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(matrix, cmap=cmap)
  ax.set_xticklabels([""] + sentence, fontproperties=CH_FONT)
  ax.set_yticklabels([""] + predicted, fontproperties=CH_FONT)
  if showticks:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  plt.subplots_adjust(bottom=0.15)
  plt.show()


def eval_and_plot(model, data, indices, show_plots=False,
                  print_predictions=False,
                  print_attention=False,
                  compute_deviation=False,
                  deviation_mask_sigma=0.3,
                  deviation_only_for_correct=False,
                  simple_skew=False,
                  report_type_stats=False,
                  figsize=(40, 100)):
  """Runs evaluation and generates the resulting plots for attention."""
  model.checkpoint.restore(tf.train.latest_checkpoint(model.checkpoint_dir))
  joiner = " "  # Keep it this way so we can use split below.
  cor = 0
  total_ratio = collections.defaultdict(int)
  total_reduced_ratio = collections.defaultdict(int)
  total_deviations = collections.defaultdict(int)
  skip_symbols = ["</s>", "<pad>"]

  # Dummy token "type" to be used if report_type_stats is False.
  all_types = "__***ALL_TYPES***___"

  def prep_text_for_plot(text):
    result = []
    for t in text.split():
      if t == "<spc>":
        t = " "
      elif t == "<targ>":
        t = "<"
      elif t == "</targ>":
        t = ">"
      result.append(t)
    return result

  trivial_targets = frozenset(["<targ><spc></targ>",
                               "<targ>,</targ>",
                               "<targ>.</targ>",
                               "<targ>?</targ>",
                               "<targ>)</targ>",
                               "<targ>(</targ>",
                               "<targ>{</targ>",
                               "<targ>}</targ>",
                               "<targ>―</targ>",
                               "<targ>!</targ>",
                               "<targ>;</targ>",
                               "<targ>:</targ>",
                               "<targ>'</targ>",
                               '<targ>"</targ>',
                               "<targ>、</targ>",
                               "<targ>。</targ>",
                               "<targ>「</targ>",
                               "<targ>」</targ>",
                               "<targ>『</targ>",
                               "<targ>』</targ>",
                               "<targ>・</targ>",
                               "<targ>〔</targ>",
                               "<targ>〕</targ>",
                               "<targ>（</targ>",
                               "<targ>）</targ>",
                               "<targ>．</targ>"])

  def is_trivial(output):
    return output.replace(" ", "") in trivial_targets

  def clean(t):
    return t.replace("<targ>", "").replace("</targ>", "").strip()

  total_predictions = len(indices)

  def find_input_target(input_text):
    target = []
    keep = False
    for t in input_text.split():
      if t == "<targ>":
        keep = True
      elif t == "</targ>":
        break
      elif keep:
        target.append(t)
    return joiner.join(target)

  for idx in indices:
    inp = [int(x) for x in data[idx][0]]
    out = [int(x) for x in data[idx][1]]
    if model.input_length == -1:
      model.update_property("_input_length", len(inp))
      model.update_property("_output_length", len(out))
    input_text = data_lib.decode_index_array(
        inp, model.input_symbols, joiner=joiner, skip_symbols=skip_symbols)
    if report_type_stats:
      input_target = find_input_target(input_text)
    else:
      input_target = all_types
    output_text = data_lib.decode_index_array(
        out, model.output_symbols, joiner=joiner, skip_symbols=skip_symbols)

    if is_trivial(output_text):
      total_predictions -= 1
      continue
    if print_predictions:
      print("*" * 80)
      print("Index: {}".format(idx))
      print("input:\t {}".format(input_text))
      print("output:\t {}".format(output_text))
    prediction, attention_plot = model.decode(inp, joiner=joiner)
    tag = "@"
    correct = False
    if clean(prediction) == clean(output_text):
      tag = " "
      correct = True
      cor += 1
    if print_predictions:
      print("pred:\t{}{}".format(tag, prediction))
    prep_input_text = prep_text_for_plot(input_text)
    prep_prediction = prep_text_for_plot(prediction)
    if print_attention:
      print(list(attention_plot))
    if show_plots:
      plot_attention(attention_plot,
                     prep_input_text,
                     prep_prediction,
                     figsize=figsize)
      # If the answer to "Keep?" is "yes", then save the matrix, input
      # and prediction out to a file.
      response = input("Keep? (Respond 'yes' to keep.)\n")
      if response == "yes":
        try:
          os.mkdir("/var/tmp/attention_plots")
        except FileExistsError:
          pass
        basename = os.path.join("/var/tmp/attention_plots",
                                "_".join(prep_prediction))
        print("Saving data to basename {}".format(basename))
        with open(basename + ".txt", "w") as stream:
          stream.write("{}\n".format(" ".join(prep_input_text)))
          stream.write("{}\n".format(" ".join(prep_prediction)))
        with open(basename + ".arr", "w") as stream:
          for row in attention_plot:
            stream.write(str(list(row)) + "\n")
        sys.exit()
    if compute_deviation:
      if deviation_only_for_correct and not correct:
        continue
      if simple_skew:
        _, _, ratio, reduced_ratio = compute_simple_skew(
            attention_plot, prep_input_text, prep_prediction)
      else:
        _, _, ratio, reduced_ratio = compute_skew_from_ideal(
            attention_plot, prep_input_text, prep_prediction,
            sigma=deviation_mask_sigma)
      total_ratio[input_target] += ratio
      total_reduced_ratio[input_target] += reduced_ratio
      total_deviations[input_target] += 1
  try:
    normalized_ratio = 0
    normalized_reduced_ratio = 0
    length = len(total_ratio)
    for typ in total_ratio:
      normalized_ratio += total_ratio[typ] / total_deviations[typ]
      normalized_reduced_ratio += (
          total_reduced_ratio[typ] / total_deviations[typ])
    normalized_ratio /= length
    normalized_reduced_ratio /= length
    if length > 1:
      print("Averaged over {} types".format(length))
  except ZeroDivisionError:
    normalized_ratio = 0
  return (total_predictions,
          cor / total_predictions,
          normalized_ratio,
          normalized_reduced_ratio)

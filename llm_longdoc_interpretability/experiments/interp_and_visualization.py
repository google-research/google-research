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

"""Plot interpretability results."""

import matplotlib.pyplot as plt
import numpy as np


def plot_big_dict(
    big_dict_of_interpretability_results,
    little_dict_of_interp_results_summary,
    verbose=True,
    plotting=False,
):
  """Plots interpretability results."""

  d_size = big_dict_of_interpretability_results["size"]

  possible_text_keys = []
  for d in range(d_size):
    for best_text in little_dict_of_interp_results_summary[d]:
      if best_text not in possible_text_keys:
        possible_text_keys.append(best_text)

  possible_text_occurences = {}
  for d in range(d_size):
    for text_pair in big_dict_of_interpretability_results[d]:
      best_text = text_pair[1]
      if best_text not in possible_text_occurences:
        possible_text_occurences[best_text] = 1
      else:
        possible_text_occurences[best_text] += 1

  num_keys = len(possible_text_keys)
  tot_occ = 0
  for c in range(num_keys):
    best_text = possible_text_keys[c]
    occ = possible_text_occurences[best_text]
    tot_occ += occ
  eps = 0.01
  new_possible_text = []
  for c in range(num_keys):
    best_text = possible_text_keys[c]
    occ = possible_text_occurences[best_text]
    if occ / tot_occ >= eps:
      new_possible_text.append(best_text)

  new_possible_text = sorted(
      new_possible_text, key=lambda x: -possible_text_occurences[x]
  )
  if verbose:
    print(new_possible_text)
  num_of_diff_generated_outputs = len(new_possible_text)
  remap = {}
  for c2 in range(num_of_diff_generated_outputs):
    best_text2 = new_possible_text[c2]
    occ2 = possible_text_occurences[best_text2]
    if verbose:
      print(c2, "\t", "{:30s}".format(best_text2), occ2)
    for c in range(num_keys):
      if new_possible_text[c2] == possible_text_keys[c]:
        remap[c2] = c
  remap[num_of_diff_generated_outputs] = num_keys
  if verbose:
    print()
    print()

  dict_to_convert = big_dict_of_interpretability_results
  dimension_local = dict_to_convert["size"]
  num_keys = len(possible_text_keys)
  pp = len(dict_to_convert[0])
  results_arr = np.zeros((2, pp, dimension_local, num_keys))
  for dz in range(dimension_local):
    for p in range(pp):
      prev_c = -1
      curr_c = -1
      if dict_to_convert[dz][p][0] in possible_text_keys:
        prev_c = possible_text_keys.index(dict_to_convert[dz][p][0])
      if dict_to_convert[dz][p][1] in possible_text_keys:
        curr_c = possible_text_keys.index(dict_to_convert[dz][p][1])
      if prev_c != curr_c:
        cs = (prev_c, curr_c)
        for ior in range(2):  # include_or_remove
          results_arr[ior, p, dz, cs[ior]] = 1
  xd = results_arr

  xd = 0 * xd[0] + 1 * xd[1]
  xd2 = np.sqrt(np.var(xd, axis=0))
  xd = np.mean(xd, axis=0)
  dimension_local = xd.shape[0]

  if plotting:
    plt.figure(figsize=(12, 8))
    plt.title("P=" + str(pp))
    for c2 in range(num_of_diff_generated_outputs):
      c = remap[c2]

      m_c = xd[:, c]
      v_c = xd2[:, c] / np.sqrt(pp)
      if c2 < num_of_diff_generated_outputs:
        MAX_STRLEN = 34  # pylint: disable=invalid-name
        label = (
            new_possible_text[c2][:MAX_STRLEN]
            + (len(new_possible_text[c2]) > MAX_STRLEN) * "..."
        )
        plt.scatter(np.arange(dimension_local), m_c, label=label)
        plt.fill_between(
            np.arange(dimension_local),
            m_c - v_c,
            m_c + v_c,
            alpha=0.3,
            color="C" + str(c2 % 10),
        )
        plt.plot(np.arange(dimension_local), m_c)
      else:
        label = "-1"
        plt.scatter(np.arange(dimension_local), m_c, label=label, c="k")
        plt.plot(np.arange(dimension_local), m_c, c="k")
        plt.fill_between(
            np.arange(dimension_local),
            m_c - v_c,
            m_c + v_c,
            alpha=0.3,
            color="k",
        )

    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.show()

    _, ax = plt.subplots(figsize=(12, 8))
    plt.title("P=" + str(pp))
    bottom = np.zeros(dimension_local)
    for c2 in range(num_of_diff_generated_outputs):
      c = remap[c2]
      if c2 < num_of_diff_generated_outputs:
        label = new_possible_text[c2][:40]
        ax.bar(np.arange(dimension_local), xd[:, c], label=label, bottom=bottom)
        bottom += xd[:, c]
      else:
        label = "-1"
        ax.bar(
            np.arange(dimension_local),
            xd[:, c],
            label=label,
            color="k",
            bottom=bottom,
        )
    plt.plot([-1, dimension_local + 1], [0, 0], c="k")
    plt.legend()
    plt.xlim(-0.8, dimension_local - 1 + 0.8)
    plt.ylim(-0.05, 1.05)
    plt.show()

  return xd

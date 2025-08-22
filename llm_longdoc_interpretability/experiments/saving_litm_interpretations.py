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

"""Top-level code for the LitM experiments."""

import copy
import json
import pickle
import time

from absl import app
from absl import flags
import numpy as np
from t5_with_flash.modeling_t5_with_flash import return_pretrained_model
import torch


_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "flan-t5-xxl",
    "The pretrained model's name. Can only be 'flan-t5-large' or 'flan-t5-xxl'"
    " for this script.",
)

_MIN_SAMPLE_ID = flags.DEFINE_integer(
    "min_sample_id", 0, "The starting index for the samples to iterate over."
)
_MAX_SAMPLE_ID = flags.DEFINE_integer(
    "max_sample_id",
    100,
    (
        "The ending index for the samples to iterate over."
        "At most 2655 for NQ and 2863 for MIRACL."
    ),
)

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    8,
    (
        "The batch size. 1,2,4,8 are fine for XXL."
        "larger sizes are fine for T5-large."
    ),
)

_INTERPRETABILITY_SAMPLES = flags.DEFINE_integer(
    "interpretability_samples",
    100,
    (
        "The number of samples to use in the approximation"
        "algorithms for interpretability."
        "For Shapley it is number of permutations."
        "For Banzhaf it is number of subsets."
        "Default is 100."
    ),
)

_DECODER_SPECULATION_IS_ON = flags.DEFINE_bool(
    "decoder_speculation_is_on",
    True,
    "Whether the decoder speculation is on.Default is True.",
)

_HIERARCHY_HYPERPARAMETERS_TO_CHECK = flags.DEFINE_multi_integer(
    "hierarchy_hyperparameter_list",
    [
        [0.3, 0.3],
        [0.3, 0.1],
        [0.1, 0.1],
    ],
    "List of lists of the two hierarchy hyperparameters to check.",
)


# pylint: disable=invalid-name
def get_litm_file_name(P, nop, gi, x):
  """Builds LitM file names."""
  file_name = (
      "LitM_shap_shup_dict_obj__P"
      + str(P)
      + "_nop_"
      + str(nop)
      + "_gi_"
      + str(gi)
      + "_x"
      + str(x)
      + ".pkl"
  )
  return file_name


def get_litm_object_to_save(
    shap_shup_perm_dict_obj,
    locations_to_check,
    locations_checked,
    first_layer,
    possible_text_keys,
    possible_text_occurences,
    total_time_taken_each_z,
    P,
    THRESHS,
    x,
    nop,
    gi,
):
  """Builds dict of LitM experiment results."""
  parameters = {
      "P": P,
      "THRESHS": THRESHS,
      "x": x,
      "nop": nop,
      "gi": gi,
  }
  thing_to_save = {
      "shap_shup_perm_dict_obj": shap_shup_perm_dict_obj,
      "locations_to_check": locations_to_check,
      "locations_checked": locations_checked,
      "first_layer": first_layer,
      "possible_text_keys": possible_text_keys,
      "possible_text_occurences": possible_text_occurences,
      "total_time_taken_each_z": total_time_taken_each_z,
      "parameters": parameters,
  }
  return thing_to_save


def get_all_litm_file_paths():
  """Gets all LitM candidate file paths."""
  ten_doc_gold_indices = [0, 4, 9]
  twenty_doc_gold_indices = [0, 4, 9, 14, 19]
  thirty_doc_gold_indices = [0, 4, 9, 14, 19, 24, 29]
  possible_gold_inds_dict = {
      10: ten_doc_gold_indices,
      20: twenty_doc_gold_indices,
      30: thirty_doc_gold_indices,
  }
  all_candidate_combinations = []
  for no_of_passages in [10, 20, 30]:
    nop = no_of_passages
    nop_gold_indices = possible_gold_inds_dict[nop]
    for gi in nop_gold_indices:
      all_candidate_combinations.append((nop, gi))
  all_candidate_paths = {}
  base_path = "lost-in-the-middle-main/qa_data/"

  for no_of_passages in [10, 20, 30]:
    nop = no_of_passages
    doc_base_path = base_path + str(nop) + "_total_documents/"

    nop_gold_indices = possible_gold_inds_dict[nop]
    for gi in nop_gold_indices:
      final_path = (
          doc_base_path
          + "nq-open-"
          + str(nop)
          + "_total_documents_gold_at_"
          + str(gi)
          + ".jsonl"
      )
      all_candidate_paths[(nop, gi)] = final_path
  return all_candidate_paths


def process_documents_from_index(json_list_gi, x, nop, gi, t):
  """Processes documents from index."""
  json_str = json_list_gi[x]
  result = json.loads(json_str)
  ctx = result["ctxs"]
  question = result["question"]
  search_results = ""

  alr_gi = 0  # gold_index is already found
  if gi == 0:
    alr_gi += 1
    search_results += (
        f"Document [{str(gi+1)}] (Title: {ctx[0]['title']}) {ctx[0]['text']}\n"
    )
  for dd, doc in enumerate(ctx[1:nop]):
    search_results += (
        f"Document [{str(dd+alr_gi+1)}] (Title: {doc['title']}) {doc['text']}\n"
    )
    if dd + alr_gi + 1 == nop:
      break
    if dd + 1 == gi:
      alr_gi += 1
      search_results += (
          f"Document [{str(dd+alr_gi+1)}] (Title: {ctx[0]['title']})"
          f" {ctx[0]['text']}\n"
      )
      if dd + alr_gi + 1 == nop:
        break

  prompt = (
      "Write a high-quality answer for the given question using only the"
      " provided search results (some of which might be"
      f" irrelevant).\n\n{search_results}\nQuestion: {question}\nAnswer:"
  )
  tokens = t.encode(prompt)
  token_strings = [t.decode(thing) for thing in tokens]

  doc_id = t.encode("Document")[0]
  doc_indices = np.argwhere(np.array(tokens) == doc_id)[:, 0]
  doc_indices = list(doc_indices)

  q_id = t.encode("Question")[0]
  q_ind = np.argwhere(np.array(tokens) == q_id)[:, 0]
  doc_indices.extend(list(q_ind))

  first_layer = {}
  for dd in range(nop):
    doc_span_tup_dd = (
        doc_indices[dd],
        doc_indices[dd + 1],
    )

    doc_dd_splits = []
    doc_dd = token_strings[doc_span_tup_dd[0] : doc_span_tup_dd[1]]
    doc_dd_toks = tokens[doc_span_tup_dd[0] : doc_span_tup_dd[1]]

    doc = ctx[dd]
    prompt_end_ind = (
        len(
            t.encode(
                "Document [" + str(dd + 1) + "] (Title: " + doc["title"] + ")"
            )
        )
        - 1
    )
    doc_dd_splits.append(0)
    doc_dd_splits.append(prompt_end_ind)

    things_to_split = [
        ".",
        ",",
    ]
    for xx in range(prompt_end_ind, len(doc_dd)):
      if doc_dd[xx] in things_to_split:
        doc_dd_splits.append(xx + 1)
    last = len(doc_dd)
    if last not in doc_dd_splits:
      doc_dd_splits.append(last)

    queue = [7904]  # arbitrary token used for splitting
    toktups = []
    for xx in range(len(doc_dd_toks)):
      queue.append(doc_dd_toks[xx])
      if len(t.decode(queue).split(" ")) > 2:
        toktups.append(queue[1:-1])
        while len(queue) > 2:
          queue.pop(0)
    toktups.append(queue[1:])

    second_layer = {}
    for dd_ph in range(len(doc_dd_splits) - 1):
      phrase_span_tup_ph = (
          doc_dd_splits[dd_ph],
          doc_dd_splits[dd_ph + 1],
      )
      third_layer = {}
      word_pos = 0
      xx = 0
      for _, toktup in enumerate(toktups):
        if (
            word_pos >= phrase_span_tup_ph[0]
            and word_pos + len(toktup) <= phrase_span_tup_ph[1]
        ):
          word_tup = (
              word_pos - phrase_span_tup_ph[0],
              word_pos + len(toktup) - phrase_span_tup_ph[0],
          )
          third_layer[xx] = (None, word_tup)
          xx += 1
        word_pos += len(toktup)
      second_layer[dd_ph] = (third_layer, phrase_span_tup_ph)
    first_layer[dd] = (second_layer, doc_span_tup_dd)
  return first_layer, tokens


def update_perm_object_with_locations(
    shap_shup_perm_dict_obj, locations_to_check, first_layer
):
  """Updates perm object with locations to check."""
  loc0 = []
  if tuple(loc0) in locations_to_check:
    D0 = len(first_layer)
    dict0 = {"size": D0}
    for d0 in range(D0):
      dict0[d0] = []
    if tuple(loc0) not in shap_shup_perm_dict_obj:
      shap_shup_perm_dict_obj[tuple(loc0)] = dict0
  for thing1 in first_layer:
    loc1 = copy.copy(loc0)
    loc1.append(thing1)
    if tuple(loc1) in locations_to_check:
      second_layer = first_layer[thing1][0]
      D1 = len(second_layer)
      dict1 = {"size": D1}
      for d1 in range(D1):
        dict1[d1] = []
      if tuple(loc1) not in shap_shup_perm_dict_obj:
        shap_shup_perm_dict_obj[tuple(loc1)] = dict1
      for thing2 in second_layer:
        loc2 = copy.copy(loc1)
        loc2.append(thing2)
        if tuple(loc2) in locations_to_check:
          third_layer = second_layer[thing2][0]
          D2 = len(third_layer)
          dict2 = {"size": D2}
          for d2 in range(D2):
            dict2[d2] = []
          if tuple(loc2) not in shap_shup_perm_dict_obj:
            shap_shup_perm_dict_obj[tuple(loc2)] = dict2


def get_model_best_text(
    inp_ids, model4, t, speculative_tree=None, verbose=False
):
  """Gets the best text from the model."""
  B = 1
  MAXLEN = 20
  if speculative_tree is not None:
    if B > 1:
      raise NotImplementedError(
          "Does not currently support multiple beams, B =", B
      )
    else:
      outputs = model4.jam_speculative_generate(
          inp_ids,
          max_length=MAXLEN,
          return_dict_in_generate=True,
          output_scores=True,
          speculative_generation_tree=speculative_tree,
      )

      if verbose:
        bs_ind = 0
        normed_scores = torch.nn.functional.log_softmax(
            outputs.scores[0][bs_ind, :], dim=-1
        )
        full_next_step = torch.argmax(normed_scores, axis=-1).cpu()
        print("full_next_step")
        print(full_next_step)
        print(speculative_tree["tok_list"])
      # VERIFY IF THE SPECULATION TREE HAS SUCEEDED
      dec_toks = [0]
      curr_tok = 0
      curr_ind = 0
      curr_tree = speculative_tree["base_tree"]
      found = False
      while curr_tok != -1:
        bs_ind = 0
        next_score = int(
            torch.argmax(outputs.scores[0][bs_ind, curr_ind], axis=-1).cpu()
        )
        dec_toks.append(next_score)

        if (
            len(dec_toks) == MAXLEN
        ):  # handles the case of maxlen not in tree but never breaks
          found = True
          break
        if next_score == 1:
          found = True
          # TODO(enounen) - this leads to a slight missing case where the
          # "#1" doesnt actually exist in the tree, but it doesn't matter bc its
          # not needed for the computation anyways
          break
        if next_score in curr_tree:
          curr_tok = next_score
          curr_ind = curr_tree[curr_tok][0]
          curr_tree = curr_tree[curr_tok][1]
        else:  # not found, so need to continue
          curr_tok = -1

      if not found:
        outputs = model4.generate(
            inp_ids,
            max_length=MAXLEN,
            return_dict_in_generate=True,
            output_scores=True,
        )
    if not found:
      all_tokens = outputs.sequences[:, 1:].cpu()
      all_text = t.batch_decode(all_tokens, skip_special_tokens=True)
    else:
      all_tokens = torch.LongTensor([dec_toks])
      all_text = t.batch_decode(all_tokens, skip_special_tokens=True)

    if not found:  # just pass all instances through
      extend_speculative_tree(speculative_tree, all_tokens[0])

  else:  # speculation is totally off
    found = False
    outputs = model4.generate(
        inp_ids,
        max_length=MAXLEN,
        return_dict_in_generate=True,
        output_scores=True,
    )
    all_tokens = outputs.sequences[:, 1:].cpu()
    all_text = t.batch_decode(all_tokens, skip_special_tokens=True)

  return found, all_text[0], all_tokens[0]


# adding "1"s = eos token is slightly inefficient, but doesnt matter bc of how
# they are used
def extend_speculative_tree(spec_tree, new_toks):
  """Extends the speculative tree with new tokens."""
  tok_list = spec_tree["tok_list"]
  curr_tree = spec_tree["base_tree"]
  attn_mask = spec_tree["attn_mask"]
  posi_inds = spec_tree["pos_bias_inds"]

  path_index = [0]
  columns_to_add = {}
  for ll in range(len(new_toks)):
    curr_tok = int(new_toks[ll])
    if curr_tok == 0:  # end for padding
      break
    need_to_add = False
    if curr_tok not in curr_tree:
      curr_tree[curr_tok] = (len(tok_list), {})
      tok_list.append(curr_tok)
      need_to_add = True
    new_index = curr_tree[curr_tok][0]
    curr_tree = curr_tree[curr_tok][1]
    path_index.append(new_index)
    if need_to_add:
      columns_to_add[new_index] = copy.deepcopy(path_index)

  # adds each of the new columns to the causal matrix, only creating the
  # new instance a single time
  if columns_to_add.keys():
    attn_mask = torch.nn.functional.pad(
        attn_mask,
        (
            0,
            len(tok_list) - attn_mask.shape[1],
            0,
            len(tok_list) - attn_mask.shape[1],
        ),
    )
    posi_inds = torch.nn.functional.pad(
        posi_inds,
        (
            0,
            len(tok_list) - posi_inds.shape[1],
            0,
            len(tok_list) - posi_inds.shape[1],
        ),
    )
    for col_ind in columns_to_add:
      col = columns_to_add[col_ind]
      col_len = len(col)
      for cc, prev_ind in enumerate(col):
        attn_mask[col_ind, prev_ind] = 1
        posi_inds[col_ind, prev_ind] = cc + 1 - col_len
    spec_tree["attn_mask"] = attn_mask
    spec_tree["pos_bias_inds"] = posi_inds


# just focus on experiments doing any gold index with docs=10
def get_relevant_tups_10doc():
  tups = []
  nop = 10
  for gi in range(10):
    tups.append((nop, gi))
  return tups


def main():
  model_name = _MODEL_NAME.value
  P = _INTERPRETABILITY_SAMPLES.value
  TURN_SPEC_OFF = not _DECODER_SPECULATION_IS_ON.value
  threshes_to_check = _HIERARCHY_HYPERPARAMETERS_TO_CHECK.value

  # LitM saving dictionaries
  min_x = _MIN_SAMPLE_ID.value
  max_x = _MAX_SAMPLE_ID.value
  xs = list(range(min_x, max_x))

  device = torch.device("cuda:0")
  model4, t = return_pretrained_model(model_name)
  start_load = time.time()
  model4.to(device)
  print("model loaded")
  print(time.time() - start_load)

  all_candidate_paths = get_all_litm_file_paths()
  tups = get_relevant_tups_10doc()

  accum_z_takens = {}
  thirty_doc_path = all_candidate_paths[(30, 0)]
  with open(thirty_doc_path, "r") as json_file:
    json_list_gi = list(json_file)

  for x in xs:
    for threshes in threshes_to_check:
      print("x", x)
      print("th", threshes)
      for tup in tups:
        nop, gi = tup
        print("tup", tup)

        first_layer, tokens = process_documents_from_index(
            t, json_list_gi, x, nop, gi
        )

        locations_to_check = [
            (),
        ]
        locations_checked = []
        shap_shup_perm_dict_obj = {}
        update_perm_object_with_locations(
            shap_shup_perm_dict_obj, locations_to_check, first_layer
        )

        the_tokens = tokens
        possible_text_keys = []
        possible_text_occurences = {}

        full_start_time = time.time()
        perm_time_takens = []
        total_time_taken_each_z = []

        # TODO(enounen) - In theory, this should be converted to
        # something which can deal with an arbitrary hierarchy of sets of
        # features. The iteration over z three times is for (1) doing the
        # passage level (2) doing the sentence level (3) doing the word level.
        # at the end of each iteration, we check the next depth of hierarchy if
        # there were any important features which we should "look deeper into"

        # Therefore, this looping structure will likely always be required to be
        # able to only investigate the sentences inside of the important
        # paragraphs; however, the recursion into the tree is likely handled
        # better via a recursive call.
        # I implemented this because we were using a fixed depth hierarchy
        # (three) and it is easier to write this code than the alternative,
        # especially if concerned with the speed overheads from processing
        # external to the GPU and needing to mix batches across different
        # levels of the hierarchy and then accumulate these results in post with
        # respect to the GPU running. (just requires recalling all of the
        # relevant details necessary from the permutation sampling, but an
        # efficient way of storing the hierarchy of permutations or generating
        # a full permutation respecting the hierarchical structure both did not
        # have obvious solutions for me.)
        spec_tree_x = {
            "base_tree": {},
            "tok_list": [0],
            "attn_mask": torch.ones((1, 1), dtype=bool),
            "pos_bias_inds": torch.zeros((1, 1), dtype=int),
        }
        if TURN_SPEC_OFF:
          spec_tree_x = None
        for z in range(3):
          print("z", z)

          p = 0
          stop_the_sampling = False
          while p < P and (not stop_the_sampling):
            perm_start_time = time.time()

            inp_ids = torch.LongTensor(the_tokens).to(device)[None] * 0
            prefix_tup = (0, first_layer[0][1][0])
            prefix_to_add = torch.LongTensor(
                the_tokens[prefix_tup[0] : prefix_tup[1]]
            ).to(device)
            inp_ids[0, prefix_tup[0] : prefix_tup[1]] = prefix_to_add
            suffix_tup = (
                first_layer[len(first_layer) - 1][1][1],
                len(the_tokens),
            )
            suffix_to_add = torch.LongTensor(
                the_tokens[suffix_tup[0] : suffix_tup[1]]
            ).to(device)
            inp_ids[0, suffix_tup[0] : suffix_tup[1]] = suffix_to_add

            loc0 = []
            if tuple(loc0) in locations_to_check:
              D0 = len(first_layer)
              perm0 = np.random.permutation(D0)

              prev_c0 = -1
              for d0 in range(D0):
                pd0 = perm0[d0]
                span0tup = first_layer[pd0][1]

                loc1 = copy.copy(loc0)
                loc1.append(pd0)
                if tuple(loc1) in locations_to_check:
                  layer1 = first_layer[pd0][0]
                  D1 = len(layer1)
                  perm1 = np.random.permutation(D1)
                  prev_c1 = prev_c0

                  for d1 in range(D1):
                    pd1 = perm1[d1]
                    span1tup = layer1[pd1][1]
                    span1tup = (
                        span0tup[0] + span1tup[0],
                        span0tup[0] + span1tup[1],
                    )

                    loc2 = copy.copy(loc1)
                    loc2.append(pd1)
                    if tuple(loc2) in locations_to_check:
                      layer2 = layer1[pd1][0]
                      D2 = len(layer2)
                      perm2 = np.random.permutation(D2)
                      prev_c2 = prev_c1

                      for d2 in range(D2):
                        pd2 = perm2[d2]
                        span2tup = layer2[pd2][1]
                        span2tup = (
                            span1tup[0] + span2tup[0],
                            span1tup[0] + span2tup[1],
                        )

                        span_to_add = torch.LongTensor(
                            the_tokens[span2tup[0] : span2tup[1]]
                        ).to(device)
                        inp_ids[0, span2tup[0] : span2tup[1]] = span_to_add

                        _, best_text2, _ = get_model_best_text(
                            inp_ids, model4, t, spec_tree_x
                        )
                        if best_text2 in possible_text_keys:
                          curr_c2 = possible_text_keys.index(best_text2)
                          possible_text_occurences[best_text2] += 1
                        else:
                          curr_c2 = len(possible_text_keys)
                          possible_text_keys.append(best_text2)
                          possible_text_occurences[best_text2] = 1
                        shap_shup_perm_dict_obj[tuple(loc2)][pd2].append(
                            (prev_c2, curr_c2)
                        )
                        prev_c2 = curr_c2
                      curr_c1 = prev_c2
                    else:
                      span_to_add = torch.LongTensor(
                          the_tokens[span1tup[0] : span1tup[1]]
                      ).to(device)
                      inp_ids[0, span1tup[0] : span1tup[1]] = span_to_add
                      _, best_text1, _ = get_model_best_text(
                          inp_ids, model4, t, spec_tree_x
                      )
                      if best_text1 in possible_text_keys:
                        curr_c1 = possible_text_keys.index(best_text1)
                        possible_text_occurences[best_text1] += 1
                      else:
                        curr_c1 = len(possible_text_keys)
                        possible_text_keys.append(best_text1)
                        possible_text_occurences[best_text1] = 1
                      shap_shup_perm_dict_obj[tuple(loc1)][pd1].append(
                          (prev_c1, curr_c1)
                      )
                    prev_c1 = curr_c1
                  curr_c0 = prev_c1

                else:
                  span_to_add = torch.LongTensor(
                      the_tokens[span0tup[0] : span0tup[1]]
                  ).to(device)
                  inp_ids[0, span0tup[0] : span0tup[1]] = span_to_add
                  _, best_text0, _ = get_model_best_text(
                      inp_ids, model4, t, spec_tree_x
                  )
                  if best_text0 in possible_text_keys:
                    curr_c0 = possible_text_keys.index(best_text0)
                    possible_text_occurences[best_text0] += 1
                  else:
                    curr_c0 = len(possible_text_keys)
                    possible_text_keys.append(best_text0)
                    possible_text_occurences[best_text0] = 1
                  shap_shup_perm_dict_obj[tuple(loc0)][pd0].append(
                      (prev_c0, curr_c0)
                  )
                prev_c0 = curr_c0
              pass
            perm_time_taken = time.time() - perm_start_time
            perm_time_takens.append(perm_time_taken)
            p += 1

          total_time_taken = time.time() - full_start_time
          print("total_time_taken", total_time_taken)
          total_time_taken_each_z.append(total_time_taken)

          # CHECK FOR MORE LOCATIONS
          if z < 3 - 1:
            temp = []
            while locations_to_check:
              loc_z = locations_to_check.pop(0)
              if loc_z not in locations_checked:
                locations_checked.append(loc_z)
                dict_to_convert = shap_shup_perm_dict_obj[loc_z]
                D_loc = dict_to_convert["size"]
                C_z = len(possible_text_keys)
                results_arr = np.zeros((2, P, D_loc, C_z))
                for dz in range(D_loc):
                  for p in range(P):
                    if dict_to_convert[dz][p][0] != dict_to_convert[dz][p][1]:
                      for ior in range(2):  # include_or_remove
                        results_arr[
                            ior, p, dz, dict_to_convert[dz][p][1 - ior]
                        ] = 1
                m_z = np.mean(results_arr[0], axis=0)
                v_z = np.sqrt(np.var(results_arr[0], axis=0)) / np.sqrt(
                    P
                )  # CI estimate not variance estimate

                score = np.max(m_z - v_z, axis=-1)
                for dz in range(D_loc):
                  if score[dz] >= threshes[z]:
                    newloc = copy.deepcopy(list(loc_z))
                    newloc.append(dz)
                    temp.append(tuple(newloc))
            print("temp", temp)
            locations_to_check = copy.deepcopy(locations_checked)
            locations_to_check.extend(temp)
            update_perm_object_with_locations(
                shap_shup_perm_dict_obj, locations_to_check, first_layer
            )
            print("locations_to_check", locations_to_check)

          print("total_time_taken", total_time_taken)
          print("z=", z, "  completed")
          print()

        print("total_time_taken_each_z")
        print(total_time_taken_each_z)
        accum_z_takens[(x, tuple(threshes))] = total_time_taken_each_z
        print("accum_z_takens")
        print(accum_z_takens)

        SAVING = True
        if SAVING:
          results_path = "results_data/shap_shub_dicts_LitM/"
          thing_to_save = get_litm_object_to_save(
              shap_shup_perm_dict_obj,
              locations_to_check,
              locations_checked,
              first_layer,
              possible_text_keys,
              possible_text_occurences,
              total_time_taken_each_z,
              P,
              threshes,
              x,
              nop,
              gi,
          )
          file_name = get_litm_file_name(P, nop, gi, x)
          pickle.dump(thing_to_save, open(results_path + file_name, "wb"))


if __name__ == "__main__":
  app.run(main)

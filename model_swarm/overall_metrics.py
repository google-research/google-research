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

import os
import json
import argparse
from sklearn.metrics import accuracy_score

def avg(lst):
    return sum(lst) / len(lst)

def ensemble_based_on_utility(preds, utility_list, top_k):
    """prediction ensemble of particles with top_k utility values."""
    assert len(preds) == len(utility_list)

    if top_k:
        preds_new = []
        utility_list_new = []
        top_k_bar = sorted(utility_list, reverse=True)[top_k-1]
        i = 0
        while len(utility_list_new) < top_k:
            if utility_list[i] >= top_k_bar:
                preds_new.append(preds[i])
                utility_list_new.append(utility_list[i])
            i += 1
        assert len(preds_new) == len(utility_list_new)
        preds = preds_new
        utility_list = utility_list_new

    assert len(preds) == len(utility_list) == top_k

    final_preds = []
    for i in range(len(preds[0])):
        preds_this_problem = []
        for j in range(len(preds)):
            preds_this_problem.append(preds[j][i])
        pred_frequency_dict = {}
        pred_utility_sum_dict = {}
        j = len(preds)-1
        for pred in preds_this_problem:
            if pred in pred_frequency_dict:
                pred_frequency_dict[pred] += 1
                pred_utility_sum_dict[pred] += utility_list[j]
            else:
                pred_frequency_dict[pred] = 1
                pred_utility_sum_dict[pred] = utility_list[j]
        max_frequency = max(pred_frequency_dict.values())
        max_utility = max(pred_utility_sum_dict.values())
        max_frequency_achieved_flag = [pred_frequency_dict[pred] == max_frequency for pred in pred_frequency_dict]
        if sum(max_frequency_achieved_flag) == 1:
            # choose the key with max_frequency
            for key in pred_frequency_dict:
                if pred_frequency_dict[key] == max_frequency:
                    final_preds.append(key)
                    break
        else:
            # choose the key with max_utility
            for key in pred_utility_sum_dict:
                if pred_utility_sum_dict[key] == max_utility:
                    final_preds.append(key)
                    break
    assert len(final_preds) == len(preds[0])
    return final_preds

def overall_metrics(name, eval_type, top_k = 10):
    """calculate a bunch of metrics for a given search to report at the end."""

    final_metrics = {}

    particle_paths = os.listdir(os.path.join("search", name))

    if eval_type == "multiple_choice" or eval_type == "AbstainQA" or eval_type == "multitask":

        golds = None
        starting_preds = []
        starting_utility = []

        ending_preds = []
        ending_utility = []

        # for particle_path in particle_paths:
        #     if "particle" in particle_path:
        for i in range(len(particle_paths)):
            if "particle_" + str(i) in particle_paths:
                particle_path = "particle_" + str(i)
                with open(os.path.join("search", name, particle_path, "personal_best/preds.json"), "r") as f:
                    particle_data = json.load(f)
                    ending_preds.append(particle_data)
                with open(os.path.join("search", name, particle_path, "personal_best/golds.json"), "r") as f:
                    gold_data = json.load(f)
                    if golds and not eval_type == "AbstainQA":
                        assert golds == gold_data
                    else:
                        golds = gold_data

        with open(os.path.join("search", name, "utility_scratchpad.json"), "r") as f:
            utility_data = json.load(f)
            for i in range(len(particle_paths)):
                if "particle_" + str(i) in particle_paths:
                    particle_path = "particle_" + str(i)
                    ending_utility.append(utility_data[particle_path + "_best"])

        starting_eval_flag = True
        try:
            for i in range(len(particle_paths)):
                if "particle_" + str(i) in particle_paths:
                    particle_path = "particle_" + str(i)
                    with open(os.path.join("search", name, particle_path, "now/preds.json"), "r") as f:
                        particle_data = json.load(f)
                        starting_preds.append(particle_data)
                    with open(os.path.join("search", name, particle_path, "now/golds.json"), "r") as f:
                        gold_data = json.load(f)
                        if golds and not eval_type == "AbstainQA":
                            assert golds == gold_data
                        else:
                            golds = gold_data

            with open(os.path.join("search", name, "utility_scratchpad.json"), "r") as f:
                utility_data = json.load(f)
                for i in range(len(particle_paths)):
                    if "particle_" + str(i) in particle_paths:
                        particle_path = "particle_" + str(i)
                        starting_utility.append(utility_data[particle_path + "_history"][0])
        except:
            print("no starting eval! starting will be the same as ending")
            starting_eval_flag = False
            starting_utility = ending_utility
            starting_preds = ending_preds

        assert len(starting_preds) == len(starting_utility) == len(ending_preds) == len(ending_utility)
        assert len(golds) == len(starting_preds[0]) == len(ending_preds[0])

        final_starting_preds = ensemble_based_on_utility(starting_preds, starting_utility, top_k)
        final_ending_preds = ensemble_based_on_utility(ending_preds, ending_utility, top_k)

        assert len(final_starting_preds) == len(final_ending_preds)

        starting_best_utility_index = starting_utility.index(max(starting_utility))
        ending_best_utility_index = ending_utility.index(max(ending_utility))

        if starting_eval_flag:

            final_metrics["starting_best_validation_utility"] = max(starting_utility)
            final_metrics["starting_best_particle_on_validation"] = starting_best_utility_index
            final_metrics["starting_best_single_test_accuracy"] = accuracy_score(golds, starting_preds[starting_best_utility_index])
            final_metrics["starting_top-k_ensemble_test_accuracy"] = accuracy_score(golds, final_starting_preds)

            print("starting best validation utility: ", max(starting_utility))
            print("starting best particle on validation: ", starting_best_utility_index)
            print("starting best single test accuracy: ", accuracy_score(golds, starting_preds[starting_best_utility_index]))
            print("starting top-k ensemble test accuracy: ", accuracy_score(golds, final_starting_preds))

        final_metrics["ending_best_validation_utility"] = max(ending_utility)
        final_metrics["ending_best_particle_on_validation"] = ending_best_utility_index
        final_metrics["ending_best_single_test_accuracy"] = accuracy_score(golds, ending_preds[ending_best_utility_index])
        final_metrics["ending_top-k_ensemble_test_accuracy"] = accuracy_score(golds, final_ending_preds)

        print("ending best validation utility: ", max(ending_utility))
        print("ending best particle on validation: ", ending_best_utility_index)
        print("ending best single test accuracy: ", accuracy_score(golds, ending_preds[ending_best_utility_index]))
        print("ending top-k ensemble test accuracy: ", accuracy_score(golds, final_ending_preds))

    elif eval_type == "exact_match" or eval_type == "external_api" or eval_type == "perplexity" or eval_type == "rm_default" or eval_type == "rm_concise" or eval_type == "rm_verbose" or eval_type == "human":
        starting_scores = []
        starting_utility = []
        ending_scores = []
        ending_utility = []

        for i in range(len(particle_paths)):
            if "particle_" + str(i) in particle_paths:
                particle_path = "particle_" + str(i)
                with open(os.path.join("search", name, particle_path, "personal_best/scores.json"), "r") as f:
                    particle_data = json.load(f)
                    ending_scores.append(particle_data)

        with open(os.path.join("search", name, "utility_scratchpad.json"), "r") as f:
            utility_data = json.load(f)
            for i in range(len(particle_paths)):
                if "particle_" + str(i) in particle_paths:
                    particle_path = "particle_" + str(i)
                    ending_utility.append(utility_data[particle_path + "_best"])

        starting_eval_flag = True
        try:
            for i in range(len(particle_paths)):
                if "particle_" + str(i) in particle_paths:
                    particle_path = "particle_" + str(i)
                    with open(os.path.join("search", name, particle_path, "now/scores.json"), "r") as f:
                        particle_data = json.load(f)
                        starting_scores.append(particle_data)

            with open(os.path.join("search", name, "utility_scratchpad.json"), "r") as f:
                utility_data = json.load(f)
                for i in range(len(particle_paths)):
                    if "particle_" + str(i) in particle_paths:
                        particle_path = "particle_" + str(i)
                        starting_utility.append(utility_data[particle_path + "_history"][0])
        except:
            print("no starting eval! starting will be the same as ending")
            starting_eval_flag = False
            starting_utility = ending_utility
            starting_scores = ending_scores

        assert len(starting_scores) == len(starting_utility) == len(ending_scores) == len(ending_utility)

        starting_best_utility_index = starting_utility.index(max(starting_utility))
        ending_best_utility_index = ending_utility.index(max(ending_utility))

        if starting_eval_flag:

            final_metrics["starting_best_validation_utility"] = max(starting_utility)
            final_metrics["starting_best_particle_on_validation"] = starting_best_utility_index
            final_metrics["starting_best_single_test_accuracy"] = sum(starting_scores[starting_best_utility_index]) / len(starting_scores[starting_best_utility_index])
            if eval_type == "exact_match" or eval_type == "rm_default" or eval_type == "rm_concise" or eval_type == "rm_verbose" or eval_type == "human":
                temp_scores = ensemble_based_on_utility(starting_scores, starting_utility, top_k)
                final_metrics["starting_top-k_ensemble_test_accuracy"] = sum(temp_scores) / len(temp_scores)
            elif eval_type == "external_api":
                top_k_utility_bar = sorted(starting_utility, reverse=True)[:top_k]
                retained_scores_list = []
                for i in range(len(starting_scores)):
                    if starting_utility[i] in top_k_utility_bar:
                        retained_scores_list.append(starting_scores[i])
                # average the scores
                final_metrics["starting_top-k_ensemble_test_accuracy"] = avg([sum(x) / len(x) for x in retained_scores_list])
                final_metrics["starting_top-k_ensemble_test_accuracy"] = avg([sum(x) / len(x) for x in retained_scores_list])
            elif eval_type == "perplexity":
                final_metrics["starting_top-k_ensemble_test_accuracy"] = None

            print("starting best validation utility: ", final_metrics["starting_best_validation_utility"])
            print("starting best particle on validation: ", starting_best_utility_index)
            print("starting best single test accuracy: ", final_metrics["starting_best_single_test_accuracy"])
            print("starting top-k ensemble test accuracy: ", final_metrics["starting_top-k_ensemble_test_accuracy"])

        final_metrics["ending_best_validation_utility"] = max(ending_utility)
        final_metrics["ending_best_particle_on_validation"] = ending_best_utility_index
        final_metrics["ending_best_single_test_accuracy"] = sum(ending_scores[ending_best_utility_index]) / len(ending_scores[ending_best_utility_index])
        if eval_type == "exact_match" or eval_type == "rm_default" or eval_type == "rm_concise" or eval_type == "rm_verbose" or eval_type == "human":
            temp_scores = ensemble_based_on_utility(ending_scores, ending_utility, top_k)
            final_metrics["ending_top-k_ensemble_test_accuracy"] = sum(temp_scores) / len(temp_scores)
        elif eval_type == "external_api":
            top_k_utility_bar = sorted(ending_utility, reverse=True)[:top_k]
            retained_scores_list = []
            for i in range(len(ending_scores)):
                if ending_utility[i] in top_k_utility_bar:
                    retained_scores_list.append(ending_scores[i])
            # average the scores
            final_metrics["ending_top-k_ensemble_test_accuracy"] = avg([sum(x) / len(x) for x in retained_scores_list])
            final_metrics["ending_top-k_ensemble_test_accuracy"] = avg([sum(x) / len(x) for x in retained_scores_list])
        elif eval_type == "perplexity":
            final_metrics["ending_top-k_ensemble_test_accuracy"] = None
        # final_metrics["ending_top-k_ensemble_test_accuracy"] = accuracy_score(golds, final_ending_preds)

        print("ending best validation utility: ", final_metrics["ending_best_validation_utility"])
        print("ending best particle on validation: ", final_metrics["ending_best_particle_on_validation"])
        print("ending best single test accuracy: ", final_metrics["ending_best_single_test_accuracy"])
        print("ending top-k ensemble test accuracy: ", final_metrics["ending_top-k_ensemble_test_accuracy"])

    # eval_type independent part of g_history analysis

    with open("search/" + name + "/utility_scratchpad.json", "r") as f:
        utility_scratchpad = json.load(f)
        g_history = utility_scratchpad["g_history"]

    # hoe many times did g_history improve
    g_history_improve_count = 0
    for i in range(len(g_history) - 1):
        if g_history[i] < g_history[i + 1]:
            g_history_improve_count += 1

    # when did g_history last change
    g_history_last_change_index = None
    for i in range(len(g_history) - 1):
        if g_history[i] < g_history[i + 1]:
            g_history_last_change_index = i + 1

    final_metrics["g_history_improve_count"] = g_history_improve_count
    final_metrics["g_history_last_change_index"] = g_history_last_change_index

    print("# g changes: ", g_history_improve_count)
    print("g last change index: ", g_history_last_change_index)

    return final_metrics
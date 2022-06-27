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

# pylint: skip-file
import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib
import numpy as np
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import argparse
from lego.common.utils import list2tuple, tuple2list

def glob_re(pattern, strings):
    return list(filter(re.compile(pattern).match, strings))

def cal_mrr(log):
    logits_topk = log['logits_topk']
    indices_topk = log['indices_topk']
    all_ans = log['all_ans']
    ranks = []

    for ans in all_ans:
        if ans not in indices_topk:
            ranks.append(np.inf)
        else:
            ranks.append(np.where(indices_topk==ans)[1][0]+1)

    ranks = np.sort(np.array(ranks))
    ranks = np.array(ranks)-np.arange(len(ranks))
    if not np.all(ranks > 0):
        print(ranks)
    mrr = np.mean(1/ranks)
    return mrr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Aggregating candidates',
    )
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--question_path', type=str, default=None)
    parser.add_argument('--candidate_path', type=str, default=None)

    args = parser.parse_args()

    file_names = glob_re(r'[0-9]*.pkl', os.listdir(args.candidate_path))
    file_names.sort(key=lambda x: int(x.split('.')[0]))
    all_logits_results = {}
    cnt = 0

    for name in tqdm(file_names):
        data = pickle.load(open(os.path.join(args.candidate_path, name), 'rb'))
        rew = data['reward']
        struct = data['structures']
        logs = data['logs']
        rew_in = []
        struct_in = []
        logs_in = []
        for i, log in enumerate(logs):
            in_flag = False
            for ans in log['all_ans']:
                if ans in log['indices_topk'][0]:
                    in_flag = True
                    break
            if in_flag:
                rew_in.append(rew[i])
                struct_in.append(struct[i])
                logs_in.append(log)

        idx = int(name.split('.')[0])
        all_logits_results[idx] = {'reward': rew_in, 'structures': struct_in, "logs": logs_in}
        cnt += 1

    ent2id = pickle.load(open(os.path.join(args.data_path, "ent2id.pkl"), "rb"))
    rel2id = pickle.load(open(os.path.join(args.data_path, "rel2id.pkl"), "rb"))
    id2ent = pickle.load(open(os.path.join(args.data_path, "id2ent.pkl"), "rb"))
    id2rel = pickle.load(open(os.path.join(args.data_path, "id2rel.pkl"), "rb"))

    datasets = pickle.load(open(os.path.join(args.question_path, "all_train_vanilla_datasets_with_query_structure.pkl"), 'rb'))


    for idx, data in enumerate(tqdm(datasets)):
        if idx in all_logits_results:
            for log in all_logits_results[idx]['logs']:
                log['mrr'] = cal_mrr(log)


    num_thres = 10
    rank = 20
    target_thresholds = np.arange(num_thres)[::-1]/num_thres/2
    question_query_dict = defaultdict(lambda: defaultdict(set))
    rt_pairs_to_save = [(i, 0.2) for i in [5, 10, 15, 20]] \
    + [(i, 0.15) for i in [5, 10, 15, 20]] \
    + [(i, 0.05) for i in [5, 10, 15, 20]]

    n_numt, n_numt_gt = np.zeros([rank, len(target_thresholds)]), np.zeros([rank, len(target_thresholds)])

    for idx, data in enumerate(tqdm(datasets)):
        gt_structure = data[0][5]
        gt_abs_structure = data[0][6]
        question = data[0][7]
        if idx in all_logits_results:
            reward = all_logits_results[idx]['reward']
            logs = all_logits_results[idx]['logs']
            structures = all_logits_results[idx]['structures']

            reward_mrr = [log['mrr'] for log in logs]
            for i, threshold in enumerate(target_thresholds):
                all_numt_args = np.argsort(-np.array(reward_mrr))[:rank+1]
                for tmp_rank in range(1, rank+1):
                    for stop_point in range(min(tmp_rank, len(all_numt_args))):
                        if reward_mrr[all_numt_args[stop_point]] < threshold:
                            break
                    numt_args = all_numt_args[:stop_point]

                    if (tmp_rank, threshold) in rt_pairs_to_save:
                        for arg in numt_args:
                            question_query_dict[(tmp_rank, threshold)][question].add(list2tuple(structures[arg][0]))


    for (tmp_rank, threshold) in rt_pairs_to_save:
        assert (tmp_rank, threshold) in question_query_dict
        print(tmp_rank, threshold, len(question_query_dict[(tmp_rank, threshold)]))
        save_str = "question_and_query_candidate_rank_{}_mrr_{}".format(tmp_rank, threshold)
        save_str += '.pkl'
        pickle.dump(question_query_dict[(tmp_rank, threshold)],
                open(os.path.join(args.candidate_path, save_str), 'wb'))


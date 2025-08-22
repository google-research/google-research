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
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import pandas as pd
from tqdm import tqdm

from agent import TableAgent, TableRAGAgent
from evaluate import evaluate
from utils.load_data import load_dataset


def solve(args):
    agent_args, data, sc_id = args
    if 'TableRAG' in agent_args['agent_type']:
        agent = TableRAGAgent(**agent_args)
    elif agent_args['agent_type'] in ['PyReAct', 'ReadSchema', 'RandSampling', 'TableSampling']:
        agent = TableAgent(**agent_args)
    else:
        raise NotImplementedError(f"Agent type {agent_args['agent_type']} not supported.")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return agent.run(data, sc_id=sc_id)


def main(
    dataset_path = 'data/tabfact/test_sub_nosynth.jsonl',
    model_name = 'gpt-3.5-turbo-0125',
    agent_type = 'PyReAct',
    retrieve_mode = 'embed',
    embed_model_name = 'text-embedding-3-large',
    log_dir = 'output/test',
    db_dir = 'db/',
    top_k = 5,
    sr = 0, # self-refine, deprecated
    sc = 1, # self-consistency
    max_encode_cell = 10000,
    stop_at = -1,
    resume_from = 0,
    load_exist = False,
    n_worker = 1,
    verbose = False,
):
    os.makedirs(os.path.join(log_dir, 'log'), exist_ok=True)

    # store the config
    task = [task_name for task_name in ['tabfact', 'wtq', 'arcade', 'bird'] if task_name in dataset_path][0]
    db_dir = os.path.join(db_dir, task + '_' + Path(dataset_path).stem)
    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as fp:
        json.dump({key: value for key, value in locals().items() if key != 'fp'}, fp, indent=4)

    dataset = load_dataset(task, dataset_path, stop_at)
    if stop_at < 0:
        stop_at = len(dataset)

    agent_args = {
        'model_name': model_name,
        'retrieve_mode': retrieve_mode,
        'embed_model_name': embed_model_name,
        'task': task,
        'agent_type': agent_type,
        'top_k': top_k,
        'sr': sr,
        'max_encode_cell': max_encode_cell,
        'log_dir': log_dir,
        'db_dir': db_dir,
        'load_exist': load_exist,
        'verbose': verbose
    }

    results = []
    if n_worker == 1:
        for data in tqdm(dataset[resume_from:stop_at]):
            for sc_id in tqdm(range(sc), position=1, leave=False):
                result = solve((agent_args, data, sc_id))
                results.append(result)
    else:
        with tqdm(total=(stop_at - resume_from) * sc) as pbar:
            with ProcessPoolExecutor(max_workers=n_worker) as executor:
                futures = [executor.submit(solve, (agent_args, data, sc_id)) for data in dataset[resume_from:stop_at] for sc_id in range(sc)]
                for future in as_completed(futures):
                    pbar.update(1)
                    results.append(future.result())

    acc = evaluate(task, results)
    print(f'Accuracy: {acc}')
    stats_keys = ['n_iter', 'init_prompt_token_count', 'total_token_count']
    stats_df = pd.DataFrame.from_records(results)[stats_keys]
    print(stats_df.describe().to_string())

    # store the result
    result_dict = stats_df.mean().to_dict()
    result_dict['accuracy'] = acc
    for key in ['model_name', 'retrieve_mode', 'embed_model_name', 'task', 'agent_type', 'top_k', 'max_encode_cell', 'sr']:
        result_dict[key] = agent_args[key]
    result_dict['sc'] = sc
    result_dict['data'] = Path(dataset_path).stem
    result_path = os.path.join(log_dir, 'result.json')
    with open(result_path, 'w') as fp:
        json.dump(result_dict, fp, indent=4)


if __name__ == '__main__':
    fire.Fire(main)

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
import fire
from tqdm import tqdm
from pathlib import Path

from agent import Retriever
from utils.load_data import load_dataset
from utils.utils import table_text_to_df

def main(dataset_path, max_encode_cell = 10000):
    task = [task_name for task_name in ['tabfact', 'wtq', 'arcade', 'bird'] if task_name in dataset_path][0]
    db_dir = os.path.join('db/', task + '_' + Path(dataset_path).stem)
    dataset = load_dataset(task, dataset_path)
    done_table_ids = set()
    retriever = Retriever(agent_type='TableRAG', embed_model_name='text-embedding-3-large', top_k=5, max_encode_cell=max_encode_cell, db_dir=db_dir)
    for data in (pbar := tqdm(dataset)):
        table_id = data['table_id']
        pbar.set_description(f'Building database {table_id}, max_encode_cell {max_encode_cell}')
        if table_id in done_table_ids:
            continue
        done_table_ids.add(table_id)
        df = table_text_to_df(data['table_text'])
        retriever.agent_type = 'TableRAG'
        retriever.init_db(table_id, df)
        retriever.agent_type = 'TableSampling'
        retriever.init_db(table_id, df)


if __name__ == '__main__':
    fire.Fire(main)
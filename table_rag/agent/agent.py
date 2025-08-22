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
from typing import Optional

from agent.model import Model
from agent.retriever import Retriever
from utils.execute import parse_code_from_string, python_repl_ast
from utils.utils import infer_dtype, get_df_info
from prompts import get_prompt

# global variables for python repl
import pandas as pd
import numpy as np
from datetime import datetime


class TableAgent:
    def __init__(
            self,
            model_name: str,
            retrieve_mode: str,
            embed_model_name: Optional[str] = None,
            task: str = 'tabfact',
            agent_type: str = 'PyReAct',
            top_k: int = 3,
            sr: int = 0,
            max_encode_cell: int = 10000,
            temperature: float = 0.8,
            top_p: float = 0.95,
            stop_tokens: Optional[list] = ['Observation:'],
            max_tokens: int = 128,
            max_depth: int = 5,
            load_exist: bool = False,
            log_dir: Optional[str] = None,
            db_dir: Optional[str] = None,
            verbose: bool = False,
    ):
        self.model = None
        self.model_name = model_name
        self.retrieve_mode = retrieve_mode
        self.embed_model_name = embed_model_name
        self.task = task
        self.agent_type = agent_type
        self.top_k = top_k
        self.sr = sr
        self.max_encode_cell = max_encode_cell
        self.max_depth = max_depth
        self.stop_tokens = stop_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.load_exist = load_exist
        self.log_dir = log_dir
        self.db_dir = db_dir
        self.verbose = verbose
        self.total_input_token_count = 0
        self.total_output_token_count = 0
        self.model = Model(self.model_name)
        self.retriever = Retriever(agent_type, retrieve_mode, embed_model_name, top_k=top_k, max_encode_cell=max_encode_cell, db_dir=db_dir, verbose=verbose)

    def is_terminal(self, text: str) -> bool:
        return 'final answer:' in text.lower()

    def query(self, prompt) -> str:
        input_token_count = self.model.get_token_count(prompt)
        if input_token_count > self.model.context_limit:
            return f'Prompt length -- {input_token_count} is too long, we cannot query the API.'
        self.total_input_token_count += input_token_count
        response_text = self.model.query(
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_tokens,
            max_tokens=self.max_tokens,
        )
        self.total_output_token_count += self.model.get_token_count(response_text)
        return response_text

    # Solver loop follows the ReAct framework: https://github.com/ysymyth/ReAct.
    def solver_loop(self, df: pd.DataFrame, prompt: str) -> str:
        if self.verbose:
            print(prompt, end='')

        memory = {}
        n_iter = self.max_depth
        solution = ''
        init_prompt = prompt

        for i in range(self.max_depth):
            solution += 'Thought: ' # always start with thoughts
            prompt = init_prompt + solution
            text = self.query(prompt).strip()
            solution += text

            if self.verbose:
                print('Thought: ' + text)

            # first check if it is terminal
            if self.is_terminal(text):
                n_iter = i + 1
                break

            if 'Action:' not in text:
                observation = 'Error: no Action provided.'
            else:
                # execute the code, we need to pass the dataframe, and pandas as pd, numpy as np to the locals
                code = parse_code_from_string(text.split('Action:')[-1].strip())
                observation, memory = python_repl_ast(code, custom_locals={'df': df}, custom_globals=globals(), memory=memory)
                if isinstance(observation, str) and self.model.get_token_count(observation) > self.model.context_limit:
                    observation = 'Observation is too long, we cannot query the API.'
                if isinstance(observation, str) and observation == '':
                    observation = 'success!'

            # if observation has multiple lines, we need to add new line at the beginning
            if '\n' in str(observation):
                observation = '\n' + str(observation)

            solution += f'\nObservation: {observation}\n'

            if self.verbose:
                print(f'Observation: {observation}')

        answer = text.split('Answer:')[-1].split('\n')[0].strip()
        return answer, n_iter, solution
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        return get_prompt(self.task, self.agent_type, prompt_type, **kwargs)

    def run(self, data:dict, sc_id: int = 0) -> dict:
        log_path = os.path.join(self.log_dir, 'log', f'{data["id"]}-{sc_id}.json')

        # Load the log file if it exists
        if os.path.exists(log_path) and self.load_exist:
            with open(log_path) as fp:
                result = json.load(fp)
            return result

        if self.verbose:
            print('=' * 25 + f' {data["id"]} ' + '=' * 25)

        # Read table
        table_caption = data.get('table_caption', '')
        query = data['statement'] if 'statement' in data else data['question']
        table_text = data['table_text']
        df = pd.DataFrame(table_text[1:], columns=table_text[0])

        if (self.agent_type == 'PyReAct' and 3 * df.shape[0] * df.shape[1] > self.model.context_limit) or (self.agent_type == 'RandSampling' and 3 * self.top_k * df.shape[1] > self.model.context_limit):
            prompt = ''
            answer = solution = 'Error: table is too large.'
            n_iter = init_prompt_token_count = 0
            if self.verbose:
                print('Error: table is too large.')
        else:
            df = infer_dtype(df)
            if self.agent_type == 'PyReAct':
                table_markdown = df.to_markdown()
            elif self.agent_type == 'ReadSchema':
                table_markdown = get_df_info(df)
            elif self.agent_type == 'RandSampling':
                if df.shape[0] > self.top_k:
                    sampled_table = df.sample(n=self.top_k).sort_index()
                else:
                    sampled_table = df
                table_markdown = sampled_table.to_markdown(index=False)
            elif self.agent_type == 'TableSampling':
                self.retriever.init_retriever(data['table_id'], df)
                sampled_table = self.retriever.sample_rows_and_columns(query=query)
                table_markdown = sampled_table.to_markdown(index=False)
            else:
                raise ValueError(f'Invalid agent type: {self.agent_type}')
            prompt = self.get_prompt('solve_table_prompt', table_caption=table_caption, query=query, table=table_markdown)
            init_prompt_token_count = self.model.get_token_count(prompt)
            answer, n_iter, solution = self.solver_loop(df, prompt)

        result = {
            'id': data['id'],
            'sc_id': sc_id,
            'table_caption': table_caption,
            'query': query,
            'solution': solution,
            'answer': answer,
            'label': data['label'],
            'n_iter': n_iter,
            'init_prompt_token_count': init_prompt_token_count,
            'total_token_count': self.total_input_token_count + self.total_output_token_count,
            'n_rows': df.shape[0],
            'n_cols': df.shape[1],
        }
        if 'orig_id' in data:
            result['orig_id'] = data['orig_id']

        with open(log_path, 'w') as fp:
            json.dump(result, fp, indent=4)
        with open(log_path.replace('.json', '.txt'), 'w') as fp:
            fp.write(prompt + solution)

        return result

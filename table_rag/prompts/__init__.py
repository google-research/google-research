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

from prompts import tabfact, wtq, arcade

def get_prompt_templates(task, agent_type):
    if task == 'tabfact' and 'TableRAG' in agent_type:
        return {
            'extract_column_prompt': tabfact.tablerag_extract_column_prompt,
            'extract_cell_prompt': tabfact.tablerag_extract_cell_prompt,
            'solve_table_prompt': tabfact.tablerag_solve_table_prompt,
        }
    elif task == 'tabfact' and agent_type in ['PyReAct', 'ReadSchema', 'RandSampling', 'TableSampling']:
        return {'solve_table_prompt': tabfact.pyreact_solve_table_prompt}
    elif task == 'wtq' and 'TableRAG' in agent_type:
        return {
            'extract_column_prompt': wtq.tablerag_extract_column_prompt,
            'extract_cell_prompt': wtq.tablerag_extract_cell_prompt,
            'solve_table_prompt': wtq.tablerag_solve_table_prompt,
        }
    elif task == 'wtq' and agent_type in ['PyReAct', 'ReadSchema', 'RandSampling', 'TableSampling']:
        return {'solve_table_prompt': wtq.pyreact_solve_table_prompt}
    elif task in ['arcade', 'bird'] and 'TableRAG' in agent_type:
        return {
            'extract_column_prompt': arcade.tablerag_extract_column_prompt,
            'extract_cell_prompt': arcade.tablerag_extract_cell_prompt,
            'solve_table_prompt': arcade.tablerag_solve_table_prompt,
        }
    elif task in ['arcade', 'bird'] and agent_type in ['PyReAct', 'ReadSchema', 'RandSampling', 'TableSampling']:
        return {'solve_table_prompt': arcade.pyreact_solve_table_prompt}
    else:
        raise NotImplementedError(f"Task {task} and agent type {agent_type} not supported.")


def get_prompt(task, agent_type, prompt_type, **kwargs):
    prompt_templates = get_prompt_templates(task, agent_type)
    return prompt_templates[prompt_type].format(**kwargs)
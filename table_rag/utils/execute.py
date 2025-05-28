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

import ast
import re
import pandas as pd
from contextlib import redirect_stdout
from io import StringIO


def parse_code_from_string(input_string):
    """
    Parse executable code from a string, handling various markdown-like code block formats.

    Parameters:
    input_string (str): The input string.

    Returns:
    str: The parsed code.
    """

    # Pattern to match code blocks wrapped in triple backticks, with optional language specification
    triple_backtick_pattern = r"```(\w*\s*)?(.*?)```"
    match = re.search(triple_backtick_pattern, input_string, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(2).strip()

    # Pattern to match code blocks wrapped in single backticks
    single_backtick_pattern = r"`(.*?)`"
    match = re.search(single_backtick_pattern, input_string, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    # Default return if no code block patterns are matched
    return input_string.strip()


def python_repl_ast(code, custom_globals=None, custom_locals=None, memory=None):
    """
    Run command with own globals/locals and returns anything printed.

    Parameters:
    code (str): The code to execute.
    custom_globals (dict): The globals to use.
    custom_locals (dict): The locals to use.
    memory (dict): The state/memory to retain between invocations.

    Returns:
    tuple: (str: The output of the code, dict: updated memory).
    """

    if memory is None:
        memory = {}

    if custom_globals is None:
        custom_globals = globals().copy()
    else:
        custom_globals = {**globals(), **custom_globals}

    if custom_locals is None:
        custom_locals = memory.copy()
    else:
        custom_locals = {**custom_locals, **memory}

    try:
        tree = ast.parse(code)
        module = ast.Module(tree.body[:-1], type_ignores=[])

        io_buffer1 = StringIO()
        # Redirect stdout to our buffer and attempt to evaluate the last line
        with redirect_stdout(io_buffer1):
            # Execute all lines except the last
            exec(ast.unparse(module), custom_globals, custom_locals)
        output1 = io_buffer1.getvalue()
        if output1 and not output1.endswith('\n'):
            output1 += '\n'

        # Prepare the last line
        module_end = ast.Module(tree.body[-1:], type_ignores=[])
        module_end_str = ast.unparse(module_end)

        # Remove print statements from the last line
        if module_end_str.strip().startswith('print('):
            module_end_str = module_end_str.strip()[6:-1]

        io_buffer2 = StringIO()

        # Redirect stdout to our buffer and attempt to evaluate the last line
        with redirect_stdout(io_buffer2):
            try:
                ret = eval(module_end_str, custom_globals, custom_locals)
                if ret is not None:
                    output = object_to_string(ret, module_end_str)
                else:
                    output = io_buffer2.getvalue()
            except Exception:
                # If evaluating fails, try executing it instead
                exec(module_end_str, custom_globals, custom_locals)
                output = io_buffer2.getvalue()

        # Update memory with new variable states
        memory.update(custom_locals)

        # Return any output captured during execution along with the updated memory
        return output1 + output, memory

    except Exception as e:
        return "{}: {}".format(type(e).__name__, str(e)), memory


def object_to_string(obj, command):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, pd.DataFrame):
        if len(obj) == 0:
            return 'Empty DataFrame'
    elif command == 'df.columns':
        obj = obj.tolist()
        if len(obj) > 20:
            return str(obj[:10])[:-1] + ', ..., ' + str(obj[-10:])[1:]
    return str(obj)

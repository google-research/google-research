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
import re
import glob
import json
import string
import unicodedata
from math import isnan, isinf
from abc import ABCMeta, abstractmethod
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional
from tqdm import tqdm

import fire
import pandas as pd

from utils.load_data import load_dataset


def evaluate(task, results, return_all = False):
    if task == 'tabfact':
        return evaluate_tabfact(results, return_all)
    else:
        return evaluate_qa(results, return_all)


########## TabFact ##########

def evaluate_tabfact_answer(pred, label):
    pred = pred.lower()
    if pred == 'true':
        pred = 'yes'
    elif pred == 'false':
        pred = 'no'
    if (pred == 'yes' and label == 1) or (pred == 'no' and label == 0):
        return True
    return False


def evaluate_tabfact(results, return_all=False):
    preds = defaultdict(Counter)
    labels = {}
    for result in results:
        qid = result['id']
        labels.setdefault(qid, result['label'])
        preds[qid].update([result['answer']])

    eval_results = {}
    for qid, label in labels.items():
        eval_results[qid] = evaluate_tabfact_answer(preds[qid].most_common(1)[0][0], label)
    acc = sum(eval_results.values()) / len(eval_results)
    if return_all:
        return acc, eval_results
    return acc


########## WTQ ##########

def evaluate_qa(results, return_all=False):
    preds = defaultdict(Counter)
    labels = {}
    for result in results:
        qid = result['id']
        labels.setdefault(qid, normalize_answer(result['label']))
        preds[qid].update([normalize_answer(result['answer'])])

    eval_results = {}
    for qid, label in labels.items():
        eval_results[qid] = (preds[qid].most_common(1)[0][0] == label)
    acc = sum(eval_results.values()) / len(eval_results)
    if return_all:
        return acc, eval_results
    return acc

########## WTQ Official Evaluator ##########

################ String Normalization ################

def normalize(x):
    if not isinstance(x, str):
        x = x.decode('utf8', errors='ignore')
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x


################ Value Types ################

class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.
        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):

    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        # return 'S' + str([self.normalized])
        return self.normalized

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):

    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        # return ('N(%f)' % self.amount) + str([self.normalized])
        return str(self.amount)

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.
        Return:
            the number (int or float) if successful; otherwise None.
        """
        if text.startswith('$'):
            text = text[1:].strip()
        try:
            return int(text)
        except:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except:
                return None


class DateValue(Value):

    def __init__(self, year, month, day, original_string=None):
        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx')
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        # return (('D(%d,%d,%d)' % (self._year, self._month, self._day))
        #         + str([self._normalized]))
        return '%d,%d,%d' % (self._year, self._month, self._day)

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.
        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


################ Value Instantiation ################

def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.
    Args:
        original_string (basestring): Original string
        corenlp_value (basestring): Optional value returned from CoreNLP
    Returns:
        Value
    """
    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    if not corenlp_value:
        corenlp_value = original_string
    # Number?
    amount = NumberValue.parse(corenlp_value)
    if amount is not None:
        return NumberValue(amount, original_string)
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    # String.
    return StringValue(original_string)


def normalize_answer(ans):
    """Convert a list of strings to a list of Values
    Args:
        ans (list[basestring])
    Returns:
        list[Value]
    """
    if isinstance(ans, str) and ans.startswith('[') and ans.endswith(']'):
        try:
            ans = eval(ans)
        except:
            ans = ans[1:-1]
    if isinstance(ans, list):
        ans = [str(a).lower() for a in ans]
    else:
        ans = str(ans)
        ans = ans.lower()
        if ' and ' in ans:
            ans = ans.replace(' and ', ', ')
        ans = [span.strip() for span in ans.split(', ')]
    assert isinstance(ans, (list, tuple, set))
    return '|'.join(sorted(list(set(str(to_value(x)) for x in ans))))

########## Main ##########

def main(result_dir, save_path = None, split_by_total_cell = 0):
    task = [task_name for task_name in ['tabfact', 'wtq', 'arcade', 'bird'] if task_name in result_dir][0]

    # load config
    config_path = os.path.join(result_dir, 'config.json')
    with open(config_path) as fp:
        config = json.load(fp)

    # load results
    results = []
    for result_path in tqdm(glob.glob(os.path.join(result_dir, 'log', '*.json'))):
        with open(result_path) as fp:
            results.append(json.load(fp))
    acc, eval_results = evaluate(task, results, return_all=True)
    print(f'Accuracy: {acc}')
    stats_keys = ['id', 'n_iter', 'init_prompt_token_count', 'total_token_count']
    stats_df = pd.DataFrame.from_records(results)[stats_keys]
    print(stats_df.describe().to_string())

    result_dict = stats_df[['n_iter', 'init_prompt_token_count', 'total_token_count']].mean().to_dict()
    result_dict['accuracy'] = acc
    for key in ['model_name', 'embed_model_name', 'task', 'agent_type', 'top_k', 'sc', 'max_encode_cell']:
        result_dict[key] = config[key]
    result_dict['data'] = Path(config['dataset_path']).stem

    # store the result
    if save_path is not None:
        with open(save_path, 'w') as fp:
            json.dump(result_dict, fp, indent=4)

    if split_by_total_cell > 0:
        dataset = load_dataset(task, config['dataset_path'])
        data_stats_df = pd.DataFrame.from_records([{'id': data['id'], 'n_row': len(data['table_text']) - 1, 'n_col': len(data['table_text'][0])}for data in dataset])
        stats_df = stats_df.merge(data_stats_df, on='id')
        stats_df['total_cell'] = stats_df['n_row'] * stats_df['n_col']
        stats_df['eval_result'] = stats_df['id'].map(eval_results).astype(int)

        print(stats_df[stats_df['total_cell'] < split_by_total_cell].describe().to_string())
        print(stats_df[stats_df['total_cell'] >= split_by_total_cell].describe().to_string())

        if save_path is not None:
            result_dict['data'] = Path(config['dataset_path']).stem + '-small'
            result_dict['accuracy'] = stats_df[stats_df['total_cell'] < split_by_total_cell]['eval_result'].mean()
            with open(save_path[:-5] + '-small.json', 'w') as fp:
                json.dump(result_dict, fp, indent=4)
            result_dict['data'] = Path(config['dataset_path']).stem + '-large'
            result_dict['accuracy'] = stats_df[stats_df['total_cell'] >= split_by_total_cell]['eval_result'].mean()
            with open(save_path[:-5] + '-large.json', 'w') as fp:
                json.dump(result_dict, fp, indent=4)


if __name__ == '__main__':
    fire.Fire(main)
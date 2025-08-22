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

import io
import csv
import json
import warnings
import unicodedata

import numpy as np
import pandas as pd

from utils.execute import parse_code_from_string


def read_json(text):
    res = parse_code_from_string(text)
    return json.loads(res)


def is_numeric(s):
    try:
        float(s)
    except:
        return False
    return True


def table_text_to_df(table_text):
    df = pd.DataFrame(table_text[1:], columns=table_text[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = infer_dtype(df)
    return df


def infer_dtype(df):
    """
    Attempt to convert columns in a DataFrame to a more appropriate data type.

    :param df: Input DataFrame
    :return: DataFrame with updated dtypes
    """

    for col in df.columns:
        try:
            # Try converting to numeric
            df[col] = pd.to_numeric(df[col], errors='ignore')

            # If the column type is still object (string) after trying numeric conversion, try datetime conversion
            if df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col], errors='raise')
        except:
            pass

    return df


def get_df_info(df):
    buf = io.StringIO()
    df.info(verbose=True, buf=buf)
    return buf.getvalue()


def to_partial_markdown(df, n_visible):
    df = df.astype('object')
    df = df.fillna(np.nan)
    if n_visible == -1:
        return df.to_markdown(index=False)
    if n_visible == 0:
        return ''
    skip_rows = n_visible < df.shape[0]
    skip_cols = n_visible < df.shape[1]
    n_visible //= 2

    if skip_cols:
        new_df = df.iloc[:,:n_visible]
        new_df.loc[:,'...'] = '...'
        new_df = pd.concat([new_df, df.iloc[:,-n_visible:]], axis=1)
    else:
        new_df = df

    if skip_rows:
        rows = new_df.to_markdown(index=False).split('\n')
        row_texts = rows[1].split('|')
        new_row_texts = ['']
        for text in row_texts[1:-1]:
            if text[0] == ':':
                new_text = ' ...' + ' ' * (len(text) - 4)
            else:
                new_text = ' ' * (len(text) - 4) + '... '
            new_row_texts.append(new_text)
        new_row_texts.append('')
        new_row = '|'.join(new_row_texts)
        output = '\n'.join(rows[:2 + n_visible] + [new_row] + rows[-n_visible:])
    else:
        output = new_df.to_markdown(index=False)
    return output


def markdown_to_df(markdown_string):
    """
    Parse a markdown table to a pandas dataframe.

    Parameters:
    markdown_string (str): The markdown table string.

    Returns:
    pd.DataFrame: The parsed markdown table as a pandas dataframe.
    """

    # Split the markdown string into lines
    lines = markdown_string.strip().split("\n")

    # strip leading/trailing '|'
    lines = [line.strip('|') for line in lines]

    # Check if the markdown string is empty or only contains the header and delimiter
    if len(lines) < 2:
        raise ValueError("Markdown string should contain at least a header, delimiter and one data row.")

    # Check if the markdown string contains the correct delimiter for a table
    if not set(lines[1].strip()) <= set(['-', '|', ' ', ':']):
        # means the second line is not a delimiter line
        # we do nothing
        pass
    # Remove the delimiter line
    else:
        del lines[1]

    # Replace '|' in the cells with ';'
    stripe_pos = [i for i, c in enumerate(lines[0]) if c == '|']
    lines = [lines[0]] + [line.replace('|', ';') for line in lines[1:]]
    for i in range(1, len(lines)):
        for j in stripe_pos:
            lines[i] = lines[i][:j] + '|' + lines[i][j+1:]

    # Join the lines back into a single string, and use StringIO to make it file-like
    markdown_file_like = io.StringIO("\n".join(lines))

    # Use pandas to read the "file", assuming the first row is the header and the separator is '|'
    df = pd.read_csv(markdown_file_like, sep='|', skipinitialspace=True, quoting=csv.QUOTE_NONE)

    # Strip whitespace from column names and values
    df.columns = df.columns.str.strip()

    # Remove index column
    df = df.drop(columns='Unnamed: 0')

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # normalize unicode characters
    df = df.map(lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = infer_dtype(df)

    return df
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

"""Utilities to load dataset statistics json files."""

import glob
import json
import os
import types
from typing import Any

open_file = open
file_exists = os.path.exists
remove_file = os.remove
glob_file = glob.glob


def check_if_chunked_and_merge(file_path):
  filenames = glob_file(f"{file_path}-chunk-*")
  if not filenames:
    return
  filenames.sort(key=lambda f: int(f.split("-")[-1]))

  with open_file(file_path, "wb") as merged_file:
    for filename in filenames:
      file = open_file(filename, "rb")
      merged_file.write(file.read())
      file.close()


def load_schema_json(file_path, dataset_name):
  file_path = os.path.join(file_path, dataset_name)
  file_path = file_path + ".schema.json"
  check_if_chunked_and_merge(file_path)

  assert file_exists(file_path), f"Could not find file ({file_path})"
  print("Using schema file: ", file_path)
  return load_json(file_path)


def load_column_statistics(
    file_path, dataset_name, namespace = True
):
  file_path = os.path.join(file_path, dataset_name)
  file_path = file_path + ".column_statistics.json"
  check_if_chunked_and_merge(file_path)

  assert file_exists(file_path), f"Could not find file ({file_path})"
  print("Using column statistics file: ", file_path)
  return load_json(file_path, namespace=namespace)


def load_string_statistics(
    file_path, dataset_name, namespace = True
):
  file_path = os.path.join(file_path, dataset_name)
  file_path = file_path + ".string_statistics.json"
  check_if_chunked_and_merge(file_path)

  assert file_exists(file_path), f"Could not find file ({file_path})"
  print("Using string statistics file: ", file_path)
  return load_json(file_path, namespace=namespace)


def load_json(path, namespace = True):
  with gfile.Open(path) as json_file:
    if namespace:
      json_obj = json.load(
          json_file, object_hook=lambda d: types.SimpleNamespace(**d)
      )
    else:
      json_obj = json.load(json_file)
  return json_obj


def load_schema_sql(dataset, sql_filename):
  sql_path = os.path.join(dataset, "schema_sql", sql_filename)
  assert os.path.exists(sql_path), f"Could not find schema.sql ({sql_path})"
  with open(sql_path, "r") as file:
    data = file.read().replace("\n", "")
  return data

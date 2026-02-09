# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Generte config files from raw data for webarena tasks."""

import os
import json


def main():
    with open("test.raw.json", "r") as f:
        raw = f.read()
    raw = raw.replace("__GITLAB__", os.environ.get("GITLAB"))
    raw = raw.replace("__REDDIT__", os.environ.get("REDDIT"))
    raw = raw.replace("__SHOPPING__", os.environ.get("SHOPPING"))
    raw = raw.replace("__SHOPPING_ADMIN__", os.environ.get("SHOPPING_ADMIN"))
    raw = raw.replace("__WIKIPEDIA__", os.environ.get("WIKIPEDIA"))
    raw = raw.replace("__MAP__", os.environ.get("MAP"))
    with open("test.json", "w") as f:
        f.write(raw)
    # split to multiple files
    data = json.loads(raw)
    for idx, item in enumerate(data):
        with open(f"{idx}.json", "w") as f:
            json.dump(item, f, indent=2)


if __name__ == "__main__":
    main()

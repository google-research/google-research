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

"""Gemini evaluate on hallucination."""

import argparse
from datetime import datetime  # pylint: disable=g-importing-member
import json
import os

from ckd_utils.eval_utils import eval_pope
from pytz import timezone  # pylint: disable=g-importing-member
from tqdm import tqdm  # pylint: disable=g-importing-member
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import Image


# update based on your project and location
vertexai.init(project="PROJECT", location="LOCATION")
model = GenerativeModel("gemini-pro-vision")


def eval_hallucination(args):  # pylint: disable=redefined-outer-name
  """Evaluates hallucination."""

  questions = [
      json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
  ]
  answers_file = os.path.expanduser(args.answers_file)
  os.makedirs(os.path.dirname(answers_file), exist_ok=True)
  ans_file = open(answers_file, "w")

  for line in tqdm(questions):
    idx = line["question_id"]
    image_file = line["image"]
    qs = line["text"]

    text_input = (
        "Question: "
        + qs
        + " Please answer this question with one word yes or no."
    )
    image_path = os.path.join(args.image_folder, image_file)

    image = Image.load_from_file(image_path)
    try:
      response = model.generate_content([image, text_input])
      output = response.text
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(e)
      output = "FAILED"

    pred = output
    ans_file.write(
        json.dumps({
            "question_id": idx,
            "prompt": text_input,
            "text": pred,
            "model_id": "gemini-pro-vision",
            "image": image_file,
            "metadata": {},
        })
        + "\n"
    )


if __name__ == "__main__":

  fmt = "%Y_%m_%d_%H_%M_%S"
  # EST5EDT, Asia/Calcutta
  job_id = str(datetime.now(timezone("PST8PDT")).strftime(fmt))

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--image-folder", type=str, default="cache/coco/images/val2014"
  )
  parser.add_argument(
      "--question-file",
      type=str,
      default="cache/POPE/aokvqa/aokvqa_pope_adversarial.json",
  )
  parser.add_argument(
      "--answers-file", type=str, default="OUTPUT/POPE/gemini.jsonl"
  )

  args = parser.parse_args()

  test_name = args.question_file.split("/")[-1][:-5]
  print(
      "#" * 20,
      test_name,
      "#" * 20,
  )
  args.answers_file = (
      args.answers_file[:-6]
      + "_"
      + test_name
      + "_"
      + job_id
      + args.answers_file[-6:]
  )
  print(f"answer file: {args.answers_file}")

  eval_hallucination(args)
  ans = eval_pope.calculate_pope_results(args)

  logfile = os.path.join("output/gemini_evaluate.txt")

  with open(os.path.join(logfile), "a") as f:
    f.write(json.dumps({test_name: ans}) + "\n")

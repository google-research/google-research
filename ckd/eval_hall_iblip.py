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

"""Training script."""

import argparse
from datetime import datetime  # pylint: disable=g-importing-member
import json
import os

from ckd_utils.eval_utils import eval_pope
from lavis.models import load_model
from lavis.processors.blip_processors import BlipImageEvalProcessor
from PIL import Image
from pytz import timezone  # pylint: disable=g-importing-member
import torch
from tqdm import tqdm  # pylint: disable=g-importing-member
from transformers import set_seed  # pylint: disable=g-importing-member


def eval_model(args):  # pylint: disable=redefined-outer-name
  """Evaluates model."""

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = load_model(args.arch, args.model_type, is_eval=True)

  if args.weight_path == "baseline":
    print("basline model evaluation")
  else:
    print(f"loading new weights from {args.weight_path}")
    state = torch.load(args.weight_path, map_location="cpu")["model"]
    msg = model.load_state_dict(state, strict=False)
    print("unexpected_keys\n", msg.unexpected_keys)
    print("missing_keys")
    for k in msg.missing_keys:  # sanity
      if not k.startswith("visual_encoder") and not k.startswith("llm_model"):
        print(k)

  model = model.to(device)

  vis_processor = BlipImageEvalProcessor(image_size=224)
  questions = [
      json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
  ]
  answers_file = os.path.expanduser(args.answers_file)
  os.makedirs(os.path.dirname(answers_file), exist_ok=True)
  ans_file = open(answers_file, "w")

  for line in tqdm(questions):
    # for line in questions:
    idx = line["question_id"]
    image_file = line["image"]
    qs = line["text"]
    prompt = (
        "Question: "
        + qs
        + " Please answer this question with one word Yes or No."
    )

    image = Image.open(os.path.join(args.image_folder, image_file)).convert(
        "RGB"
    )
    image = vis_processor(image).to(device)
    if len(image.shape) == 3:
      image = image.unsqueeze(0)

    # with model.maybe_autocast(dtype=torch.bfloat16):
    outputs = model.generate(
        {"prompt": prompt, "image": image},
        use_nucleus_sampling=True,
        num_beams=1,
        max_length=10,
        min_length=1,
        top_p=1,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    # print(outputs)

    pred = outputs[0]
    # print(pred)

    ans_file.write(
        json.dumps({
            "question_id": idx,
            "prompt": prompt,
            "text": pred,
            "model_id": args.arch + "_" + args.model_type,
            "image": image_file,
            "metadata": {},
        })
        + "\n"
    )
    ans_file.flush()
  ans_file.close()


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
      "--answers-file", type=str, default="OUTPUT/POPE/iblip_viccuna.jsonl"
  )

  # parser.add_argument("--num-chunks", type=int, default=1)
  # parser.add_argument("--chunk-idx", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)

  parser.add_argument("--arch", type=str, default="blip2_vicuna_instruct_ckd")
  parser.add_argument("--model_type", type=str, default="vicuna7b")

  parser.add_argument(
      "--weight_path",
      type=str,
      default="baseline",
  )

  args = parser.parse_args()
  set_seed(args.seed)

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

  eval_model(args)
  ans = eval_pope.calculate_pope_results(args)

  if args.weight_path == "baseline":  # baseline models
    tag = args.arch + "_" + args.model_type + "_baseline"
    logfile = os.path.join(f"output/{tag}_evaluate.txt")
  else:
    tag = args.weight_path.split("/")[-2]
    logfile = os.path.join(
        os.path.dirname(args.weight_path), f"{tag}_evaluate.txt"
    )

  with open(os.path.join(logfile), "a") as f:
    f.write(json.dumps({test_name: ans}) + "\n")

  print("results are saved in ", logfile)

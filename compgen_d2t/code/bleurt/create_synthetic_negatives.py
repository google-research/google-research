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

r"""Create synthetic negative data for BLEURT.

Example command:

python create_synthetic_negatives.py \
  --input_path=example.txt \
  --min_num_deletions=1 \
  --max_num_deletions=7 \
  --num_deletion_negatives=3 \
  --num_repetition_negatives=3 \
  --min_num_repetitions=1 \
  --max_num_repetitions=7 \
  --num_flip_negatives=2 \
  --num_random_negatives=1 \
  --num_digit_negatives=1 \
  --dev_frac=0.1 \
  --output_dir="" \
  --use_source_as_reference=True \
  --upsampling=True
"""
import argparse
import json
import os
import random

import synthetic_data_utils

parser = argparse.ArgumentParser()

parser.add_argument("--input_path", help="Input path.")

parser.add_argument(
    "--num_deletion_negatives", help="Number of deletion negatives.")

parser.add_argument(
    "--min_num_deletions", help="Minimum number of tokens can be deleted.")

parser.add_argument(
    "--max_num_deletions", help="Maximum number of tokens can be deleted.")

parser.add_argument(
    "--num_repetition_negatives", help="Number of repetition negatives.")

parser.add_argument("--min_num_repetitions", help="Minimum repetition length.")

parser.add_argument("--max_num_repetitions", help="Maximum repetition length.")

parser.add_argument("--num_flip_negatives", help="Number of flip negatives.")

parser.add_argument(
    "--num_random_negatives",
    help="Number of negatives by randomly matching source and target.")
parser.add_argument(
    "--num_digit_negatives",
    help="Number of negatives by changing the digits in the target.")

parser.add_argument("--output_dir", help="Output directory.")

parser.add_argument(
    "--dev_frac", help="Fraction of examples to be used for development set.")

parser.add_argument(
    "--use_source_as_reference",
    help="Whether or not to use source as reference. If set to false, the "
    "target will be used as reference.")

parser.add_argument("--do_lowercase", help="Whether to lowercase the inputs.")

parser.add_argument(
    "--upsampling",
    help="Whether to upsample positive examples to create a balanced dataset.")


def load_input_data(input_path, do_lowercase):
  """Load input data."""
  examples = []
  lc = 0
  with open(input_path, "r") as f:
    for line in f:
      lc += 1
      line_split = line.strip().split("\t")
      print(line_split)
      if len(line_split) != 2:
        raise Exception("Each line is expect to only have two columns.")
      source, target = line_split

      if do_lowercase:
        source, target = source.lower(), target.lower()
      examples.append((source, target))

  print("Number of examples processed: %d" % lc)
  return examples


def main(args):
  orig_examples = load_input_data(args.input_path, bool(args.do_lowercase))

  negative_sampler = synthetic_data_utils.NegativeSampler(
      num_deletion_negatives=int(args.num_deletion_negatives),
      min_num_deletions=int(args.min_num_deletions),
      max_num_deletions=int(args.max_num_deletions),
      num_repetition_negatives=int(args.num_repetition_negatives),
      min_num_repetitions=int(args.min_num_repetitions),
      max_num_repetitions=int(args.max_num_repetitions),
      num_flip_negatives=int(args.num_flip_negatives),
      num_random_negatives=int(args.num_random_negatives),
      num_digit_negatives=int(args.num_digit_negatives),
      use_source_as_reference=bool(args.use_source_as_reference))

  positive_examples, negative_examples = [], []
  for index, orig_example in enumerate(orig_examples):
    source, target = orig_example
    if args.use_source_as_reference:
      positive_examples.append((source, target))
    else:
      positive_examples.append((target, target))

    negative_examples.extend(negative_sampler.get_negatives(source, target))
    negative_examples.extend(
        negative_sampler.get_random_negatives(orig_examples, index))
  print(f"{len(orig_examples)} examples processed, {len(positive_examples)} / "
        f"{len(negative_examples)} positive/negative examples processed.")

  # Arbitrary decision to have twice as many negatives as positives.
  # Note the positives are just duplicated.
  desired_pos_size = len(negative_examples) // 2
  duplication_factor = desired_pos_size // len(positive_examples)
  if duplication_factor > 1 and bool(args.upsampling):
    duplicated_pos_examples = positive_examples * duplication_factor
  else:
    duplicated_pos_examples = positive_examples

  # pylint: disable=g-complex-comprehension
  negative_json_examples = [{
      "reference": x[0],
      "candidate": x[1],
      "score": 0.0
  } for x in negative_examples]

  # pylint: disable=g-complex-comprehension
  positive_json_examples = [{
      "reference": x[0],
      "candidate": x[1],
      "score": 1.0
  } for x in duplicated_pos_examples]

  print("Num positive examples: %d" % len(positive_json_examples))
  print("Num negative examples: %d" % len(negative_json_examples))

  all_json_examples = negative_json_examples + positive_json_examples
  random.shuffle(all_json_examples)

  num_dev_examples = int(len(all_json_examples) * float(args.dev_frac))

  dev_json_examples = all_json_examples[:num_dev_examples]
  train_json_examples = all_json_examples[num_dev_examples:]

  prefix = "source" if bool(args.use_source_as_reference) else "ref"

  train_output_path = os.path.join(args.output_dir,
                                   prefix + "_synthetic_train.jsonl")
  dev_output_path = os.path.join(args.output_dir,
                                 prefix + "_synthetic_dev.jsonl")

  # First 100 examples of training set for debugging purposes.
  debug_output_path = os.path.join(args.output_dir,
                                   prefix + "_synthetic_debug.jsonl")

  with open(train_output_path, "w") as f:
    for json_example in train_json_examples:
      f.write(json.dumps(json_example) + "\n")

  with open(dev_output_path, "w") as f:
    for json_example in dev_json_examples:
      f.write(json.dumps(json_example) + "\n")

  # Output debug examples which are first 100 examples of training set.
  with open(debug_output_path, "w") as f:
    for debug_index, json_example in enumerate(train_json_examples):
      if debug_index >= 100:
        break
      f.write(json.dumps(json_example) + "\n")


if __name__ == "__main__":
  cmd_args = parser.parse_args()
  main(cmd_args)

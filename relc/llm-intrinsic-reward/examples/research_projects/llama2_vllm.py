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

r"""VLLM Llama2 inference."""

import argparse
import json
import random

from tqdm import tqdm
from vllm import LLM, SamplingParams


def process_prompt(text, prompt_template=None):
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    if prompt_template is not None:
        text = prompt_template.format(text)

    return text


def main(args):
    with open(args.prompts_file, "r") as f:
        prompts = [json.loads(line) for line in f]

    if args.sample_size is not None:
        prompts_sampled = random.sample(
            prompts, min(args.sample_size, len(prompts)))
    else:
        prompts_sampled = prompts

    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=1,
        download_dir=args.download_dir,
        dtype="auto",
        gpu_memory_utilization=0.5,
    )
    print(f"- {args.model_name} loaded.")

    if args.prompt_file:
        with open(args.prompt_file, "r") as rf:
            prompt_template = rf.read()
        print(f"- Use prompt: {args.prompt_file}")
    else:
        prompt_template = None

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=args.num_returns
    )

    # Generate continuations
    continuations = []
    batch = []
    for s in tqdm(prompts_sampled):
        batch.append(process_prompt(s["prompt"]["text"], prompt_template))

        if len(batch) == args.batch_size:
            outputs = llm.generate(batch, sampling_params)

            for output in outputs:
                continuations.append([g.text for g in output.outputs])

            batch = []

    if len(batch) > 0:
        outputs = llm.generate(batch, sampling_params)
        for output in outputs:
            continuations.append([g.text for g in output.outputs])

    # Save generations
    assert len(continuations) == len(prompts_sampled)
    for gens, pmt in zip(continuations, prompts_sampled):
        pmt["generations"] = []
        for g in gens:
            pmt["generations"].append({"text": g})

    with open(args.output_file, "w") as wf:
        for item in prompts_sampled:
            wf.write(json.dumps(item) + "\n")

    print(f"Generation saved at: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model name or path."
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default="huggingface/meta-llama/Llama-2-7b-chat-hf",
        help="Model download directory"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="realtoxicityprompts-data/prompts_rescored.jsonl",
        help="A jsonl file."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to evalute.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="nontoxic_prompts-10k.jsonl",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_returns",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--prompt_file",
        default=None,
        type=str,
        help="Prompt file path"
    )

    args = parser.parse_args()
    main(args)

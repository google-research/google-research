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

r"""This file is for ppo training on summarization ask with rouge reward."""

import argparse
import evaluate
import torch
import nltk
import numpy as np
import re

from os.path import join
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead
from trl import PPOConfig
from trl import PPOTrainer
from trl import create_reference_model
from trl import set_seed
from trl.core import LengthSampler

from trl.extras.openai_scores import query_batch_span

tqdm.pandas()


# Below is an example function to build the dataset. In our case, we use the
# IMDB dataset from the `datasets` library. One should customize this function
# to train the model on its own dataset.
def build_tldr_dataset(
    tokenizer,
    dataset_name="CarperAI/openai_summarize_tldr"
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`,
    one should customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    ds = load_dataset(dataset_name, split="train[:20000]")

    def filter_fn(sample):
        return len(sample["prompt"]) <= 2000

    ds = ds.filter(filter_fn, batched=False)

    def tokenize(sample):
        sample["prompt"] = "Summarize: " + sample["prompt"]
        sample["input_ids"] = tokenizer.encode(
            sample["prompt"], truncation=True, max_length=500)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["label_ids"] = tokenizer.encode(
            sample["label"], truncation=True, max_length=128)
        sample["label"] = tokenizer.decode(sample["label_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def parse_intrinsic_rewards(
    spans,
    tokens,
    negative_reward=-1.0,
    positive_reward=1.0
):
    """Parse intrinsic rewards.

    Args:
        spans (`str`):
            The spans to parse.
        tokens (`str`):
            The tokens to parse.
        negative_reward (`float`):
            The negative reward value.
        positive_reward (`float`):
            The positive reward value.
    """
    results = []

    for span, token_list in zip(spans, tokens):
        if "None identified" in span:
            span = span.replace("None identified", "")

        pattern = re.compile(r"Span \d+: (.*?)( \(Label: .*?\))?(\n|$)")
        matches = pattern.findall(span)

        # Get the main content from each match and join them with newlines
        combined_span = "\n".join(match[0] for match in matches)

        # For each token in token_list, check if it's in the combined_span
        res = []
        for token in token_list:
            r = negative_reward if token in combined_span else positive_reward
            res.append(r)

        results.append(res)
    return results


def remove_prefix(input_str, prefix="Summarize: "):
    if input_str.startswith(prefix):
        return input_str[len(prefix):]
    return input_str


def remove_suffix(input_str, suffix="TL;DR: "):
    if input_str.endswith(suffix):
        return input_str[:-len(suffix)]
    return input_str


def process_doc(document):
    document = remove_prefix(document, prefix="Summarize: ")
    document = remove_suffix(document, suffix="TL;DR: ")
    document = document.replace("\n", " ")
    document = re.sub(r"\s+", " ", document)
    return document


def reward_to_score(r):
    score = min(round(r * 2, 1), 10)
    return score


def get_intrinsic_rewards(
    batch,
    query_tensors,
    response_tensors,
    prompt,
    config,
    tokenizer,
    rewards
):
    """Get intrinsic rewards by calling LLM.

    Args:
        batch (`dict`):
            The batch of data.
        query_tensors (`dict`):
            The query tensors.
        response_tensors (`dict`):
            The response tensors.
        prompt (`str`):
            The prompt to be used for prompting LLM.
        config (`dict`):
            The config to be used.
        tokenizer (`dict`):
            The tokenizer.
        rewards (`dict`):
            Extrinsic rewards.
    """
    # Select samples with an extrinsic reward below the threshold
    # for intrinsic reward calculation
    selected_query_indices, selected_queries = [], []
    for idx, r in enumerate(rewards):
        if r.item() <= args.intrinsic_reward_threshold:
            selected_query_indices.append(idx)
            llm_query = prompt.format(
                process_doc(batch["query"][idx]),
                batch["response"][idx].replace("\n", " "),
                reward_to_score(r.item()),
            )
            selected_queries.append(llm_query)

    # Calcualte intrinsic rewards
    selected_query_llm_responses = []
    if len(selected_queries) > 0:
        selected_query_llm_responses = query_batch_span(
            selected_queries, max_workers=2, max_tokens=100)

    llm_responses = ["" for _ in range(config.batch_size)]
    idx_ = 0
    for s_idx in selected_query_indices:
        llm_responses[s_idx] = selected_query_llm_responses[idx_]
        idx_ += 1

    # Parse LLM responses to get numerical rewards
    batch_tokens = []
    for i in range(config.batch_size):
        temp = []
        for t in torch.cat((query_tensors[i], response_tensors[i])):
            temp.append(tokenizer.decode(t))
        batch_tokens.append(temp)

    intrisic_rewards = parse_intrinsic_rewards(
        llm_responses,
        batch_tokens,
        negative_reward=args.negative_reward_value,
        positive_reward=args.positive_reward_value,
    )

    return intrisic_rewards


def main(args):
    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        log_with=args.log_with,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        tracker_project_name=args.tracker_project_name,
        use_score_scaling=args.use_score_scaling
    )

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by
    # default. We need to set it to eos_token. only for this model.
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_tldr_dataset(tokenizer)

    if args.bf16:
        # Now let's build the model, the reference model, and the tokenizer.
        # We first load the model in bfloat16 to save memory.
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name)

    # We create a reference model by sharing 20 layers
    ref_model = create_reference_model(
        model,
        num_shared_layers=args.num_shared_layers)

    # We make sure to use `Adam` optimizer on the model parameters
    # that require gradients.
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=config.learning_rate)

    # We then build the PPOTrainer, passing the model,
    # the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # ======================= BUILD REWARD MODEL =========================
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(preds, labels):
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(preds, labels)
        results = []
        for pred, label in zip(decoded_preds, decoded_labels):
            result = metric.compute(
                predictions=[pred], references=[label], use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            prediction_lens = []
            for pred in preds:
                prediction_lens.append(
                    np.count_nonzero(pred != tokenizer.pad_token_id)
                )
            result["gen_len"] = np.mean(prediction_lens)
            results.append(result)
        return results
    # =====================================================================

    # We then define the arguments to pass to the `generate` function.
    # These arguments are passed to the `generate` function of the PPOTrainer,
    # which is a wrapper around the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    output_min_length = args.output_min_length
    output_max_length = args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    with open(args.prompt_file, "r") as rf:
        prompt = rf.read()
    # ppo_trainer.accelerator.log({"prompt": prompt})

    step = 0
    for epoch in range(args.epochs):
        for _, batch in tqdm(
            enumerate(ppo_trainer.dataloader),
            total=len(ppo_trainer.dataloader),
            desc=f"Epoch {epoch+1}"
        ):
            query_tensors = batch["input_ids"]

            # Get response from the policy model
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True)

            # Compute sentiment score # noqa
            pipe_outputs = compute_metrics(batch["response"], batch["label"])
            rewards = [torch.tensor(output["rouge1"]) for output in pipe_outputs]
            assert len(rewards) == config.batch_size

            token_rewards = []
            for i in range(config.batch_size):
                tok_r = torch.zeros(
                    len(query_tensors[i]) + len(response_tensors[i]) - 1)
                tok_r[-1] = rewards[i]
                token_rewards.append(tok_r)

            if args.use_instric_reward:
                intrisic_rewards = get_intrinsic_rewards(
                    batch,
                    query_tensors,
                    response_tensors,
                    prompt,
                    config,
                    tokenizer,
                    rewards,
                )

                for tok_rewards, i_rewards in zip(token_rewards, intrisic_rewards):
                    i_rewards = i_rewards[1:]
                    assert len(tok_rewards) == len(i_rewards)
                    for tok_i in range(len(tok_rewards)):
                        tok_rewards[tok_i] += i_rewards[tok_i]

            # Run PPO step
            stats = ppo_trainer.step(
                query_tensors, response_tensors, token_rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            # Save model every 100 epochs
            if step % 100 == 0:
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.save_pretrained(
                        join(args.model_save_path, f"step_{step}/"))
            step += 1

    # save final model after training
    if ppo_trainer.accelerator.is_main_process:
        ppo_trainer.save_pretrained(args.model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="ybelkada/gpt-j-6b-sharded-bf16",
        type=str,
        help="the model name"
    )
    parser.add_argument(
        "--log_with",
        default=None,
        type=str,
        help="use 'wandb' to log with wandb"
    )
    parser.add_argument(
        "--learning_rate",
        default=(1.47e-5) * 2,
        type=float,
        help="the learning rate"
    )
    parser.add_argument(
        "--mini_batch_size",
        default=4,
        type=int,
        help="the PPO minibatch size"
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="the batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="the number of gradient accumulation steps"
    )
    parser.add_argument(
        "--model_save_path",
        default="./gpt-j-6B-detoxified-long-context-26-shl-1e4-final",
        type=str,
        help="the path to save the model"
    )
    parser.add_argument(
        "--ppo_epochs",
        default=100,
        type=int,
        help="Number of optimisation epochs per batch of samples"
    )
    parser.add_argument("--seed", default=0, type=int, help="the random seed")
    parser.add_argument(
        "--tracker_project_name",
        default="trl",
        type=str,
        help="wandb project name"
    )
    parser.add_argument(
        "--num_shared_layers",
        default=None,
        type=int,
        help="Number of shared layers between reference model and model"
    )
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument(
        "--output_min_length",
        default=10,
        type=int,
        help="Minimum length of generated responses"
    )
    parser.add_argument(
        "--output_max_length",
        default=20,
        type=int,
        help="Maximum length of generated responses"
    )
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Number of epochs"
    )
    parser.add_argument(
        "--prompt_file",
        default="./prompts/prompt_summ_3shot_rouge.txt",
        type=str,
        help="Prompt file path"
    )
    parser.add_argument(
        "--positive_reward_value",
        default=1.0,
        type=float,
        help="Positive intrinsic reward value"
    )
    parser.add_argument(
        "--negative_reward_value",
        default=-1.0,
        type=float,
        help="Negative intrinsic reward value"
    )
    parser.add_argument(
        "--use_instric_reward",
        action="store_true",
        help="whether to instrinsic rewards"
    )
    parser.add_argument(
        "--intrinsic_reward_threshold",
        default=1.0,
        type=float,
        help="Threshold to use intrinsic rewards."
    )
    parser.add_argument(
        "--use_score_scaling",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)

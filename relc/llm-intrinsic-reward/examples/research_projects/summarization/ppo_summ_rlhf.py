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

r"""This file is for ppo on summarization ask with preference reward."""

import argparse

from os.path import join
from typing import List
import re
import torch
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from copy import deepcopy

from trl import AutoModelForCausalLMWithValueHead
from trl import PPOConfig
from trl import PPOTrainer
from trl import create_reference_model
from trl import set_seed
from trl.core import LengthSampler

from trl.extras.summ_reward_model import GPTRewardModel
from trl.extras.openai_scores import query_batch_span
from trl.extras.str_checker import is_gibberish


tqdm.pandas()


def build_tldr_dataset(
    dataset_name="CarperAI/openai_summarize_tldr",
    max_prompt_len=500,
    max_reference_len=128,
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
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, split="train[:20000]")

    def filter_fn(sample):
        return len(sample["prompt"]) <= 2000

    ds = ds.filter(filter_fn, batched=False)

    def tokenize(sample):
        tmp = tokenizer.decode(
            tokenizer(
                sample["prompt"].split("TL;DR:")[0],
                truncation=True,
                max_length=max_prompt_len - 6,
                add_special_tokens=False,
            )["input_ids"],
            skip_special_tokens=True,
        ).strip()
        sample["prompt"] = tmp + "\nTL;DR:"
        sample["input_ids"] = tokenizer.encode(
            sample["prompt"],
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=False,
        )
        sample["query"] = tokenizer.decode(
            sample["input_ids"], skip_special_tokens=True).strip()
        sample["label_ids"] = tokenizer.encode(
            sample["label"], truncation=True, max_length=max_reference_len)
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
    positive_reward=1.0,
):
    """Parse intrinsic rewards.

    Args:
        spans: list of spans
        tokens: list of tokens
        negative_reward: negative reward value
        positive_reward: positive reward value
    """
    results = []

    for span, token_list in zip(spans, tokens):
        if span is None:
            results.append([0.0 for _ in token_list])
        else:
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


def reward_to_score(num):
    """convert score to reward.

    Args:
        num: score number
    """
    quantiles = [
        (-1.5, -1), (-1, -0.5), (-0.5, 0), (0, 0.5),
        (0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5),
        (2.5, 3), (3, 3.5)
    ]

    # Check if the number is smaller than the smallest value in quantiles
    if num < quantiles[0][0]:
        return 0
    # Check if the number is greater than the largest value in quantiles
    elif num > quantiles[-1][1]:
        return 10
    else:
        for idx, (low, high) in enumerate(quantiles):
            if low <= num <= high:
                return idx + 1

    return None


def get_intrinsic_rewards(
    batch,
    query_tensors,
    response_tensors,
    prompt,
    config,
    tokenizer,
    rewards
):
    """Get intrinsic rewards from the preference model.
    """
    # Select samples with an extrinsic reward below the threshold
    # for intrinsic reward calculation
    selected_query_indices, selected_queries = [], []
    for idx, r in enumerate(rewards):
        if r.item() <= args.intrinsic_reward_threshold:
            selected_query_indices.append(idx)
            llm_query = prompt.format(
                process_doc(batch["query"][idx]),
                batch["response"][idx].strip().replace("\n", " "),
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


def get_scores(
    rw_model,
    rw_tokenizer,
    rw_device,
    samples,
    batch_size = 4,
):
    """Get reward scores from the preference model."""
    scores_list = []
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i : i + batch_size]
        sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
        encodings_dict = rw_tokenizer(
            sub_samples,
            truncation=True,
            max_length=550,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings_dict["input_ids"].to(rw_device)
        attn_masks = encodings_dict["attention_mask"].to(rw_device)
        input_ids = input_ids.repeat(2, 1)
        attn_masks = attn_masks.repeat(2, 1)
        with torch.no_grad():
            sub_scores = rw_model(
                input_ids=input_ids, attention_mask=attn_masks)
        scores_list.append(sub_scores["chosen_end_scores"])
    scores = torch.cat(scores_list, dim=0)
    return scores


class Normalizer:
    """Normalize rewards for RL training."""
    def __init__(self):
        self.mean = 0
        self.variance = 0
        self.count = 0

    def _update_stats(self, data):
        """Update the mean and variance with the new data."""
        for number in data:
            self.count += 1
            delta = number - self.mean
            self.mean += delta / self.count
            delta2 = number - self.mean
            self.variance += delta * delta2

        if self.count > 1:
            self.variance /= (self.count - 1)
        else:
            self.variance = 0

    def _normalize(self, data):
        if self.variance == 0:
            raise ValueError("Variance is zero, cannot normalize data.")

        # Standardize data (z-score normalization)
        standardized_data = []
        for x in data:
            standardized_data.append((x - self.mean) / (self.variance ** 0.5))

        # Scale data to range [-1, 1]
        max_abs_val = max(abs(min(standardized_data)),
                          abs(max(standardized_data)))
        if max_abs_val == 0:
            raise ValueError("All numbers are the same, cannot normalize.")

        return [x / max_abs_val for x in standardized_data]

    def normalize(self, data):
        if len(data) == 0:
            raise ValueError("Input data list is empty.")

        # Update the mean and variance with the new data
        self._update_stats(data)

        # Normalize the data
        return self._normalize(data)


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
        use_score_scaling=args.use_score_scaling,
        remove_unused_columns=False,
        kl_penalty="abs",
    )

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by
    # default. We need to set it to eos_token. only for this model.
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_tldr_dataset()

    if args.bf16:
        # Now let's build the model, the reference model, and the tokenizer.
        # We first load the model in bfloat16 to save memory
        # using `transformers`.
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name)

    # We create a reference model by sharing 20 layers
    ref_model = create_reference_model(
        model, num_shared_layers=args.num_shared_layers)

    # make sure to use `Adam` optimizer on parameters that require gradients.
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=config.learning_rate)

    # We then build the PPOTrainer, passing the model, the reference model,
    # the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # ========================== BUILD REWARD MODEL ==========================
    print("Loading reward model...")
    rw_tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-j-6B", use_fast=False)
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_model = GPTRewardModel("CarperAI/openai_summarize_tldr_ppo")
    rw_model.load_state_dict(
        torch.load(args.reward_model_ckpt_path), strict=False)
    # rw_model = AutoModelForCausalLM.from_pretrained("gpt2")
    rw_model.half()
    rw_model.eval()
    rw_device = ppo_trainer.accelerator.device # set reward model device
    rw_model.to(rw_device)
    print(f"Reward model loaded on device: {rw_device}")

    def reward_fn(
        prompts,
        ref_summaries,
        generated_summaries,
        rw_batch_size=4,
    ):
        assert len(prompts) == len(ref_summaries) == len(generated_summaries)
        documents = list(prompts)

        samples, original_samples = [], []
        for text, ref_summ, gen_summ in zip(
            documents,
            ref_summaries,
            generated_summaries
        ):
            samples.append(text + gen_summ)
            original_samples.append(text + ref_summ)

        scores = get_scores(
            rw_model,
            rw_tokenizer,
            rw_device,
            samples,
            batch_size=rw_batch_size
        )
        return scores

    def reward_decorator(
        reward,
        prompt,
        ref_summary,
        pred_summary,
    ):
        reward_decorated = deepcopy(reward)
        if is_gibberish(pred_summary):
            reward_decorated = -3.0
        elif len(pred_summary.split()) < len(ref_summary.split()):
            len_diff = len(ref_summary.split()) - len(pred_summary.split())
            len_diff = max(0, len_diff - 5)
            reward_decorated -= len_diff * args.length_penalty
        return reward_decorated

    def intrinsic_reward_decorator(r, step):
        return r
    # =========================================================================

    # We then define the arguments to pass to the `generate` function.
    # These arguments are passed to the `generate` function of the PPOTrainer,
    # which is a wrapper around the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        # "min_new_tokens": args.output_min_length,
        # "max_new_tokens": args.output_max_length,
    }

    output_min_length = args.output_min_length
    output_max_length = args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    prompt = None
    if args.use_intrinsic_rewad:
        with open(args.prompt_file, "r") as rf:
                prompt = rf.read()

    normalizer = Normalizer()

    step = 0
    for _ in range(args.epochs):
        for _, batch in tqdm(enumerate(ppo_trainer.dataloader)):
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

            # compute rewards
            rewards = reward_fn(
                batch["prompt"],
                batch["label"],
                batch["response"],
                args.rw_batch_size
            )
            assert len(rewards) == config.batch_size

            rewards_normalized = normalizer.normalize(rewards.tolist())

            # process rewards
            rewards_processed = []
            for i in range(config.batch_size):
                # add length penalty here
                rewards_processed.append(
                    reward_decorator(
                        rewards_normalized[i],
                        batch["prompt"][i],
                        batch["label"][i],
                        batch["response"][i]
                    )
                )

            token_rewards = []
            for i in range(config.batch_size):
                tok_r = torch.zeros(
                    len(query_tensors[i]) + len(response_tensors[i]) - 1)
                tok_r[-1] = rewards_processed[i]
                token_rewards.append(tok_r)

            if args.use_intrinsic_rewad:
                intrisic_rewards = get_intrinsic_rewards(
                    batch,
                    query_tensors,
                    response_tensors,
                    prompt,
                    config,
                    tokenizer,
                    rewards
                )
                for tok_rewards, i_rewards, q_tensor in zip(
                    token_rewards,
                    intrisic_rewards,
                    query_tensors
                ):
                    i_rewards = i_rewards[1:]
                    assert len(tok_rewards) == len(i_rewards)
                    for tok_i in range(len(tok_rewards)):
                        if tok_i >= len(q_tensor):
                            tok_rewards[tok_i] += intrinsic_reward_decorator(
                                i_rewards[tok_i], step)

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
        "--mini_batch_size", default=4, type=int, help="the PPO minibatch size")
    parser.add_argument(
        "--batch_size", default=16, type=int, help="the batch size")
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
        default=4,
        type=int,
        help="Number of optimisation epochs per batch of samples"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="the random seed")
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
        default=20,
        type=int,
        help="Minimum length of generated responses"
    )
    parser.add_argument(
        "--output_max_length",
        default=40,
        type=int,
        help="Maximum length of generated responses"
    )
    parser.add_argument(
        "--total_max_length",
        default=550,
        type=int,
        help="Maximum length of post and summary combined."
    )
    parser.add_argument(
        "--epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument(
        "--prompt_file",
        default="./prompts/prompt_toxicity_3shot_v2.txt",
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
        "--use_intrinsic_rewad",
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
        "--reward_model_device",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--rw_batch_size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--use_score_scaling",
        action="store_true",
    )
    parser.add_argument(
        "--reward_model_ckpt_path",
        default="./reward_model/rm_checkpoint/pytorch_model.bin",
        type=str,
        help="Reward model checkpoint path"
    )
    parser.add_argument(
        "--length_penalty",
        default=0.0,
        type=float,
        help="Penalty when response becomes too short"
    )

    args = parser.parse_args()
    main(args)

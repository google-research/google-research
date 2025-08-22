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

r"""This file is for summarization task evaluation."""

import argparse
from typing import List
from datasets import load_dataset
import evaluate
import nltk
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from trl import PPOConfig
from trl import PPOTrainer
from trl import set_seed
from trl.extras.summ_reward_model import GPTRewardModel


tqdm.pandas()


def build_tldr_dataset(
    tokenizer,
    dataset_name="CarperAI/openai_summarize_tldr",
    num_samples=None,
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
    if num_samples is not None:
        ds = load_dataset(dataset_name, split=f"test[:{num_samples}]")
    else:
        ds = load_dataset(dataset_name, split="test")

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


def calculate_mean(list_of_dicts):
    attribute_sums = {}
    for data_dict in list_of_dicts:
        for attribute, score in data_dict.items():
            attribute_sums[attribute] = attribute_sums.get(attribute, 0) + score

    attribute_means = {}
    for attr, total in attribute_sums.items():
        attribute_means[attr] = total / len(list_of_dicts)
    return attribute_means


def main(args):
    config = PPOConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        seed=args.seed,
        remove_unused_columns=False,
    )

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_tldr_dataset(
        tokenizer,
        dataset_name="CarperAI/openai_summarize_tldr",
        num_samples=args.num_samples_to_eval,
    )
    print("Total number of samples to evaluate: {}".format(len(dataset)))

    print(f"Loading policy mode from: {config.model_name}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    model.half()
    model.eval()
    print("Policy model loaded.")

    # We then build the PPOTrainer, passing the model, the reference model,
    # the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=None,
    )

    # ========================== BUILD REWARD MODEL ==========================
    rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    print("Building reward model from config file...")
    rw_model = GPTRewardModel("CarperAI/openai_summarize_tldr_ppo")
    print("Loading reward model checkpoint...")
    rw_model.load_state_dict(
        torch.load(args.reward_model_ckpt_path), strict=False)
    rw_model.half()
    rw_model.eval()
    rw_device = torch.device("cuda:{}".format(args.reward_model_device))
    rw_model.to(rw_device)
    print("Reward model loaded.")

    def get_scores(samples, batch_size=4):
        scores_list = []
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            stok, etok = "<|startoftext|>", "<|endoftext|>"
            sub_samples = [stok + chosen + etok for chosen in sub_samples]
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
                    input_ids=input_ids,
                    attention_mask=attn_masks,
                )
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)
        return scores

    def reward_fn(
        prompts,
        ref_summaries,
        generated_summaries,
        rw_batch_size=8,
    ):
        assert len(prompts) == len(ref_summaries) == len(generated_summaries)
        documents = [text.split("Summarize: ")[1] for text in prompts]

        samples = []
        for text, gen_summ in zip(documents, generated_summaries):
            samples.append(text + gen_summ)

        scores = get_scores(samples, batch_size=rw_batch_size)
        return scores
    # ========================================================================

    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(preds, labels):
        decoded_preds, decoded_labels = postprocess_text(preds, labels)
        results = []
        for pred, label in zip(decoded_preds, decoded_labels):
            result = metric.compute(
                predictions=[pred], references=[label], use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            results.append(result)
        return results

    # We then define the arguments to pass to the `generate` function.
    # These arguments are passed to the `generate` function of the PPOTrainer,
    # which is a wrapper around the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.output_max_length
    }

    scores, rouge_scores = [], []
    post_list, ref_list, preds_list = [], [], []
    for _, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response from the policy model
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            # length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(
            response_tensors,
            skip_special_tokens=True,
        )

        # compute preference rewards
        rewards = reward_fn(
            batch["prompt"],
            batch["label"],
            batch["response"],
            args.rw_batch_size
        )
        assert len(rewards) == config.batch_size
        scores.extend(rewards.tolist())

        # compute ROUGE reward
        rouge_outputs = compute_metrics(batch["response"], batch["label"])
        rouge_scores.extend(rouge_outputs)

        post_list.extend(batch["prompt"])
        ref_list.extend(batch["label"])
        preds_list.extend(batch["response"])

    print(f"Model Name: {args.model_name}")
    print(f"Total Samples: {len(scores)}")
    print(f"Mean Preference Score: {sum(scores) / len(scores)}")

    mean_scores = calculate_mean(rouge_scores)
    for attribute, mean in mean_scores.items():
        print(f"Mean {attribute}: {mean}")

    # save scores
    df = pd.DataFrame.from_dict(
        {
            "pred": preds_list,
            "truth": ref_list,
            "post": post_list,
            "pref_score": scores,
            "rouge1": [r["rouge1"] for r in rouge_scores],
            "rouge2": [r["rouge2"] for r in rouge_scores],
            "rougeL": [r["rougeL"] for r in rouge_scores],
            "rougeLsum": [r["rougeLsum"] for r in rouge_scores],
        }
    )
    df.to_csv(args.save_path, index=False)
    print(f"Evaluation results saved at: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="ybelkada/gpt-j-6b-sharded-bf16",
        type=str,
        help="the model name"
    )
    parser.add_argument(
        "--save_path",
        default="scores.csv",
        type=str,
        help="results save path"
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="the batch size"
    )
    parser.add_argument(
        "--rw_batch_size",
        default=4,
        type=int,
        help="the batch size for reward model"
    )
    parser.add_argument("--seed", default=0, type=int, help="the random seed")
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
        "--reward_model_device",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--reward_model_ckpt_path",
        default="./reward_model/rm_checkpoint/pytorch_model.bin",
        type=str,
        help="Reward model checkpoint path"
    )
    parser.add_argument(
        "--num_samples_to_eval",
        default=None,
        type=int,
    )

    args = parser.parse_args()
    main(args)

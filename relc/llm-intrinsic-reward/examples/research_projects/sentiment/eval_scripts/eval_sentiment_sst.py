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

#!/usr/bin/env python3

import os
import json
import time
import torch
import random
import logging
import argparse
import numpy as np

from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def generate_continuation(model, tokenizer, device, prompts, max_length=40, num_returns=10):
    prompt_encoded = tokenizer(prompts, padding='longest', add_special_tokens=False, return_tensors="pt")
    prompt_encoded = prompt_encoded.to(device)

    prompt_len = prompt_encoded['input_ids'].shape[1]
    output_tensors = model.generate(
        prompt_encoded['input_ids'],
        attention_mask=prompt_encoded['attention_mask'],
        max_length=prompt_len + max_length,
        temperature=1.0,
        repetition_penalty=1.0,
        do_sample=True,
        top_p=0.9,
        # top_k=1000,
        # num_beams=15,
        num_return_sequences=num_returns,
    )

    outputs = tokenizer.batch_decode(
        output_tensors[:, prompt_len:],
        clean_up_tokenization_spaces=True,
        skip_special_tokens=True
    )

    return outputs


def sentiment_classification(inputs, tokenizer, model, device=None):
    inputs_tokenized = tokenizer(inputs, padding='longest', return_tensors="pt")
    with torch.no_grad():
        inputs_tokenized = inputs_tokenized.to(device) if device is not None else inputs_tokenized
        logits = model(**inputs_tokenized).logits.detach()

    probs = torch.softmax(logits, dim=-1)
    predicted_class_ids = logits.argmax(dim=1)
    return [model.config.id2label[i] for i in predicted_class_ids.tolist()], probs


def main():
    # Configure the logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path", type=str, default="gpt2-xl")
    parser.add_argument("prompt_path", type=str, help="Prompts for evaluation.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed."
    )
    parser.add_argument(
        "--num_return",
        type=int,
        default=25,
        help="Number of returns."
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Sentiment classifier path."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/outputs.jsonl",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Read dataset
    prompts = []
    with open(args.prompt_path, 'r') as f:
        for line in f:
            prompts.append(json.loads(line))
    logger.info(f"Read {len(prompts)} prompts for evaluation.")

    # Load the model to be evaluated
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.to(device)
    model.eval()

    logger.info(f"Model {args.model_name_or_path} loaded.")

    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(args.classifier_path)
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(args.classifier_path)
    distilbert_model.to(device)
    distilbert_model.eval()

    logger.info("sentiment classifier loaded.")

    generations = []
    # eval_dataset = random.sample(prompts, min(len(prompts), 1000))
    for s in tqdm(prompts):
        outputs = generate_continuation(
            model,
            tokenizer,
            device,
            [s["prompt"]["text"]],
            max_length=20,
            num_returns=args.num_return
        )
        generations.append(outputs)

    assert len(prompts) == len(generations)
    for p, gens in zip(prompts, generations):
        p['generations'] = []
        for g in gens:
            p['generations'].append({'text': g})

    with open(args.output_file, 'w') as wf:
        for item in prompts:
            wf.write(json.dumps(item) + "\n")

    logger.info(f"Generation saved at: {args.output_file}")

    INDEX = 1  # 0: probability of being negative
    preds = []
    for o in tqdm(generations):
        _, probs = sentiment_classification(o, distilbert_tokenizer, distilbert_model, device=device)
        preds.append(probs[:, INDEX].tolist())

    prob = 0.
    for gens_p in preds:
        assert len(gens_p) == args.num_return
        count = 0
        for p in gens_p:
            if p > 0.5:
                count += 1
        prob += count / len(gens_p)

    logger.info(f"Total samples: {len(preds)}")
    logger.info(f"Number of predictions: {len(preds[0])}")
    logger.info("Positive Probability: {:.4f}".format(100 * prob / len(preds)))
    logger.info("Negative Probability: {:.4f}".format(100 * (1 - prob / len(preds))))


if __name__ == "__main__":
    main()

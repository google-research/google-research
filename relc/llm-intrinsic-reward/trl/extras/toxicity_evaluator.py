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

r"""Use API or classifier for toxicity evaluation."""
import torch

from multiprocessing import Pool
from googleapiclient import discovery
from googleapiclient.errors import HttpError

from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
)


class RobertaHateSpeechDetector:
    def __init__(
        self,
        model_id="facebook/roberta-hate-speech-dynabench-r4-target",
        device=None
    ):
        self.model_id = model_id
        self.device = device

        self._build()

    def _build(self):
        self.toxicity_tokenizer = RobertaTokenizer.from_pretrained(self.model_id)
        # We load the toxicity model in fp16 to save memory.
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_id, torch_dtype=torch.float16).to(self.device)

    def evaluate(self, texts):
        toxicity_inputs = self.toxicity_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        logits = self.model(**toxicity_inputs).logits.float()
        toxicity_labels = (logits[:, 0]).tolist()

        return [torch.tensor(output) for output in toxicity_labels]


class PerspectiveAPI:
    def __init__(self, api_key=None, num_thread=5):
        self.api_key = api_key
        self.num_thread = num_thread
        self.ATTRIBUTES = {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'PROFANITY': {},
            'THREAT': {},
            'SEXUALLY_EXPLICIT': {},  # Experimental attributes
            'FLIRTATION': {},  # Experimental attributes
        }

        self._build()

    def _build(self):
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def get_perspective_scores(self, sentence):
        analyze_request = {
            'comment': {'text': sentence},
            'requestedAttributes': self.ATTRIBUTES
        }
        response = self.client.comments().analyze(body=analyze_request).execute()
        return {k: v['summaryScore']['value'] for k, v in response['attributeScores'].items()}["TOXICITY"]

    def fn(self, text):
        try:
            score = self.get_perspective_scores(text)
        except Exception as err:
            print(err)
            score = None
        return score

    def evaluate(self, texts):
        with Pool(self.num_thread) as pool:
            scores = []
            for r in pool.imap(self.fn, texts):
                scores.append(r)

        for i in range(len(scores)):
            if scores[i] is None:
                try:
                    scores[i] = self.get_perspective_scores(texts[i])
                except Exception as err:
                    scores[i] = 0.5

        return [torch.tensor(1.0 - s) for s in scores]

# Towards Learning a Universal Non-Semantic Representation of Speech

Papers using this code:

1. [TRILLsson: Distilled Universal Paralinguistic Speech Representations](https://arxiv.org/abs/2203.00236)
1. ICASSP 2022: [Universal Paralinguistic Speech Representations Using Self-Supervised Conformers](https://arxiv.org/abs/2110.04621)
1. [BigSSL: Exploring the Frontier of Large-Scale Semi-Supervised Learning for Automatic Speech Recognition](https://arxiv.org/abs/2109.13226)
1. Interspeech 2021: [FRILL: A Non-Semantic Speech Embedding for Mobile Devices](https://arxiv.org/abs/2011.04609)
1. Interspeech 2021: [Comparing Supervised Models And Learned Speech Representations For Classifying Intelligibility Of Disordered Speech On Selected Phrases](https://arxiv.org/abs/2107.03985)
1. Interspeech 2020: [Towards Learning a Universal Non-Semantic Representation of Speech](https://arxiv.org/abs/2002.12764)


This paper and code repository describe a benchmark for comparing speech representations,
and the evaluation code to run it. It also contains a description of our baseline
best representation, TRILL.

## Things you can do

1. Reproduce the results from our [paper](https://arxiv.org/abs/2002.12764)
1. Compute performance of a new embedding on the [Non-Semantic Speech
   Benchmark (NOSS)](https://www.tensorflow.org/datasets/catalog/overview#audio)
1. Run our embedding [TRILL](https://aihub.cloud.google.com/s?q=nonsemantic-speech-benchmark),
   or any of the other embedding networks on a new dataset.

## Citation
To use this benchmark or embeddings, please cite as follows:

```
@inproceedings{trill,
  author={Joel Shor and Aren Jansen and Ronnie Maor and Oran Lang and Omry Tuval and Félix de Chaumont Quitry and Marco Tagliasacchi and Ira Shavitt and Dotan Emanuel and Yinnon Haviv},
  title={Towards Learning a Universal Non-Semantic Representation of Speech},
  year=2020,
  booktitle={Interspeech},
  pages={140--144},
  doi={10.21437/Interspeech.2020-1242}
}
```

#### For questions reach out to

Joel Shor ([joelshor@google.com](mailto:joelshor@google.com))

Oran Lang ([oranl@google.com](mailto:oranl@google.com))

## Overview

<img src="https://github.com/google-research/google-research/raw/master/non_semantic_speech_benchmark/images/data_flowchart.png" alt="Data flowchart" width="400">

<img src="https://github.com/google-research/google-research/raw/master/non_semantic_speech_benchmark/images/embedding_flowchart.png" alt="Embedding flowchart" width="400">

<img src="https://github.com/google-research/google-research/raw/master/non_semantic_speech_benchmark/images/eval_model_flowchart.png" alt="Eval model flowchart" width="400">


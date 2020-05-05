# Towards Learning a Universal Non-Semantic Representation of Speech

Paper: [Towards Learning a Universal Non-Semantic Representation of Speech](https://arxiv.org/abs/2002.12764)

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
@article{shor2020,
    title={Towards Learning a Universal Non-Semantic Representation of Speech},
    author={Joel Shor and Aren Jansen and Ronnie Maor and Oran Lang and Omry Tuval and Felix de Chaumont Quitry and Marco Tagliasacchi and Ira Shavitt and Dotan Emanuel and Yinnon Haviv},
    year={2020},
    journal = {ArXiv e-prints},
    eprint={2002.12764},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url = {https://arxiv.org/abs/2002.12764}
}
```

#### For questions reach out to

Joel Shor ([joelshor@google.com](mailto:joelshor@google.com))

Oran Lang ([oranl@google.com](mailto:oranl@google.com))

## Overview

<img src="https://github.com/google-research/google-research/raw/master/non_semantic_speech_benchmark/images/data_flowchart.png" alt="Data flowchart" width="400">

<img src="https://github.com/google-research/google-research/raw/master/non_semantic_speech_benchmark/images/embedding_flowchart.png" alt="Embedding flowchart" width="400">

<img src="https://github.com/google-research/google-research/raw/master/non_semantic_speech_benchmark/images/eval_model_flowchart.png" alt="Eval model flowchart" width="400">


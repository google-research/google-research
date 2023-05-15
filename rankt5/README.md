# RankT5

RankT5 fine-tunes the sequence-to-sequence T5 model for ranking problems.
Specifically, RankT5 allows the model to output ranking scores for each query-document pair, and fine-tunes the model with ranking losses to optimize ranking performance.
Our experiments show that the proposed models with ranking losses can achieve substantial ranking performance gains on different public text ranking data sets. Moreover, when fine-tuned with listwise ranking losses, the ranking model has better zero-shot ranking performance on out-of-domain data sets compared to the model fine-tuned with classification losses.

Refer to our SIGIR 2023 paper ([preprint](https://arxiv.org/abs/2210.10634)) for more details.

## Checkpoints
The following table contains fine-tuned RankT5 checkpoints.
We release 3 models (Base, Large and 3B) using the RankT5 Encoder-Decoder architecture, fine-tuned with the listwise softmax cross entropy loss on the MS MARCO data set. All models are fine-tuned for 50,000 steps.

| Model Structure |    Loss     |    Size   |  Step  | Fine-tuning Data  | Checkpoint  |
|:---------------:|:-----------:|:---------:|:------:|:--------------:|:-----------:|
| Encoder-Decoder |   Softmax   | Base      | 1049900|    MS MARCO    |  [encdec-base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/rankt5/base)    |
| Encoder-Decoder |   Softmax   | Large     | 1050700|    MS MARCO    |  [encdec-large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/rankt5/large)   |
| Encoder-Decoder |   Softmax   | 3B        | 1050000|    MS MARCO    |  [encdec-3B](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/rankt5/3B)  |


## Citation
Please cite our work by copying the following bibtex:

```
@inproceedings{RankT5,
  title={{RankT5}: Fine-Tuning {T5} for Text Ranking with Ranking Losses},
  author={Zhuang, Honglei and Qin, Zhen and Jagerman, Rolf and Hui, Kai and Ma, Ji and Lu, Jing and Ni, Jianmo and Wang, Xuanhui and Bendersky, Michael},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2023}
}
```
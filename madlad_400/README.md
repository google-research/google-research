# MADLAD-400: A Multilingual And Document-Level Large Audited Dataset

This repository contains the checkpoints and vocabularies from [MADLAD-400: A Multilingual And Document-Level Large Audited Dataset](https://arxiv.org/abs/2309.04662).

## Checkpoints

| Model          | Checkpoint                |
|-------------------------------------------------------------|-------------------------------------------------------------------------------------|
| 8B parameter LM | [link](https://console.cloud.google.com/storage/browser/madlad-400-checkpoints/checkpoints/8b-lm) |
| 3B parameter MT model | [link](https://console.cloud.google.com/storage/browser/madlad-400-checkpoints/checkpoints/3b-mt) |
| 7.2B parameter MT model | [link](https://console.cloud.google.com/storage/browser/madlad-400-checkpoints/checkpoints/7b-mt) |
| 7.2B parameter MT model (finetuned on backtranslated data) | [link](https://console.cloud.google.com/storage/browser/madlad-400-checkpoints/checkpoints/7b-mt-bt) |
| 10.7B parameter MT model | [link](https://console.cloud.google.com/storage/browser/madlad-400-checkpoints/checkpoints/10b-mt) |

## Vocabulary

The vocabularies used to train the models listed above are [here](https://console.cloud.google.com/storage/browser/madlad-400-checkpoints/vocabulary/256k_vocab).

## Example usage

We provide a simple [colab example](./inference_example.ipynb) showcasing how to use the released checkpoints for translation.

## Contact

Please reach out to {snehakudugunta, icaswell}꩜google.com for any questions or observed issues. Issues will be listed on this page to aid future users. For questions about the canaries, reach out to cchoquette@google.com.





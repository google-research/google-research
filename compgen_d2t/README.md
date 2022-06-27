# Compositional Generalization for Data-to-Text Generation

This repository releases code and data for `Improving Compositional Generalization with Self-Training for
Data-to-Text Generation` our paper accepted at ACL 2022.

## Data
We released our fewshot weather data, please find more detailed descriptions under `data/`.

## Code

To prepare finetuning data for BLEURT (change input file to yours), run:
```
sh run.sh
```


To finetune BLEURT, follow [these instructions](https://github.com/google-research/bleurt).

## Cite

```
@inproceedings{Mehta2022compgen,
  title = {Improving Compositional Generalization with Self-Training for Data-to-Text Generation},
  author = {Sanket Vaibhav Mehta, Jinfeng Rao, Yi Tay, Mihir Kale, Ankur Parikh, Hongtao Zhong, Emma Strubell},
  year = {2022},
  booktitle = {Proceedings of ACL}
}
```

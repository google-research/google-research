# StructFormer

This repository contains the code used for masked language model and unsupervised parsing experiments in 
[StructFormer: Joint Unsupervised Induction of Dependency and Constituency Structure from Masked Language Modeling](https://arxiv.org/abs/2012.00857) paper.
If you use this code or our results in your research, we'd appreciate if you cite our paper as following:

```
@misc{shen2020structformer,
      title={StructFormer: Joint Unsupervised Induction of Dependency and Constituency Structure from Masked Language Modeling}, 
      author={Yikang Shen and Yi Tay and Che Zheng and Dara Bahri and Donald Metzler and Aaron Courville},
      year={2020},
      eprint={2012.00857},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Software Requirements
Python 3.6, NLTK and PyTorch 1.5.1 are required for the current codebase.

## Steps

1. Install PyTorch and NLTK

2. Download PTB data. Note that the two tasks, i.e., language modeling and unsupervised parsing share the same model strucutre but require different formats of the PTB data. For language modeling we need the standard 10,000 word  [Penn Treebank corpus](https://github.com/pytorch/examples/tree/75e435f98ab7aaa7f82632d4e633e8e03070e8ac/word_language_model/data/penn) data and for parsing we need [Penn Treebank Parsed](https://catalog.ldc.upenn.edu/LDC99T42) data. The [Penn Treebank Parsed](https://catalog.ldc.upenn.edu/LDC99T42) should be put into NLTK's corpus folder.

3. Scripts and commands, from `google-research/`:

  	+  Train Language Modeling
  	```python -m structformer.main --cuda --pos_emb --save /path/to/your/model --data /path/to/your/ptb/corpus```

  	+ Test Unsupervised Parsing
    ```python -m structformer.test_phrase_grammar --cuda --checkpoint /path/to/your/model --print```
    
    The default setting in `main.py` achieves a perplexity of approximately `60.9` on PTB test set, unlabeled F1 of approximately `54.0` and unlabeled attachment score of approximately `46.2` on WSJ test set.
    
## Acknowledgements
Much of our preprocessing and evaluation code is based on the following repository:  
- [Ordered Neurons](https://github.com/yikangshen/Ordered-Neurons)  

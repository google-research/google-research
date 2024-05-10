# **Pa**rallel **D**ecoding by **I**terative **R**efinement

PaDIR (**Pa**rallel **D**ecoding by **I**terative **R**efinement) is a
non-autoregressive decoder implementation in Jax. It builds on top of the T5X
framework, for which documentation is available
[here](https://github.com/google-research/t5x).

Please note that this codebase is experimental.

## Vocabulary

The [MT5](https://github.com/google-research/multilingual-t5) multilingual
vocabulary (256k tokens) is used by default.

In our paper, we use custom bilingual vocabularies.
These can be created as follows:

* Run the `padir/vocab/extract_features.py` script for the desired
[TFDS](https://www.tensorflow.org/datasets) dataset (e.g., WMT14 EN-DE).

* Create a [sentence-piece](https://github.com/google/sentencepiece) vocabulary
using the data extracted in the previous step.
For our experiments we run the sentence-piece training script with the
following parameters:
```--vocab_size=32000 --max_sentence_length=100000 --pad_id=0 --eos_id=1 --unk_id=2 --bos_id=3```.

* Update the _CUSTOM_VOCAB_FILE variable inside `padir/config_options.py`
and update the PaDIR config accordingly, before training your model.

## Dataset

Distilled datasets in our paper were created via the Google Cloud Translate API.
Reducing noise in the data helps train significantly better models.
We are not able to release these datasets but similar datasets can be
recreated with any Translation API or most large language models.

## Citation

If you use or extend this work, please cite our paper as follows:

```
@article{
  li2024gmlm,
  title={Promises and Pitfalls of Generative Masked Language Modeling: Theoretical Framework and Practical Guidelines},
  author={Yuchen Li and Alexandre Kirchmeyer and Aashay Mehta and Yilong Qin and Boris Dadachev and Kishore A Papineni and Sanjiv Kumar and Andrej Risteski},
  booktitle={International Conference on Machine Learning (2024)},
  year={2024},
}
```

## Note

This is not an officially supported Google product.
# Fast Weight Layers

## Background

**Fast Weight Layers (FWLs)** can easily be added on top of text generation models to improve their performance, especially on long documents. They are essentially hidden layers added between the body of a Transformer decoder (or RNN) and the output softmax. However, similar to [dynamic evaluation](https://arxiv.org/abs/1709.07432), their parameters change base on observed tokens. They provide similar performance gains to dynamic evaluation, but are much faster and easier to use. For more details, see our paper *Meta-Learning Fast Weight Language Models* (published in EMNLP 2022).

## Usage
We have provided implementations in both jax and tensorflow. The jax implementation subclasses [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#module); the tensorflow implementation subclasses [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). The jax code has not been thoroughly tested, so please let us know if you spot any issues! Both implementations are constructed with three arguments:
* `size`: size of the FWLs; a fast weight block consists of a dense layer projecting the input to 4 * size, a squared ReLU activation, a dense layer projecting back to size, and a layer norm.
* `vocab size`: number of tokens in the vocabulary.
* `attn_chunks`: number of chunks for the [mixed chunk attention](https://arxiv.org/abs/2202.10447) used in dense fast weight layers. 16 works well for long (e.g. 4096 tokens) inputs; the number could be smaller for shorter inputs.

They are called with three inputs:
* `x`: a [batch_size, seq_len, repr_size] tensor containing input representations (e.g. those produced by a transformer).
* `labels`: a [batch_size, seq_len, vocab_size] tensor containing one-hot labels for the tokens to be predicted. These can typically be constructed as something like `one_hot(roll(input_tokens, -1, 1), vocab_size)`.
* `weights`: a [batch_size, seq_len] tensor of weights for the loss; typically 1 for most tokens and 0 for padding and the last token in each sequence.

For example, usage could look something like:

```
labels, weights = make_labels(input_tokens)
x = CausalTransformer(...)(input_tokens, ...)
logits = FWLBlock(size, vocab_size, attn_chunks)(x, labels, weights)
```

## Citation
If you find the code or paper useful then please cite:

```
@inproceedings{clark2022meta,
  title = {Meta-Learning Fast Weight Language Models},
  author = {Kevin Clark and Kelvin Guu and Ming-Wei Chang and Panupong Pasupat and Geoffrey Hinton and Mohammad Norouzi},
  booktitle = {Empirical Methods in Natural Language Processing},
  year = {2022}
}
```

## Questions?
If you have any questions, comments, or suggestions, please reach out to Kevin Clark ([kevclark@google.com](mailto:kevclark@google.com)).
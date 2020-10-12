# (AL)BERT in Flax

## Current Status

***NOTE: This implementation is work in progress!***

This implementation works, but does not (yet) correspond exactly to BERT. A few
things to note:

*  The implementation actually corresponds to [ALBERT](https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html). The main difference with
   BERT is that the weights between the Transformer layers are shared.

*  Pre-training is using a different vocabulary than the default one used by
   BERT. This is to avoid some of the pre-processing that was done in the
   original ALBERT code.
This implementation allows one to pre-train and fine an [ALBERT](https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html)
model.

## Planned changes

- [ ] Create Colab.
- [ ] Add tests.
- [ ] Add type annotations.
- [ ] Do extensive benchmarks of pre-training and fine-tuning.
- [ ] Simplify attention (use Flax attention instead of efficient_attention).
- [ ] Use [CLU](https://pypi.org/project/clu/) library for metrics reporting.

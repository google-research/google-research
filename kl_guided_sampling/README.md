T5X implementation for the KL-divergence guided temperature sampling

https://arxiv.org/abs/2306.01286

The algorithm is composed of two parallel decodings: one with evidence in the
inputs and one without. The distinction between with/without evidences
is specified by the sign of the input token sequences. Any negative token IDs
are considered part of the evidence. In one decoding loop, we will take the
absolute of the token IDs and, hence, preserving the evidence; In the other
decoding loop, we remove negative token IDs and, hence, removing the evidence.
The logic is implemented in kl_guided_sampling/feature_converters.py.

The specification of the evidence through negative token IDs are expected to be
implemented in SEQIO task preprocessing and by users. The output tasks features
remains "inputs" and "targets", except that some tokens IDs in "inputs" are
negative.

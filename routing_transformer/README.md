# Efficient content-based sparse attention with Routing Transformers

<img src = "image/routing_attention.png" alt="Routing attention">

Code-base accompanying the [paper](https://arxiv.org/abs/2003.05997).

## Explanation of hyperparameters

For local attention the main hyperparameters are `query_shape`, `memory_flange`
and `local_num_heads`. The hparams `query_shape` and `memory flange` represent
the size of the local block used for local attention, for e.g. if the total
block size of local attention is 512, then `query_shape` and `memory_flange`
should be set to (256,) each. The hparam `local_num_heads` represents the number
of local attention heads. An example hparam setting can be found in the
`sparse_transformer.py` file under the name `pg19_local8k`.

For routing attention the main hyperparameters are `sparsity_cluster_size`,
`sparsity_cluster_attention_window` and `sparsity_cluster_num_heads`. The hparam
`sparsity_cluster_size` represents the number of clusters. The
hparam `sparsity_cluster_attention_window` represents the average size of a
cluster. So for example, if the total sequence length is 8192, and
`sparsity_cluster_attention_window` is set to 512, a good choice for
`sparsity_cluster_size` could be 16 (8192/512=16). In practice, we also
divide the total attention head budget equally between local and routing
attention, so `sparsity_cluster_num_heads` is set to be equal to
`local_num_heads`. The other useful hyperparameter is `sparsity_skip_first`
which allows us to skip clustering attention in the first few layers.
This is motivated from our empirical finding that long range attention is only
needed in the later layers of the model. An example hparam setting can
be found in the `sparse_transformer.py` file under the name
`pg19_local_cluster8k`. For using the routing attention in encoder only mode
(i.e., no left to right masking), set the hparam `masked = False` in
`utils.multihead_attention()`.

The other main hparam to think about would be `num_decoder_layers`. For local
attention you want this to be very high (as much as fits in your memory).
However, for Routing Transformers the number of decoder layers needed is lower
(since each attention layer routes information more efficiently than local
attention). Usually, setting it to half the number of layers used for local
attention suffices.

Other hparams to take care of are `max_relative_position`, which denotes the max
distance considered for relative attention. This should be bigger than
`query_shape + memory_flange` and `sparsity_cluster_attention_window`.

## Unconditional samples

Some unconditional samples from a partially trained model on Gutenberg books
(sequence length 8k):

- [sample1](https://docs.google.com/document/d/1YE6644MprOr1vJkY0lJPeYswJQxncBmD_O12LQAMxIA/edit?usp=sharing)
- [sample2](https://docs.google.com/document/d/1UwCYAbIMHOXe07X5ELMwTPa90rqrZCGiJML4jywc0yY/edit?usp=sharing)
- [sample3](https://docs.google.com/document/d/1dC2zNExumaaxTu7BiClo88bZ0JKJMAJolJQDkcOHT70/edit?usp=sharing)
- [sample4](https://docs.google.com/document/d/1zoYG-x_1ElNZc6TatHfGgasNKAuOEqtaBI91ygfb2jA/edit?usp=sharing)
- [sample5](https://docs.google.com/document/d/1XvwY8jFUGGEw3S2HzNx7gBg-9nzSRWHtQVNQAyTVuAU/edit?usp=sharing)
- [sample6](https://docs.google.com/document/d/1RZrOI8e7n7czgA_a7Mt34ePymUFwyjEYrjohZ8aoBoc/edit?usp=sharing)
- [sample7](https://docs.google.com/document/d/1WfSqLCAEd8W3_s3dpaLPH3JwCG3ucBiK_JsoG8q0K3U/edit?usp=sharing)

Document Machine Translation Samples (sequence length 8k):
- [sample](https://docs.google.com/document/d/1yfWHL3JBXAnnfYzrMvxZLpetAsWK5nbF1TslEEqGiPw/edit?usp=sharing)

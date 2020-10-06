# Efficient content-based sparse attention with Routing Transformers

<img src = "image/routing_attention.png" alt="Routing attention">

Code-base accompanying the [paper](https://arxiv.org/abs/2003.05997).

## Explanation of hyperparameters

### Local Attention

*   `local_num_heads`: Number of local attention heads
*   `query_shape`: This represents the shape of the query block.
    *   For 1-d local attention with block size `b`, this would be `(b,)`
*   `memory_query_shape`: This represents the query shape of memory antecedent
    and is useful for encoder-decoder attention
    * This is usually set the same as `query_shape` by default
    * This is useful when input and targets are of different lengths
    * E.g., if inputs are of length 4096 and targets of length 8192
    * Plausible setting:`query_shape = (256,)`, `memory_flange = (256,)` and
      `memory_query_shape = (128,)`
    * This is because with block size `256`, the targets will have `32` blocks
    * To match this in enc-dec attention, the inputs must have `32` blocks
    * This is why we set `memory_query_shape = (4096/32,) = (128,)`
*   `memory_flange`: This represents the overlap of the memory block
    *   For full attention, or for axial attention this would be `(0,)`
    *   Usually for local attention we set it equal to query shape
*   Example setting can be found in `sparse_transformer.py` under `pg19_local8k`

### Routing Attention

*   `sparsity_cluster_num_heads`: Number of routing attention heads
*   `sparsity_cluster_size`: Number of clusters
*   `sparsity_cluster_attention_window`: Average size of each cluster
*   `sparsity_skip_first`: Number of initial layers to skip routing attention
    *   `sparsity_skip_first = 0` would have routing attention in every layer
    *   `sparsity_skip_first` equalling total layers would have no routing
        attention
*   Example setting can be found in `sparse_transformer.py` under
    `pg19_local_cluster8k`

## Samples

### PG-19 (sequence length 8k):

- [sample1](https://docs.google.com/document/d/1YE6644MprOr1vJkY0lJPeYswJQxncBmD_O12LQAMxIA/edit?usp=sharing)
- [sample2](https://docs.google.com/document/d/1UwCYAbIMHOXe07X5ELMwTPa90rqrZCGiJML4jywc0yY/edit?usp=sharing)
- [sample3](https://docs.google.com/document/d/1dC2zNExumaaxTu7BiClo88bZ0JKJMAJolJQDkcOHT70/edit?usp=sharing)
- [sample4](https://docs.google.com/document/d/1zoYG-x_1ElNZc6TatHfGgasNKAuOEqtaBI91ygfb2jA/edit?usp=sharing)
- [sample5](https://docs.google.com/document/d/1XvwY8jFUGGEw3S2HzNx7gBg-9nzSRWHtQVNQAyTVuAU/edit?usp=sharing)
- [sample6](https://docs.google.com/document/d/1RZrOI8e7n7czgA_a7Mt34ePymUFwyjEYrjohZ8aoBoc/edit?usp=sharing)
- [sample7](https://docs.google.com/document/d/1WfSqLCAEd8W3_s3dpaLPH3JwCG3ucBiK_JsoG8q0K3U/edit?usp=sharing)

### Document Machine Translation (sequence length 4k):

-   [sample](https://docs.google.com/document/d/1wqKAyHx7IzJIS0nH9zFYM6KxkjR1qlnYjaECUI9YmmY/edit?usp=sharing)

## How to Cite

If you extend or use this work, please cite the
[paper](https://arxiv.org/abs/2003.05997) where it was introduced:

```
@article{roy2020efficient,
  title={Efficient content-based sparse attention with routing transformers},
  author={Roy, Aurko and Saffar, Mohammad and Vaswani, Ashish and Grangier, David},
  journal={arXiv preprint arXiv:2003.05997},
  year={2020}
}
```

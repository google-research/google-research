# ScaNN Algorithms and Configuration

ScaNN supports several vector search techniques, each with their own tuning
parameters. This page explains the basics of the various techniques.

ScaNN performs vector search in three phases. They are described below:

1.  Partitioning (optional): ScaNN partitions the dataset during training time,
    and at query time selects the top partitions to pass onto the scoring stage.
2.  Scoring: ScaNN computes the distances from the query to all datapoints in
    the dataset (if partitioning isn't enabled) or all datapoints in a partition
    to search (if partitioning is enabled). These distances aren't necessarily
    exact.
3.  Rescoring (optional): ScaNN takes the best k' distances from scoring and
    re-computes these distances more accurately. From these k' re-computed
    distances the top k are selected.

All three phases can be configured through the ScaNNBuilder class. Before going
into details, below are some general guidelines for ScaNN configuration.

## Rules-of-thumb

*   For a small dataset (fewer than 20k points), use brute-force.
*   For a dataset with less than 100k points, score with AH, then rescore.
*   For datasets larger than 100k points, partition, score with AH, then
    rescore.
*   When scoring with AH, `dimensions_per_block` should be set to 2.
*   When partitioning, `num_leaves` should be roughly the square root of the
    number of datapoints.

## Partitioning

Partitioning is configured through `.tree(...)`. The most important parameters
are `num_leaves` and `num_leaves_to_search`. The higher `num_leaves`, the
higher-quality the partitioning will be; however, raising this parameter also
makes partitioning take longer. `num_leaves_to_search / num_leaves` determines
the proportion of the dataset that is pruned. Raising this proportion increases
accuracy but leads to more points being scored and therefore less speed.

If a dataset has n points, the number of partitions should generally be the same
order of magnitude as `sqrt(n)` for a good balance of partitioning quality and
speed. `num_leaves_to_search` should be tuned based on recall target.

## Scoring

Scoring can either be done with brute-force or asymmetric hashing (AH). The
corresponding functions in ScannBuilder are `score_brute_force` and `score_ah`.
Unless near-perfect accuracy is required, AH gives better speed/accuracy
tradeoffs. See
[Product quantization for nearest neighbor search](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf)
and Gersho and Gray's *Vector Quantization and Signal Compression* for
background on AH, and
[Accelerating Large-Scale Inference with Anisotropic Quantization](https://arxiv.org/abs/1908.10396)
for an academic descrition of `anisotropic_quantization_threshold`. We recommend
setting `dimensions_per_block` to 2.

The only parameter for brute-force scoring is `quantize`. If enabled, this
quantizes each dimension of each datapoint into an 8-bit integer. This is a 4x
compression ratio over 32-bit floats and can therefore quarter the latency in
memory-bandwidth bound scenarios. However, quantized brute force is slower than
full brute force in non-memory bandwidth bound scenarios, which occur when the
batch size is sufficiently large or when the dataset is small enough to fit into
cache. Quantized brute-force generally leads to negligible accuracy losses over
full brute force.

## Rescoring

Rescoring is highly recommended if AH scoring is used.
`reordering_num_neighbors` should be greater than k, the final number of
neighbors. Raising `reordering_num_neighbors` increases accuracy at the cost of
speed. `quantize` has the same meaning as in brute-force scoring.

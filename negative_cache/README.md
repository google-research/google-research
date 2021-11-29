# Efficient Training of Retrieval Models using Negative Cache

This is code for the NeurIPS 2021 paper [Efficient Training of Retriveal Models
  using Negative Cache](https://openreview.net/pdf?id=824xC-SgWgU). It implements
a streaming negative cache for training dual encoders. The approach is memory
efficient and simple to use.

At each iteration, we sample a negative from the cache and use it to approximate
the cross-entropy loss function. The cache is designed to allow a large amount of
negatives in a memory efficient way.

For full details on the approach see the paper or watch the video (available soon).


## Usage

Set up the specs that describe the document feature dictionary. These describe
the feature keys and shapes for the items we need to cache.

```
data_keys = ('document_feature_1', 'document_feature_2')
embedding_key = 'embedding'
specs = {
    'document_feature_1': tf.io.FixedLenFeature([document_feature_1_size], tf.int32),
    'document_feature_2': tf.io.FixedLenFeature([document_feature_2_size], tf.int32),
    'embedding': tf.io.FixedLenFeature([embedding_size], tf.float32)
}
```

Set up the cache loss.

```
cache_manager = negative_cache.CacheManager(specs, cache_size=131072)
cache_loss = losses.CacheClassificationLoss(
    embedding_key=embedding_key,
    data_keys=data_keys,
    score_transform=lambda score: 20.0 * score,  # Optional, applied to scores before loss.
    top_k=64  # Optional, restricts returned elements to the top_k highest scores.
)
handler = handlers.CacheLossHandler(
    cache_manager, cache_loss, embedding_key=embedding_key, data_keys=data_keys)
```

Calculate the cache loss using your query and document networks and data.

```
query_embeddings = query_network(query_data)
document_embeddings = document_network(document_data)
loss = handler.update_cache_and_compute_loss(document_network, query_embeddings,
                                             document_embeddings, document_data)
```
## Bibtex

```
@inproceedings{negative_cache_2021,
  title={Efficient Training of Retrieval Models using Negative Cache},
  author={Lindgren, Erik and Reddi, Sashank and Guo, Ruiqi and Kumar, Sanjiv},
  booktitle={Neural Information Processing Systems},
  year={2021},
  URL={https://openreview.net/pdf?id=824xC-SgWgU}
}
```

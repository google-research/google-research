
# Beyond 512 Tokens: Siamese Multi-depth Transformer-based Hierarchical Encoder for Long-Form Document Matching

We will release the code of SMITH (Siamese Multi-depth Transformer-based Hierarchical) model. Please check back for details.

**<a href="https://arxiv.org/abs/2004.12297">Beyond 512 Tokens: Siamese Multi-depth Transformer-based Hierarchical Encoder for Long-Form Document Matching
</a>**
<br>
Liu Yang, Mingyang Zhang, Cheng Li, Michael Bendersky, Marc Najork
<br>
Accepted at [CIKM 2020](https://www.cikm2020.org/).


*Please note that this is not an officially supported Google product.*

If you find this code useful in your research then please cite

```
@inproceedings{yang2020beyond,
  title={Beyond 512 Tokens: Siamese Multi-depth Transformer-based Hierarchical Encoder for Long-Form Document Matching},
  author={Liu Yang and Mingyang Zhang and Cheng Li and Michael Bendersky and Marc Najork},
  booktitle={CIKM},
  year={2020}
}
```

## Introduction

Many natural language processing and information retrieval problems can be
formalized as the task of semantic matching. Existing work in this area has
been largely focused on matching between short texts (e.g., question answering),
or between a short and a long text (e.g., ad-hoc retrieval). Semantic matching
between long-form documents, which has many important applications like news
recommendation, related article recommendation and document clustering, is
relatively less explored and needs more research effort. In recent years,
self-attention based models like Transformers and BERT have achieved
state-of-the-art performance in the task of text matching. These models,
however, are still limited to short text like a few sentences or one paragraph
due to the quadratic computational complexity of self-attention with respect to
input text length.

In this project, we address the issue by proposing the Siamese Multi-depth
Transformer-based Hierarchical (SMITH) Encoder for long-form document matching.
Our model contains several innovations to adapt self-attention models for longer
text input. Our experimental results on several benchmark data sets for
long-form document matching show that our proposed SMITH model outperforms the
previous state-of-the-art models including hierarchical attention multi-depth
attention-based hierarchical recurrent neural network, and BERT.

We will release the code of the SMITH model in this repository.

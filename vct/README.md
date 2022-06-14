# VCT: A Video Compression Transformer

Fabian Mentzer, George Toderici, David Minnen, Sung-Jin Wang, Sergi Caelles, Mario Lucic, Eirikur Agustsson

https://arxiv.org/abs/ [Coming soon]

**Work In Progress, Code Coming Soon**


## Abstract

We show how transformers can be used to vastly simplify neural video
compression. Previous methods have been relying on an increasing number of
architectural biases and priors, including motion prediction and warping
operations, resulting in complex models. Instead, we independently map input
frames to representations and use a transformer to model their dependencies,
letting it predict the distribution of future representations given the past.
The resulting video compression transformer outperforms previous methods on
standard video compression data sets. Experiments on synthetic data show that
our model learns to handle complex motion patterns such as panning, blurring and
fading purely from data. Our approach is easy to implement, and we release code
to facilitate future research.


# Fractal Patterns in Language

## Paper
[*Fractal Patterns May Unravel the Intelligence in Next-Token Prediction*](https://arxiv.org/abs/2402.01825).
Ibrahim Alabdulmohsin, Vinh Q. Tran, and Mostafa Dehghani, arXiv:2402.01825 [cs.CL].

*Abstract*:
We study the fractal structure of language, aiming to provide a precise
formalism for quantifying properties that may have been previously suspected but
not formally shown. We establish that language is: (1) self-similar, exhibiting
complexities at all levels of granularity, with no particular characteristic
context length, and (2) long-range dependent (LRD), with a Hurst parameter of
approximately H=0.70. Based on these findings, we argue that short-term
patterns/dependencies in language, such as in paragraphs, mirror the
patterns/dependencies over larger scopes, like entire documents. This may shed
some light on how next-token prediction can lead to a comprehension of the
structure of text at multiple levels of granularity, from words and clauses to
broader contexts and intents. We also demonstrate that fractal parameters
improve upon perplexity-based bits-per-byte (BPB) in predicting downstream
performance. We hope these findings offer a fresh perspective on language and
the mechanisms underlying the success of LLMs.


## Colab
The attached colab provides a walk-through example for how to calculate fractal
parameters from a collection of documents using a large language model (LLM). Specifically, we outline
how to calculate: (1) the self-similarity exponent *S*, (2) the Hurst exponent *H*,
and (3) the Joseph exponent *J*.

Please reach out to the authors for any inquiries.
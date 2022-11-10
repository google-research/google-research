# Scaling Transformer Inference Efficiency
This repo includes

* Benchmarks to replicate the results in the paper - [Scaling Transformer Inference Efficiency](http://arxiv.org/abs/2211.05102)
* A complete implementation of text generation with a transformer using the techniques in the paper

## To replicate the head-to-head benchmarks from the paper at 540B scale

* Ensure you are running on 64 TPUv4 chips, smaller numbers would be better suited for smaller models

```python
python3 run_benchmark.py
```

This generates the latency and MFU numbers for the PALM and MT-NLG implementations in the following plot from the paper. The FastertTransformer baseline numbers are drawn from NVIDIA's repo.

<!-- <p align="center">
<img src="./assets/fasterformer_comparison_v2.pdf" alt="fastertransformer_comparison" width="600"/>
</p> -->


## To generate text

```python
python3 run_generation.py --model 540b --quantized False
```

The current weight paths only load internal PaLM weights, which are unavailable externally. Using this externally will require modification of the checkpoint paths and transformer layer def to suit your own models.
Text generation currently uses the pjit based code paths, updating to the faster xmap based code paths is in progress and should be done by next week.

TODO:

* Insert table from benchmarks run
* Include benchmark at larger setpoints
* Update text generation to xmap code path
* Include helper scripts for running TPU pod slices
* Update this documentation
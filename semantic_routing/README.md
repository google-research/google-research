# Semantic Routing via Autoregressive Modeling

[Paper link](https://nips.cc/virtual/2024/poster/95691).

## Environment Setup

```bash
pip install -r requirements.txt
```

## Dataset Setup

Download the dataset into the root directory:
```bash
gsutil cp gs://argon-zoo-432801-i8/semantic_routing .
```

Update `benchmark/config.py` to point to the dataset's path.

## Getting Started

Please see `examples` for notebooks demonstrating the use of this benchmark
software, and `run_example.py` for a short script demonstrating this benchmark
as a Tensorflow dataset.
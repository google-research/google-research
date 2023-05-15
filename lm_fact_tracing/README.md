# Tracing Factual Knowledge in LMs Back To Their Training Data

Code for our paper, [Towards Tracing Factual Knowledge in Language Models Back to the Training Data](https://arxiv.org/abs/2205.11482).

**Authors:** Ekin Akyürek, Tolga Bolukbasi, Frederick Liu, Binbin Xiong, Ian Tenney, Jacob Andreas, Kelvin Guu.

**Abstract:** Language models (LMs) have been shown to memorize a great deal of factual knowledge contained in their training data. But when an LM generates an assertion, it is often difficult to determine where it learned this information and whether it is true. In this paper, we propose the problem of fact tracing: identifying which training examples taught an LM to generate a particular factual assertion. Prior work on training data attribution (TDA) may offer effective tools for identifying such examples, known as "proponents". We present the first quantitative benchmark to evaluate this. We compare two popular families of TDA methods -- gradient-based and embedding-based -- and find that much headroom remains. For example, both methods have lower proponent-retrieval precision than an information retrieval baseline (BM25) that does not have access to the LM at all. We identify key challenges that may be necessary for further improvement such as overcoming the problem of gradient saturation, and also show how several nuanced implementation details of existing neural TDA methods can significantly improve overall fact tracing performance.

## Installation

```bash
$ cd /path/to/google_research  # Navigate to root of the repository.
$ python3 -m venv .venv  # Create a Python virtual environment.
$ source .venv/bin/activate  # Activate the environment.
$ pip install -r lm_fact_tracing/requirements.txt  # Install dependencies.
```

## Running data generation to create FTRACE-SYNTH

```bash
$ cd /path/to/google_research  # Navigate to root of the repository.
$ python -m lm_fact_tracing.synth_data.synth_data_run \
--output_path="/tmp/ftrace_synth" \
--entity_vocabulary="/path/to/google_research/lm_fact_tracing/synth_data/names.txt" \
--relation_vocabulary="/path/to/google_research/lm_fact_tracing/synth_data/relations.txt"
```

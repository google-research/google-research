# Open source release of [QANet](https://arxiv.org/abs/1804.09541)

This directory contains a preliminary release of the QANet model.

Training a model requires the raw SQuAD v1.1 json files and word embeddings
such as [fasttext](https://fasttext.cc/docs/en/english-vectors.html).


From the root google_research directory, run:
```
python -m qanet.train --model_dir /tmp/qanet_models/run_1 --config fn=SQUADExperiment,train_steps=100,dataset.data_path=/path/containing/squad_v1.1.json/and/fasttext/vocab.vec
```

This will train a model for 100 steps.  See the config.json produced in the
model directory for a complete specification of all parameters.

One can provide a complete json config through the --config_file argument.

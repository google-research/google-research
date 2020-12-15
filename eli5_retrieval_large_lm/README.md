
# GPT2 and Retrieval
Project realized by **Jules Gagnon-Marchand** ([Mila](mila.quebec)) while interning at Google Brain,

under the supervision of **Noam Shazeer** (Brain) and **Aurko Roy** (Brain).

*[jgagnonmarchand@gmail.com](jgagnonmarchand@gmail.com), [noam@google.com](noam@google.com), [aurkor@google.com](aurkor@google.com)*


### Description
The objective is to test a reasonably large language model 
(GPT2-XL, 1.5B parameters) on the [Kilt](https://ai.facebook.com/tools/kilt/) 
version of the [ELI5](https://www.aclweb.org/anthology/P19-1346/) task when it is
combined with a retriever ([REALM](https://arxiv.org/abs/2002.08909) in this case). 


We wanted to observe whether larger causal language models can make use of the 
retrieved contextual information, as their reasoning capacities are stronger than that 
previously tested models. Indeed, it has been shown recently that smaller 
language models don't make use of the retrieved text in the context of ELI5. 
In this project we investigate whether increasing the model capacity can play a 
role in making the models use retrievals. Models with increased capacity have 
improved reasoning capabilities, which may help here.

Once the usefulness of retrievals for larger language models would be established, 
inspecting what kinds of retrievals are useful and why would be a next step, as 
well as investigating the effects of the retrieval on factual consistency in 
generation, which is a problem of major interest right now, with massive 
financial implications, as it would allow the use of generative text models in 
products (other than translation).


Inspecting what kinds of retrievals are useful and why would be a next step, as 
well as investigating the effects of the retrieval on factual consistency in 
generation, which is a problem of major interest right now.


### Usage
To train the model, you can find examples of configuration in the `configs/train_configs/` folder, 
and use them as follows:

```
python main.py $(python json_to_args.py configs/train_configs/file_of_your_chosing.json)
```


### Approaches:
##### Simple Language Modelling:
We use GPT2-XL to train over the whole of [Kilt](https://ai.facebook.com/tools/kilt/) 
ELI5, masking the question in the loss computation.

##### Real time retrieval with Scann and REALM:
We use GPT2-XL (Or GPT2 models of other size). Retrievals are done over 
Wikipedia and concatenated to the question, and the answer is then concatenated. 
The question and the context are masked in the loss. The model is trained with 
simple language modeling otherwise. Spaces are added between the question and
first context, between the contexts, and between the last context and the 
answer. There is also to add helper words such as "Question: ", " Context: ",
and " Answer: " (all masked in the loss), to make the task easier to understand
for the model. The retriever used is the REALM retriever. Retrieval is made 
using the REALM query embedder on the question.

##### Pre-made retrieval with exact retrievals and runtime sampling
Instead of using ScaNN to do live retrievals at training and evaluation time,
as the questions don't change, we do all the retrievals in advance with an 
exact retriever. Each question is embedded with the REALM retriever, and 
we save a number (100) of exact MIP nearest neighbors in TFRecord files, as well
as the id of the question they are associated with. We save the indices 
of the REALM wikipedia segments DB of the MIP nearest neighbors, as well as the 
inner products, so the inner products can be used as logits for sampling, at
language model training time. We started by using HDF5 instead of TFRecords,
but it turns out that `tf.data` tries to load the whole file in memory,
defeating the point of a memory mapped array (and often crashing the instance),
and didn't allow to use `tf.distribute.TPUStrategy.experimental_distribute_dataset`
and automatic dataset sharding per TPU. 

When we train over ELI5 with GPT2, we use the indices of the current question
to get back the indices of the entries of the REALM Wikipedia segments that are
the closest neighbors. We also get the inner products, which are used to 
sample from the neighbors saved for a question, with a temperature parameter.
We obtain the probability of sampling a neighbor by doing 
`softmax(inner_products / temperature)`, and we then do sampling without 
replacement.


### Parallelism:
- The training script supports large scale parallelism on TPUs and on 
GPUs through `tf.distribute.TPUStrategy` and `tf.distribute.MirroredStrategy`,
respectively.
- The full data pipeline uses `tf.data.Datasets`, including the retriever.
This allows us to use `tf.distribute.Strategy.experimental_distribute_dataset` 
to automatically shard the dataset on the TPUs. 
- The query caching script supports parallelism with 
`tf.distribute.TPUStrategy` and `tf.distribute.MirroredStrategy`, although in 
our experience, a single V100 is enough (done in slightly over an hour). 


### Executables:
- **`main.py`**: Script to launch the distributed training of one of the different approaches.
- **`generation.py`**: Script to launch generation from previously trained models. Also massively distributed.
- **`query_cacher_tfrecord.py`**: Script to prepare the pre-made retrievals for ELI5, 
for the FullyCachedRetriever, with TFRecords.
- **`util_scripts/scann_test_recall.py`**: Tests the recall of one's desired Scann configuration
for a certain specified datast, by comparing to exact retrieval.
- `check_flags.py`: Tool that looks at a script to check if
all variables of the type `_FLAG_*` and `FLAG_*` end with 
`.value` if they aren't being defined with `flag.DEFINE_*`. 
This is just a baseline test to check to detect easy mistakes.
- `json_to_args.py`: Simple utility that reads a `.json` file and outputs command 
line arguments compatible with `absl.flags`, so one can run 
`python script.py $(python to_flag.py config/script_flags.json)`
- `util_scripts/count_records.py`: Counts the number of records in the REAM
database.
- `util_scripts/create_data_subset_realm.py`: Creates a subset of the REALM 
dataset, for debugging purposes, to prevent long loading times.

### Libraries:
 - **`retrievers.py`**: Location of the retriever classes and the retriever
 related logic in general.
 - **`task_specific_.py`**: Location of the dataset preparation logic,
 of the model loading logic, and of an important part of the parallelism logic.
 - `bert_utils.py`: Various BERT related utilities, for things such as loading 
 it's tokenizer.
 - `constants.py`: Various configuration constants used throughout the solution,
 such as the different types of parallelism flags that are supported, the
 different training approaches that are supported, the different retrieval types
 that are supported.
 - `modeling_tf_gpt2_model_par.py`: GPT2 modeling script modified from 
 HuggingFace's GPT2 modeling script to support splitting models vertically 
 amongst a number of accelerators, over a number of replicas.
 - `scann_utils.py`: Various utilities relating to scann.
 - `tf_utils.py`: Utilities involving Tensorflow, such as logic directly dealing
 with TPUs and other devices.
 - `utils.py`: All general purpose utilities not involving Tensorflow can be 
 found here.
 
### Notebooks:
- **`Compute_Cumul_Lengths.ipynb`**: 
Computes the distribution of lengths for 
`gpt2_tokenizer.tokenize(question_text + answer_text)` arrays. Gives an idea
of the fraction of the dataset that will be able to get different amounts of 
retrieved contexts.  
- **`Cumul_Lengths_Retrieval.ipynb`**:
Computes the distribution of the lengths of a representative subset of the 
Wikipedia reference document with the GPT2 tokenizer. Helpful again to predict
the number of retrievals each segment will be able to obtain.
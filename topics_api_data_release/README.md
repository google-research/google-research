# Data and code release for 'Differentially Private Synthetic Data Release for Topics API Outputs' KDD'25 Datasets and Benchmarks track.

This repository contains the code for the work described in the KDD'25 (Datasets
and Benchmarks) paper 'Differentially Private Synthetic Data Release for Topics
API Outputs', Travis Dick, Alessandro Epasto, Adel Javanmard, Josh Karlin,
Andres Munoz medina, Vahab Mirrokni, Sergei Vassilvitskii, Peilin Zhong.
The data release is hosted on Kaggle
[here](https://www.kaggle.com/datasets/googleai/topics-api-private-data-release).

Details of the methodology used to obtain the data, the privacy protection, and
limitation of the work are available in the paper.

## Usage
If you use our code, or data, please cite the paper mentioned above.

After cloning this directory, install the dependencies to a virtual environment
by running the following within the `topics_api_data_release` directory:

```bash
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

To fit parameters to the statistics, you will need three files: `q_within.json`,
`q_across.json`, and `taxonomy.json` which can be found on the kaggle page
[here](https://www.kaggle.com/datasets/googleai/topics-api-private-data-release).
Then you can fit the model parameters to the statistics by running the following
command from the parent directory of `topics_api_data_release`:

```bash
python3 -m topics_api_data_release.fit_synthetic_distribution_main \
  --across_week_stats_path=[path to q_across.json] \
  --within_week_stats_path=[path to q_within.json] \
  --topic_taxonomy_path=[path to topic_taxonomy.json] \
  --output_path=[path to output_parameters.json] \
  --num_weeks=4 \
  --num_types=500 \
  --alsologtostderr
```

Once you have fit the parameters (or if you use our released parameters, which
can be found named output_parameters.json on the kaggle page
[here](https://www.kaggle.com/datasets/googleai/topics-api-private-data-release),
you can sample synthetic users from it by running the following command from the
parent directory of `topics_api_data_release`:

```bash
python -m topics_api_data_release.sample_topic_sets_main \
  --distribution_path=[path to output_parameters.json] \
  --topic_taxonomy_path=[path to topic_taxonomy.json] \
  --num_samples=1000000 \
  --output_prefix=[prefix of output files]
  --alsologtostderr
```
This will output a sharded TFRecord file with paths of the form
`{output_prefix}-#####-of-#####.tfrecord` (and create any directories that are
included in the prefix). The format of the sampled output is described below.

## Data
The data is hosted in a separated hosting service. To obtain the data please
download it from the
[Topics API Kaggle dataset](https://www.kaggle.com/datasets/googleai/topics-api-private-data-release).
Here we describe the data.

The 3 main set of files: qwithin.json and qacross.json, and the topic sequences.
The first two files contain the differentially private statistics measured
from real users’ topics data. They are json files containing a pandas
DataFrame. The topic sequences are the result of the sampling of the model
trained on the DP statistics. We now describe these files.

### qwithin.json

This file contains the within week topics distribution (see Section 4.1 of the
paper for details). The file is in json format. It is possible to read it with:

```python
import pandas as pd
df_within = pd.read_json("q_within.json")
```
The data frame contains 5 columns: `topic_1_id`, `topic_1_label`, `topic_2_id`,
`topic_2_label`, and `frequency`. `topic_i_id` and `topic_i_label` are the id
and label of topic `i`, and `frequency`is the frequency with which those two
topics appear within the same week. Notice that the `frequency` values sum to
approximately (k choose 2) with k = 5, as there are 5 topics per week. The rows
with the highest frequencies are:

|   topic_1_id | topic_1_label         |   topic_2_id | topic_2_label       |   frequency |
|-------------:|:----------------------|-------------:|:--------------------|------------:|
|            1 | /Arts & Entertainment |          243 | /News               |   0.118257  |
|          243 | /News                 |          299 | /Sports             |   0.0905165 |
|            1 | /Arts & Entertainment |          215 | /Internet & Telecom |   0.0664754 |
|          215 | /Internet & Telecom   |          243 | /News               |   0.0553493 |
|          299 | /Sports               |          325 | /Sports/Soccer      |   0.0491392 |

### qacross.json

This file contains the across week topics distribution (see Section 4.1 of the
paper for details). The format is the same of the prior file.

```
import pandas as pd
df_across = pd.read_json("q_across.json")
```

The data frame contains 5 columns: `week_1_topic_id`, `week_1_topic_label`,
`week_2_topic_id`, `week_2_topic_label`, and `frequency`. `week_i_topic_id` and
`week_i_topic_label` are the id and label for the topic in week `i` and
`frequency` is the frequency with which topic 1 appears in one week followed by
topic 2. Notice that the frequency table sums to approximately k^2 with k = 5,
as there are 5x5 topic pairs across 2 weeks.

The rows with highest frequencies are:

|   week_1_topic_id | week_1_topic_label    |   week_2_topic_id | week_2_topic_label    |   frequency |
|------------------:|:----------------------|------------------:|:----------------------|------------:|
|               243 | /News                 |               243 | /News                 |    0.230174 |
|               299 | /Sports               |               299 | /Sports               |    0.115274 |
|                 1 | /Arts & Entertainment |                 1 | /Arts & Entertainment |    0.115148 |
|               289 | /Shopping             |               289 | /Shopping             |    0.113124 |
|               243 | /News                 |                 1 | /Arts & Entertainment |    0.102356 |

### topic_set_sequences-*-of-00100.tfrecord

These 100 files represent the syntheic Topics sequences generated for 10
million synthetic users sampled from the distribution obtained by our model
based on the DP statistics. The files are sharded TFRecord file containing
serialized tf.train.Example protocol buffer messages, each corresponding to a
single synthethic user. In particular, each example has the following features:
 user_id, a unique string for that user, epoch_r_topics for r in {0, 1, 2, 3}
which is an integer feature of length 5 containing the topics for that epoch,
and epoch_r_weights for r in {0, 1, 2, 3}, which is an all 1’s vector. Due to
the post-processing properties of differential privacy, this release is also DP.

The users can be loaded in python using tensorflow via the following:

```python
paths = [f"topic_set_sequences-{n:05}-of-00100.tfrecord" for n in range(100)]
raw_data = tf.data.TFRecordDataset(paths)

feature_description = {
    'user_id': tf.io.FixedLenFeature([], tf.dtypes.string, default_value=""),
    'epoch_0_topics': tf.io.FixedLenFeature([5], tf.int64),
    'epoch_0_weights': tf.io.FixedLenFeature([5], tf.float32),
    'epoch_1_topics': tf.io.FixedLenFeature([5], tf.int64),
    'epoch_1_weights': tf.io.FixedLenFeature([5], tf.float32),
    'epoch_2_topics': tf.io.FixedLenFeature([5], tf.int64),
    'epoch_2_weights': tf.io.FixedLenFeature([5], tf.float32),
    'epoch_3_topics': tf.io.FixedLenFeature([5], tf.int64),
    'epoch_3_weights': tf.io.FixedLenFeature([5], tf.float32),
}

def _parse_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature_description)

data = raw_data.map(_parse_function)
```

Notice that the topics id are integer and represent the integer id in the Topics
 v2 [taxonymy](https://raw.githubusercontent.com/patcg-individual-drafts/topics/refs/heads/main/taxonomy_v2.md).
 To map them to the labels used in the statistics file, it is possible to use
the mapping provided in taxonomy.json. This file contains a Dataframe with
columns: topics_id and topics_label.


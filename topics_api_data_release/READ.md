# Data and code release for 'Differentially Private Synthetic Data Release for Topics API Outputs' KDD'25 Datasets and Benchmarks track.

This repository contains the data and code for the work described in the KDD'25
(Datasets and Benchmarks) paper 'Differentially Private Synthetic Data Release
for Topics API Outputs', Travis Dick, Alessandro Epasto, Adel Javanmard,
Josh Karlin, Andres Munoz medina, Vahab Mirrokni, Sergei Vassilvitskii, Peilin
Zhong.

Details of the methodology used to obtain the data, the privacy protection, and
limitation of the work are available in the paper.

## Usage
If you use our code, or data, please cite the paper mentioned above.

## Data
The data is hosted in a separated hosting service. To obtain the data please
download it from the [Topics API Kaggle dataset](https://www.kaggle.com/datasets/googleai/topics-api-private-data-release). Here we describe the data.

The 3 main set of files: qwithin.json and qacross.json, and the topic sequences.
The first two files contain the differentially private statistics measured
from real users’ topics data. They are json files containing a pandas
DataFrame. The topic sequences are the result of the sampling of the model
trained on the DP statistics. We now describe these files.

### qwithin.json

This file contains the within week topics distribution (see Section 4.1 of the
paper for details). The file is in json format. It is possible to lead it with:

```
import pandas as pd

with open('qwithin.json', 'r') as f_i:
  df_within_json = pd.read_json(f_i)
df_within_json.head()
```
The data frame contains 3 columns: topic1, topic2, frequency. topic1 and topic2
are Topics API ids and frequency is the frequency with which the two topics
appear within the same week. Notice that the frequency table sums to approx
(k choose 2) with k = 5, as there are 5 topics per week.
One example row is:

```
topic1, topic2, frequency
/Arts & Entertainment /News 0.117659
```

### qacross.json

This file contains the across week topics distribution (see Section 4.1 of the
paper for details). The format is the same of the prior file.

```
import pandas as pd

with open('qacross.json', 'r') as f_i:
  df_qacross_json = pd.read_json(f_i)
df_qacross_json.head()
```
The data frame contains 3 columns: topic_week1, topic_week2, frequency.
topic_week1 and topic_week2 are Topics API ids and frequency is the frequency
with which the first topics appears in week 1 and the second topic appears in
week 2. Notice that the frequency table sums to approx k^2 with k = 5, as there
are 5x5 topic pairs across 2 weeks.

One example row is:

```
topic_week1, topic_week2, frequency
/News /News 0.220051
```

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
 v2 (taxonymy)[https://raw.githubusercontent.com/patcg-individual-drafts/topics/refs/heads/main/taxonomy_v2.md].
 To map them to the labels used in the statistics file, it is possible to use
the mapping provided in taxonomy.json. This file contains a Dataframe with
columns: topics_id and topics_label.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import pandas as pd
import json


FLAGS = flags.FLAGS

flags.DEFINE_string("input", "data/train.tsv",
                    "Input tsv file.")

flags.DEFINE_string("mapping_dict", None,
                    "File containing a mapping dictionary from new targets to old targets.")

flags.DEFINE_string("target_file", "data/targets.txt",
                    "File containing list of old targets.")

flags.DEFINE_string("output_target_file", "data/new_targets.txt",
                    "Output file for list of new targets.")

flags.DEFINE_string("output_data", "data/new_train.tsv",
                    "Output file new data.")


def replace_labels(labels, idx2target, mapping_dict, target2idx):
    split = labels.split(",")
    new_labels = []
    for label_idx in split:
        old_target = idx2target[int(label_idx)]
        for new_target, v in mapping_dict.items():
            if old_target in v:
                new_labels.append(str(target2idx[new_target]))
                break
    assert(len(new_labels) > 0)
    return ",".join(new_labels)


def main(_):

    data = pd.read_csv(FLAGS.input, sep="\t", header=None, names=["text", "labels", "id"])
    targets = open(FLAGS.target_file).read().splitlines() + ["neutral"]
    idx2target = {i: t for i, t in enumerate(targets)}

    with open(FLAGS.mapping_dict) as f:
        mapping_dict = json.loads(f.read())

    new_targets = sorted(list(mapping_dict.keys()) + ["neutral"])

    # Find those targets that are not in the mapping dictionary
    for t in targets:
        found = False
        for k, v in mapping_dict.items():
            if t in v:
                break
        if not found:
            print("%s is not found" % t)

    print("New targets:")
    print(new_targets)
    target2idx = {t: i for i, t in enumerate(new_targets)}

    data["labels"] = data["labels"].apply(replace_labels, argv=(idx2target, mapping_dict, target2idx))

    with open(FLAGS.output_target_file, "w") as f:
        f.write("\n".join(new_targets))

    data.to_csv(FLAGS.output_data, sep="\t", encoding="utf-8", header=False, index=False)


if __name__ == "__main__":
    app.run(main)

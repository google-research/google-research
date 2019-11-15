This is the code for the paper "On Concept-Based Explanations in Deep Neural
Networks" https://arxiv.org/abs/1910.07969".

1.  Please download all data from "gs://concept_discovery/data/" or
    "https://console.cloud.google.com/storage/browser/concept_discovery", and
    change the directory paths when loading data.
2.  To run the toy example, just run python3 toy_main.py.
3.  To create the toy example, just run python3 create_toy.py.
4.  To run the AwA example, just run python3 awa_main.py.

ipca.py is a general helper function for calculating the completeness of a given
model, and toy_helper.py and awa_helper.py contain helper functions that are
specific to the datasets.

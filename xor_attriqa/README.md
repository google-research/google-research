# XOR-AttriQA

Published as a long paper at EMNLP 2023.

This dataset contains data for the attribution task in https://arxiv.org/abs/2305.14332. It is based on data from the XOR-TyDiQA dataset (https://arxiv.org/abs/2010.11856) and model predictions from the CoRA system (https://arxiv.org/abs/2107.11976).

## Data

XOR-AttriQA data: https://storage.googleapis.com/gresearch/xor_attriqa/xor_attriqa.zip.

We include data annotations for both the in-language and in-English settings in the in-language and in-english folders respectively. Use the data from the in-language setting to compare with results in our work. The data from the in-English setting was used to compare inter-annotator agreement between the two settings and is included only as a reference.

## Data Fields

* query: Query from XOR-QA.
* query_language: Language of the query.
* answers: Answers to the query provided by XOR-QA.
* prediction: Prediction from the CoRA system.
* prediction_correct: Whether the prediction was determined to match the answers.
* query_translated_en: Query translated to English.
* answers_translated_en: Answers translated to English.
* prediction_translated_en: Prediction translated to English.
* passage_in_language: Passage in language (see passage_retrieved_language to determine if translated or not).
* passage_en: Passage in English (see passage_retrieved_language to determine if translated or not).
* passage_retrieved_language: Language of retrieved passage.
* intrepetability_vote: Ratio of votes on whether raters could understand the prediction.
* ais_vote: Ratio of votes on whether the prediction is attributable to the passage and query.
* interpretability: True or False on whether raters could understand the prediction.
* ais: True or False on whether the prediction is attributable to the passage and query.

## Usage

The data can be used for both in-language and cross-lingual attribution. In attribution, the task is to take a query, a passage, and an answer (model prediction), determine if the passage does indeed show that the provided answer is correct for the provided question.

For in-language attribution: Use the query, prediction, and passage_in_language fields to predict True or False. The gold label is in the ais field.

For cross-language attribution: Use the query, prediction, and passage_en fields to predict True or False. The gold label is again in the ais field.

For the task used in our paper, we used a mix of in-language and cross-language attribution, depending on which passages were retrieved from the CORA system. To reproduce this setting, use the passage_retrieved_language field to determine whether to select the passage in passage_in_language or the passage in passage_en.

Lastly, we have a validation set composed of 100 examples from each language and a training set composed of 50 high-confidence (all raters agreed on the ais label) examples.

## Number of Data Points per Split

In-Language Setting:

* bn: 1407
* fi: 659
* ja: 954
* ru: 634
* te: 1066
* train: 250
* validation: 500

In-English Setting:

* bn: 567
* fi: 812
* ja: 1262
* ru: 790
* te: 443

## Citation

    @article{muller2023evaluating,
      title={Evaluating and Modeling Attribution for Cross-Lingual Question Answering},
      author={Muller, Benjamin and Wieting, John and Clark, Jonathan H and Kwiatkowski, Tom and Ruder, Sebastian and Soares, Livio Baldini and Aharoni, Roee and Herzig, Jonathan and Wang, Xinyi},
      journal={arXiv preprint arXiv:2305.14332},
      year={2023}
    }

# UGIF: UI Grounded Instruction Following

This repository contains the
[dataset](https://storage.googleapis.com/gresearch/ugif/ugif-dataset.tar.gz)
released along with the paper
"[UGIF: UI Grounded Instruction Following](https://arxiv.org/abs/2211.07615)".
The dataset is released under
[CC BY 4.0 International license](https://github.com/google-research/google-research#google-research)
and any code under the
[Apache 2.0 license](https://github.com/google-research/google-research/blob/master/LICENSE).

## UGIF-DataSet

UGIF-DataSet is a multi-lingual, multi-modal UI grounded dataset for
step-by-step task completion on the smartphone.

It is a corpus of Android how-to queries (speech and text) in 8 languages
(en, kn, mr, gu, hi, bn, sw, es) along with how-to instruction steps in English paired with sequences of UI screens and actions as the how-to is completed by
human annotators on Android devices with different UI language settings. This dataset is an enhanced and updated successor of the
[PixelHelp](https://github.com/google-research/google-research/tree/master/seq2act)
dataset with multilingual queries and UI and also screenshots of the UI.

## Structure of UGIF-DataSet

![UGIF dataset structure](https://storage.googleapis.com/gresearch/ugif/ugif-dataset-vis.png)

The dataset download consists of three directories `train`, `val`, and   `test`.
Inside these, each directory corresponds to one sample how-to task which
consists of a sequence of screenshots and a JSON file containing the fields
described above.

## Sample datapoint

![UGIF dataset sample data point](https://storage.googleapis.com/gresearch/ugif/ugif-dataset-sample.png)

## How was the dataset collected?

The [Android support pages](https://support.google.com/) provide step-by-step
instructions for performing common tasks on Android. This is an example task:
"How to block unknown numbers?" for which the instruction text is "1. Open your
Phone app 2. Tap More. 3. Tap Settings and then Blocked numbers. 4. Turn on
Unknown". We crawl the Android support site and extract the how-to steps using
simple rules that look for ordered lists under a header since all the help pages
in the support site have the same structure. How-to instructions that require
physical actions outside the UI such as plugging in a HDMI cable are excluded.

We used the TaskMate platform for crowdsourcing annotations from taskers in
India, Kenya, and Mexico. The task description explained the purpose of this
data collection effort and how it would be used for research. To protect the
privacy of the taskers, the random TaskMate ID associated with each tasker is
not included in this dataset release. The task design was reviewed by
privacy, ethics, and legal committees. The price set for each task was based on
historical precedent on the TaskMate platform and in compliance with local laws.

For the valid queries, annotators were asked to translate the English how-to
query to one of 7 languages (Kannada, Marathi, Gujarati, Hindi, Bengali,
Swahili, Spanish) they are familiar with and to speak out loud the query.
Separate taskers were asked to translate the queries, validate the translations, speak the queries out loud, and validate the speech samples.

The instruction steps for each valid how-to query are parsed by annotators to a
sequence of macros (see the table below). We also asked annotators to highlight
spans in the instruction text to be given as argument to the macros.

| Macro               | Function                                             |
| ------------------- | ---------------------------------------------------- |
| tap(e)              | Taps on the UI element specified in the argument (e) |
| toggle(e, val=True) | Finds the UI element in the argument (e) and then toggles the nearest Switch element |
| home()              | Presses the home button in Android |
| back()              | Presses the back button |
| prompt(a)           | Requests the user to take some action (a) and waits until an action is performed |

For each how-to task, annotators are asked to operate a virtual Android device to carry out the steps in the how-to while the screen of the device and the annotator's actions are recorded. Just before each action taken by the annotator is forwarded to the virtual device and executed using the UIAutomator tool, we record a screenshot of the device, the view hierarchy in XML, and the action taken by the annotator at that step. We restrict the possible actions that the annotator can take at each step to:

- tapping on a UI element
- pressing the home button
- pressing the back button
- prompting the end-user for an input
- toggling a switch / checkbox
- scrolling up / down
- noting the completion of the task
- noting an error in the how-to instruction text and ending the recording before completion.

Although we can automate the extraction of how-tos from support sites, the manual annotation process for collecting UI screens from the Android emulator is difficult to scale. In particular, the data collection effort would scale linearly with the number of UI languages. To mitigate this, we collect UI screens from annotators only in English and search for each UI string in the resources directory of the app's APK and replace it with the translation provided by the developer in the APK wherever it is available. Note that not all strings in the UI have translations provided by the developer, so wherever a translation is unavailable, we default to English. A typical UI screen has a mixture of strings in English and other languages, but this is distinct from code mixing where two languages are used in a single sentence.

## Dataset statistics

| Dataset characteristic                                         | Value      |
| -------------------------------------------------------------- | ---------- |
| Total Number of how-to queries                                 | 523        |
| Number of train samples                                        | 152        |
| Number of dev samples                                          | 106        |
| Number of test samples                                         | 265        |
| Number of languages                                            | 8          |
| Total number of UI screens in the dataset                      | 3312       |
| Average number of UI screens per how-to                        | 6.3        |
| Percentage of how-tos with errors due to UI drift              | 29.9%      |
| Maximum number of tasks per tasker                             | 50         |

## Citing UGIF
Please use the following bibtex entry:

```
@article{gubbi2022ugif,
  title={UGIF: UI Grounded Instruction Following},
  author={Gubbi Venkatesh, Sagar and Talukdar, Partha and Narayanan, Srini},
  journal={arXiv preprint arXiv:2211.07615},
  year={2022},
  url={https://arxiv.org/abs/2211.07615}
}
```
# Querying OpenAI API for Labeling Instructions

Please first change the OpenAI key in ```query_openai_labelling.py``` to your own key.

The ```query_openai_labelling.py``` script takes a pickle (.pkl) file, and select the given number of instructions to label. We create 20 parallel processes to query the OpenAI GPT-3.5-Turbo API.


To prevent long sync waits, we only label 5000 instructions for each ```query_openai_labelling.py``` call. The script will automatically save the results, and pick unlabelled instructions when it is called again. The script ```batch_label_commands.py``` provides an end-to-end way to label all instructions inside a .pkl file.

Below is an example command.
```
python batch_label_commands.py --instruction_file [path to .pkl] --output_file [path of the output .pkl] --samples_to_label [how many instructions to label in total?]
```

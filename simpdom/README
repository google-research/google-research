# SimpDOM (simplified DOM Tree model for attribute extraction from web) code.
This directory maintains the model code for SimpDOM model which can be used to
extract attributes from webpages. For example, extraction name, height, weight,
team from a page about NBA player.

## Introduction
There has been a steady need to precisely extract structured knowledge from the
web (i.e. HTML documents). Given a web page, extracting a structured object
along with various attributes of interest
(e.g. price, publisher, author, and genre for a book) can facilitate a
variety of downstream applications such as large-scale knowledge
base construction, e-commerce product search, and personalized
recommendation. Considering each web page is rendered from an
HTML DOM tree, existing approaches formulate the problem as
a DOM tree node tagging task. However, they either rely on computationally
expensive visual feature engineering or are incapable
of modeling the relationship among the tree nodes. In this paper,
we propose a novel transferable method, Simplified DOM Trees
for Attribute Extraction (SimpDOM), to tackle the problem by efficiently
retrieving useful context for each node by leveraging the
tree structure. We study two challenging experimental settings:
(i) intra-vertical few-shot extraction, and (ii) cross-vertical few-shot
extraction, to evaluate our approach. Extensive experiments on the SWDE public
dataset show that SimpDOM outperforms the state-of-the-art (SOTA) method by
1.44% on the F1 score. We also find that utilizing knowledge from a
different vertical (cross-vertical extraction) is surprisingly useful
and helps beat the SOTA by a further 1.37%.

See details in full paper here: https://arxiv.org/abs/2101.02415

## Environment setup:
Please make sure you are in google_research/
```bash
sudo apt-get install python3-pip
sudo apt-get install cmake
python3 -m pip install --user virtualenv
python3 -m virtualenv ~/env
source ~/env/bin/activate
pip3 install -r simpdom/requirements.txt
```

## Usage
The python -m command assume you are in google_research/.

### 1. Download training data.
We use SWDE dataset which is originally proposed from the paper below:
"From One Tree to a Forest: a Unified Solution for Structured Web Data Extraction"
Qiang Hao, Rui Cai, Yanwei Pang, and Lei Zhang
in Proc. of the 34th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2011), pp.775-784, Beijing, China. July 24-28, 2011.

Please download the SWDE dataset from its official website:
https://archive.codeplex.com/?p=swde
The 'sourceCode' subdirectory contains all the information needed for
this project, we call the path to this directory /path/to/swde in the following
instruction.
* ./{vertical}: contains all the html for different websites of given verticals.
  There are 8 such verticals.
* ./groundtruth: contains groundtruth label for all the verticals.

To prepare /path/to/swde, please do:
curl \
https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip \
--output swde.zip

unzip swde.zip
cd sourceCode
unzip sourceCode.zip
cd sourceCode
7z x auto.7z
7z x book.7z
7z x camera.7z
7z x nbaplayer.7z
7z x university.7z
7z x restaurant.7z
7z x job.7z
7z x movie.7z
7z x groundtruth.7z
rm *7z
rm *zip

### 2. Pack data to a pickle file.
Please make sure you are in google_research/ for all the following steps.

python -m simpdom.pack_data \
--input_swde_path=/path/to/swde/ --output_pack_path=/path/to/swde/swde.pickle \
--first_n_pages=-1

### 3. Extract features and labels for DOM nodes and save them as json files.
Run for all the verticals and websites:
mkdir /path/to/swde/extracted_xpaths
python -m simpdom.extract_xpaths --input_groundtruth_path \
/path/to/swde/groundtruth/ \
--input_pickle_path \
/path/to/swde/swde.pickle \
--output_data_path \
/path/to/swde/extracted_xpaths \
--build_circle_features --max_depth_search 5 --max_xpath_dist 2 \
--max_friends_num 10

If you only want to run for a subset of the data, add the following flags to the
command above to limit vertical, website and number of pages to process:
--vertical auto --website autoweb --n_pages=34

Note that the vertical/website/n_pages have to exist in swde.pickle which is
created in the previous steps.

### 4. Generate vocabulary and embedding files.
We use glove's 6B 100d embeddings, which can be downloaded here:
https://nlp.stanford.edu/projects/glove/

The vocabularies are generated from the json file from previous steps which
contains tokenized text features. We generate both per-vertical and over-all
vocabularies/embeddings files.

python -m simpdom.process_domtree_data \
--domtree_path /path/to/swde/extracted_xpaths \
--word_embedding_path <glove_dir>/glove.6B.100d.txt \
--word_frequence_cutoff 3 --dim_word_glove 100

### 5. Download annotation type data.
Download goldmine_data.zip from gs://gresearch/simpdom/goldmine_data.zip and
unzip it. The data contains the annotation types for each node, e.g. date,
address, we call such annotation 'goldmine annotations'.

gsutil cp gs://gresearch/simpdom/goldmine_data.zip <your_dir>

### 6. Generate combined file for different seed number.

Seed number indicates the number of websites used for training. For example,
'auto' vertical contains 10 different websites. When seed_number=3, we use 3 of
the websites to train the model and the rest 7 websites to eval the model. We
choose 3 consecutive websites, there are 10 different set of such consecutive
websites. In this step, we concatenates all the training websites' data to one
file and the rest data to another file.

You can leave vertical as empty to run over all the verticals or set it to one
specific vertical.
python -m simpdom.prepare_data_for_seed_n \
--seed_num 2 \
--domtree_data_path /path/to/swde/extracted_xpaths \
--goldmine_data_path /path/to/goldmine \
--vertical auto

### 7. Train the model.
Source website should be '-' concatenated websites names for the given vertical,
e.g. it can be 'aol-kbb' for 'auto' vertical if you use 'seed_num=2' in step 6.
Usually you need to find the output files in --domtree_data_path from step 6 and
pick one of website combinations from the file names. Target website are
concatenated by '_', e.g. 'automotive_carquotes'. Each website will be processed
separately for eval. The eval results will be outputed in console directly and
saved in files in <result_path>/<vertical>/<one_target_website>-results/score/
<one_target_website>.metric.hit.constrained.voted.txt.

python -m simpdom.run_node_level_model --run train \
--vertical auto --source_website aol-kbb --target_website aol_kbb \
--result_path <result_path> \
--domtree_data_path /path/to/swde/extracted_xpaths \
--goldmine_data_path=/path/to/goldmine \
--batch_size 16 --epochs 1 --circle_features partner \
--semantic_encoder cos_sim --objective classification \
--vertical=auto --use_uniform_embedding=false --add_goldmine=false

### 8. Visualize nodes' relationship.
We also provide tools to visualize the nodes' relationship. Each node is vertex
in the graph. Two nodes are connected in the graph if one is in the neighborhood
of the other. See example results in Fig 2 of the paper linked above. You can
remove the "--vertical" setup if you want to visualize all the verticals.
python -m simpdom.visualize_field_graph \
--domtree_data_path=/path/to/swde/extracted_xpaths --vertical=auto


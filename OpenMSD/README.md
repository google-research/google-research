# OpenMSD: Open-access Multilingual Scientific Documents dataset

This repository has the scripts to build the OpenMSD dataset.

## Quick start

Run the command below

```python
bash run.sh
```

It will set up a virtual environment, download the necessary libraries, and create example papers and citation relations. The created papers can be found under *processed_data/example_papers/*, and the created citation, co-citation, and bibliographic-couple pairs can be found under *processed_data/example_citation_relations*. All scripts are tested with Python 3.10.9.

To build the full OpenMSD dataset, follow the steps below.

## Step 1: Download data

OpenMSD uses data from three sources.

* OpenCitations
  * Download the CSV files from [the COCI dataset](http://opencitations.net/download#coci), unzip the downloaded files and put the extracted CSV files under *raw_data/open_citations*. The total size of the extracted files is 240 GB (as of the December 2022 dump).
  * License: [CC0](http://opencitations.net/datasets)
* CrossRef Metadata
  * Download data from [this link](https://academictorrents.com/details/d9e554f4f0c3047d9f49e448a7004f7aa1701b69) using BitTorrent tools, and put the downloaded files under *raw_data/crossref_metadata*. The downloaded files are in the format of json.gz, and the total size is 185 GB (as of the April 2023 version). For more download options, please refer to [this page](https://www.crossref.org/blog/2023-public-data-file-now-available-with-new-and-improved-retrieval-options/).
  * License: [Facts or CC0](https://www.crossref.org/documentation/retrieve-metadata/rest-api/rest-api-metadata-license-information/)
* Unpaywall/OpenAlex data
  * Unpaywall.org [no longer provides snapshots for download](https://unpaywall.org/products/snapshot), but its data can be downloaded with OpenAlex. Download the data by following the instructions on [this page](https://docs.openalex.org/download-all-data/download-to-your-machine), and it will download a folder called *openalex-snapshot*. Move the downloaded folder to *raw_data/open_alex*. To total size is 316 GB (as of the 2023-05 version).
  * License: [CC0](https://openalex.s3.amazonaws.com/LICENSE.txt)

Example (dummy) data are provided under *raw_data*, in the same directory structures as the real data. Replace them with the downloaded and extracted data once available.

**NOTE**: By using the scripts in this repository, you accept the T&Cs of all the data sources above.

## Step 2: Get papers
Run the command below:

```python
python -m get_merge_paper_metadata \
  --crossref_input_pattern="./raw_data/crossref_metadata/*.json.gz" \
  --openalex_input_pattern="./raw_data/open_alex/openalex-snapshot/data/works/*/*.gz" \
  --output_path="./processed_data/example_papers/merged_paper_metadata.json"
```

It will merge the metadata from CrossRef and OpenAlex, and write the merged information to *processed_data/example_papers/merged_paper_metadata.json*. Each entry in the json file corresponds to one paper's information, including its DOI, title, venue, abstract, content, language, and category labels. Some fields may be empty (`null`) if they cannot be extracted from the data sources. If you only want to output the papers with abstract or content available, add flag *--only_output_papers_with_text*.

**NOTE 1**: By default, this script uses the [Apache Beam Direct Runner](https://beam.apache.org/documentation/runners/direct/) to process the data. It works fine with small-size data (e.g., the example provided in *run.sh*), but fails to process the full OpenAlex and CrossRef data. You may consider using more powerful runners, e.g., the [Spark Runner](https://beam.apache.org/documentation/runners/spark/), or running the script on [Google Cloud](https://cloud.google.com/dataflow/docs/guides/deploying-a-pipeline#python).

**NOTE 2**: For papers without abstract and content, you can find their URLs in the generated json file (in field *urls*), and use any third-party tools to download/scrape their contents therefrom.

## Step 3: Get Citation, Co-Citation and Bibliographic-Coupling pairs
Run the command below:

```python
python -m get_citation_relations \
  --metadata_input_pattern="./processed_data/example_papers/merged_paper_metadata.json*" \
  --opencitations_input_pattern="./raw_data/open_citations/*.csv" \
  --output_dir="./processed_data/example_citation_relations/"
```

It will create citation, co-citation, and bibliographic-coupling pairs, using the papers' metadata generated in step 2 and the pairs from OpenCitations. If you don't want it to generate the co-citation or bibliographic-coupling pairs, you can add flags *--no-want_cocitations* or *--no-want_bibcouples* when running the command, respectively.

**NOTE**: Similar to Step 2, this script also uses the [Apache Beam Direct Runner](https://beam.apache.org/documentation/runners/direct/) by default, which fails to process the full OpenCitations data. Follow the instructions in Step 2 to use more powerful runners.

## License
Apache License Version 2.0
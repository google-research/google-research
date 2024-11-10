# TableRAG: Million-Token Table Understanding with Language Models

*This is not an officially supported Google product.*


## Environment 

```shell
conda create --name tablerag python=3.10 -y
conda activate tablerag
pip install -r requirements.txt
```

## Data
Download datasets and pre-built databases from [here]().

```shell
tar zxvf arcade_qa.tar.gz
tar zxvf bird_qa.tar.gz
```

- `data/arcade/test.jsonl`: ArcadeQA test set
- `data/bird/test.jsonl`: BirdQA test set
- `db/arcade_test/`: Pre-built databases for ArcadeQA
- `db/bird_test/`: Pre-built databases for BirdQA

## Code Structure
- `build_db.py`: Builds databases for retrieval for each table in a dataset
- `run.py`: Main script to execute experiments
- `evaluate.py`: Evaluates the results stored in the output directory
- `agent/agent.py`: Base class for baseline agents like ReadTable, ReadSchema, RandRowSampling, and RowColRetrieval
- `agent/rag_agent.py`: Implementation of the TableRAG agent
- `agent/model.py`: Manages calls to LLM APIs from OpenAI, VertexAI, and HuggingFace
- `agent/retriever.py`: Handles building databases and performs schema/cell/row/column retrieval

## Usage

### Command Arguments

- `--dataset_path`: Path to the dataset, default: 'data/tabfact/test_sub_nosynth.jsonl'
- `--model_name`: Name of the model, default: `gpt-3.5-turbo-0125`, options: `text-bison@001`, `text-bison@002`, `text-unicorn@001`, `gemini-pro`, `gemini-ultra`, `gemini-1.5-pro-preview-0409`, `gpt-3.5-turbo-0125`, `gpt-4-0125-preview`, `gpt-4-turbo-2024-04-09`
- `--agent_type`: Type of agent, default: `PyReAct` (ReadTable), options: `PyReAct`, `ReadSchema`, `RandSampling` (RandRowSampling), `TableSampling` (RowColRetrieval), `TableRAG`
- `--embed_model_name`: Name of the embedding model, default: `text-embedding-3-large`, options: `text-embedding-x-xxx` (OpenAI), `textembedding-gecko@xxx` (VertexAI), others (HuggingFace)
- `--log_dir`: Directory for logs, default: 'output/test/'
- `--db_dir`: Directory for databases, default: 'db/'
- `--top_k`: Number of retrieval results, default: 5
- `--sc`: Self-consistency, default: 1
- `--max_encode_cell`: Cell encoding budget $B$
- `--stop_at`: Stopping point, default: -1 means no specific stop
- `--resume_from`: Point to start/resume from, default: 0
- `--load_exist`: Load existing results, default: False
- `--n_worker`: Number of workers, default: 1
- `--verbose`: Verbose output, default: False  

### Examples

1. Build databases for each table with a specific cell encoding budget (skip if using downloaded pre-built databases):

```shell
python build_db.py --dataset_path data/arcade/test.jsonl --max_encode_cell 10000
```

2. Execute the first 5 cases with verbose output:

```shell
# For ReadTable agent
python run.py --stop_at 5 --verbose
# For TableRAG agent
python run.py --agent_type TableRAG --stop_at 5 --verbose
```

3. Run and evaluate TableRAG on the ArcadeQA dataset:

```shell
python run.py \
--dataset_path data/arcade/test.jsonl \
--model_name gpt-3.5-turbo-0125 \
--agent_type TableRAG \
--log_dir 'output/arcade_gpt3.5_tablerag' \
--top_k 5 \
--sc 10 \
--max_encode_cell 100 \
--n_worker 16
```

## Run vLLM Server (for models not hosted by OpenAI or VertexAI)

```shell
# To start the vLLM server
python vllm_server.py $MODEL_NAME
```
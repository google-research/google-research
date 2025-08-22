# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import traceback
import time

import pandas as pd
from elasticsearch import Elasticsearch
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain.schema import Document
from langchain_core.rate_limiters import InMemoryRateLimiter
import google.generativeai as genai

def setup_gemini_api_key():
    """Sets up the Gemini API key.

    Reads the API key from the environment variable 'GOOGLE_API_KEY'.
    If not found, prompts the user to enter the key and sets the environment variable.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API key: ")
        os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

def extract_pdf_text(file_path):
    """
    Extracts text from PDF and chunks it to prepare for embedding generation

    Args:
        file_path (str): File path for the PDF

    Returns:
        PDF text in chunked docs
    """

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return "\n".join([doc.page_content for doc in docs])

def generate_embeddings(docs, embedding_model, rate_limit=140, time_window=60):
    """Generates embeddings for the documents.

    Args:
        docs (list): List of documents to generate embeddings for.
        embedding_model: The embedding model to use.
        rate_limit (int, optional): Maximum number of requests before pausing, else Gemini will start error'ing out. Defaults to 140.
        time_window (int, optional): Time to pause in seconds after reaching the rate limit. Defaults to 60.

    Returns:
        list: A list of embeddings for the documents.
    """
    embeddings = []
    request_count = 0
    for doc in tqdm(docs, desc="Generating embeddings"):
        if request_count >= rate_limit:
            time.sleep(time_window)
            request_count = 0
        try:
            embeddings.append(embedding_model.embed_query(doc))
            request_count += 1
        except Exception as e:
            print(f"Error generating embedding for document: {e}")
    return embeddings

def index_papers(
    es_host,
    es_port,
    index_name,
    data_file_path,
    paper_path,
):
    """
    Indexes research paper metadata into Elasticsearch.

    Args:
        es_host (str): Elasticsearch host.
        es_port (int): Elasticsearch port.
        index_name (str): Name of the Elasticsearch index.
        data_file_path (str): Path to the pickle file containing paper metadata.
        paper_path (str): Base path for accessing paper PDF files.
    """

    try:
        df = pd.read_pickle(data_file_path)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    df.drop_duplicates(subset=["uuid"], inplace=True)
    df.dropna(subset=["uuid"], inplace=True)

    # Connect to Elasticsearch
    try:
        es = Elasticsearch([{"host": es_host, "port": es_port, "scheme": "http"}])
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        return


    embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="retrieval_document",
    )


    # Create index with dense_vector mapping
    if not es.indices.exists(index=index_name):
        try:
            es.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "uuid": {"type": "keyword"},
                            "paper_title": {"type": "text"},
                            "abstract": {"type": "text"},
                            "authors": {"type": "text"},
                            "topic_keywords": {"type": "keyword"},
                            "topic_categories": {"type": "keyword"},
                            "sdgs": {"type": "keyword"},
                            "pdf_text": {"type": "text"},
                            "abstract_embedding": {
                                "type": "dense_vector",
                                "dims": 768,
                                "index": True,
                                "similarity": "cosine",
                            },
                            "pdf_text_embedding": {
                                 "type": "dense_vector",
                                 "dims": 768,
                                 "index": True,
                                 "similarity": "cosine",
                            },
                            "models": {"type": "text"},
                        }
                    }
                },
            )
        except Exception as e:
            print(f"Error creating Elasticsearch index: {e}")
            return

    print(df.columns)

    # Index data into Elasticsearch
    for index, row in df.iterrows():
        try:
            paper_path_full = os.path.join(paper_path, row["paper_path"])
            print("Processing: ", paper_path_full)

            pdf_text = extract_pdf_text(paper_path_full)

            # Generate embeddings
            abstract_embedding, pdf_text_embedding = generate_embeddings([row['abstract'].replace('\udbc0', ' '), pdf_text.replace('\udbc0', ' ')], embedding_model)

            models = None
            if not pd.isna(row["transformed_models"]):
                models = json.dumps(json.loads(row["transformed_models"]))

            topic_categories = []
            if isinstance(row['LLM'], list):
                topic_categories = [item for item in row['LLM'] if item]

            topic_keywords = []
            if isinstance(row['MMR'], list):
                topic_keywords = [x.strip() for x in row["MMR"]]


            document = {
                "uuid": row["uuid"],
                "paper_title": row["title"],
                "abstract": row["abstract"],
                "authors": row["authors"],
                "topic_keywords": topic_keywords,
                "topic_categories": topic_categories,
                "sdgs": row["sdgs"],
                "models": models,
                "pdf_text": pdf_text,
                "abstract_embedding": abstract_embedding,
                "pdf_text_embedding": pdf_text_embedding,
            }
            es.index(index=index_name, id=index, document=document)
        except Exception as e:
            print(f"Error processing {row['paper_path']}: {e}")
            print(row)
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index research papers into Elasticsearch."
    )
    parser.add_argument(
        "--es_host", type=str, default="localhost", help="Elasticsearch host"
    )
    parser.add_argument("--es_port", type=int, default=9200, help="Elasticsearch port")
    parser.add_argument(
        "--index_name", type=str, default="sd_papers_index", help="Index name"
    )
    parser.add_argument(
        "--data_file_path",
        type=str,
        required=True,
        help="Path to the pickle file containing paper metadata",
    )
    parser.add_argument(
        "--paper_path",
        type=str,
        required=True,
        help="Base path for accessing paper files",
    )
    args = parser.parse_args()

    setup_gemini_api_key()

    index_papers(
        args.es_host,
        args.es_port,
        args.index_name,
        args.data_file_path,
        args.paper_path,
    )

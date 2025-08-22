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

# Imports
import os
import time
import argparse
import numpy as np
import pandas as pd
import google.generativeai as genai
import hdbscan

from tqdm import tqdm
from bertopic import BERTopic
from bertopic.representation import LangChain
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from scipy.cluster import hierarchy as sch
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.rate_limiters import InMemoryRateLimiter

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

def load_and_preprocess_data(input_file, paper_content_to_use, papers_dir_path=None):
    """Loads and preprocesses the data.

    Args:
        input_file (str): Path to the input pickle file containing paper metadata.
        paper_content_to_use (str): Whether to use 'abstract_text' or 'full_paper_text'
                                     for embeddings and topic modeling.
        papers_dir_path (str, optional): Path to the directory containing PDF paper files.
                                          Required if paper_content_to_use is 'full_paper_text'.

    Returns:
        tuple: A tuple containing the DataFrame with paper metadata and a list of preprocessed documents.

    Raises:
        SystemExit: If there is an error reading the input file.
        ValueError: If paper_content_to_use is invalid or if papers_dir_path is not provided
                    when using 'full_paper_text'.
    """
    try:
        df = pd.read_pickle(input_file)
    except Exception as e:
        raise SystemExit(f"Error reading input file: {e}")

    docs = []
    if paper_content_to_use == "full_paper_text":
        if not papers_dir_path:
            raise ValueError("papers_dir_path must be provided when using full_paper_text")
        if not os.path.exists(papers_dir_path):
            raise ValueError(f"Invalid papers_dir_path: {papers_dir_path}")

        for _, row in df.iterrows():
            paper_path = os.path.join(papers_dir_path, row["paper_path"])
            try:
                loader = PyPDFLoader(paper_path)
                documents = loader.load()
                paper_text = "\n".join([doc.page_content for doc in documents])
                docs.append(f"{row['paper_title']}\n{paper_text}")
            except Exception as e:
                print(f"Error processing {paper_path}: {e}")
    elif paper_content_to_use == "abstract_text":
        for _, row in df.iterrows():
            docs.append(f"{row['title']}\n{row['abstract']}")
    else:
        raise ValueError(f"Invalid paper_content_to_use value: {paper_content_to_use}")
    return df, docs

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

def create_and_train_topic_model(embeddings, docs):
    """Creates and trains the topic model.

    Args:
        embeddings (list): List of document embeddings.
        docs (list): List of documents.

    Returns:
        BERTopic: The trained topic model.
    """
    # Reduce dimensionality
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )

    # Cluster reduced embeddings
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Tokenize topics
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

    # Fine-tune topic representation
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.1, check_every_n_seconds=0.1, max_bucket_size=10
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        n=1,
        safety_settings={
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
        rate_limiter=rate_limiter,
    )
    chain = load_qa_chain(llm)
    prompt = "Given the associated [KEYWORDS], what are the preceding documents about? Provide a single topic label in less than 5 words."
    representation_model = {
        "MMR": MaximalMarginalRelevance(diversity=0.7),
        "LLM": LangChain(
            chain, prompt=prompt, nr_docs=3, doc_length=300, tokenizer="vectorizer"
        ),
    }

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(
        ctfidf_model=ctfidf_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=True,
        nr_topics="auto",
    )

    # Train model
    topics, probs = topic_model.fit_transform(docs, np.array(embeddings))

    return topic_model

def save_and_visualize_topic_model(topic_model, output_file):
    """Saves and visualizes the topic model.

    Args:
        topic_model (BERTopic): The trained topic model.
        output_file (str): Filename for saving the topic model output.
    """
    topic_model.save(output_file)
    topic_model_info = topic_model.get_topic_info()
    print(topic_model_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process papers dataset.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input pickle file containing paper metadata.",
    )
    parser.add_argument(
        "--paper_content_to_use",
        choices=["abstract_text", "full_paper_text"],
        default="abstract_text",
        help="Whether to use abstracts or full paper text for embeddings and topic modeling.",
    )
    parser.add_argument(
        "--papers_dir_path",
        type=str,
        help="Path to the directory containing PDF paper files (required if using full_paper_text).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="topic_model_output",
        help="Filename for saving the topic model output.",
    )
    args = parser.parse_args()

    setup_gemini_api_key()

    df, docs = load_and_preprocess_data(
        args.input_file, args.paper_content_to_use, args.papers_dir_path
    )

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", task_type="clustering", output_dimensionality=100
    )
    embeddings = generate_embeddings(docs, embedding_model)

    topic_model = create_and_train_topic_model(embeddings, docs)

    save_and_visualize_topic_model(topic_model, args.output_file)

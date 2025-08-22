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

import os
from typing import Optional, List, Any
from collections import Counter

import numpy as np
import pandas as pd
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings


class Retriever:
    def __init__(self, agent_type, mode, embed_model_name, top_k = 5, max_encode_cell = 10000, db_dir = 'db/', verbose = False):
        self.agent_type = agent_type
        self.mode = mode
        self.embed_model_name = embed_model_name
        self.schema_retriever = None
        self.cell_retriever = None
        self.row_retriever = None
        self.column_retriever = None
        self.top_k = top_k
        self.max_encode_cell = max_encode_cell
        self.db_dir = db_dir
        self.verbose = verbose
        os.makedirs(db_dir, exist_ok=True)

        if self.mode == 'bm25':
            self.embedder = None
        elif 'text-embedding' in self.embed_model_name:
            self.embedder = OpenAIEmbeddings(model=self.embed_model_name)
        elif 'gecko' in self.embed_model_name: # VertexAI
            self.embedder = VertexAIEmbeddings(model_name=self.embed_model_name)
        else:
            self.embedder = HuggingFaceEmbeddings(model_name=self.embed_model_name)

    def init_retriever(self, table_id, df):
        self.df = df
        if 'TableRAG' in self.agent_type:
            self.schema_retriever = self.get_retriever('schema', table_id, self.df)
            self.cell_retriever = self.get_retriever('cell', table_id, self.df)
        elif self.agent_type == 'TableSampling':
            max_row = max(1, self.max_encode_cell // 2 // len(self.df.columns))
            self.df = self.df.iloc[:max_row]
            self.row_retriever = self.get_retriever('row', table_id, self.df)
            self.column_retriever = self.get_retriever('column', table_id, self.df)

    def get_retriever(self, data_type, table_id, df):
        docs = None
        if self.mode == 'embed' or self.mode == 'hybrid':
            db_dir = os.path.join(self.db_dir, f'{data_type}_db_{self.max_encode_cell}_' + table_id)
            if os.path.exists(db_dir):
                if self.verbose:
                    print(f'Load {data_type} database from {db_dir}')
                db = FAISS.load_local(db_dir, self.embedder, allow_dangerous_deserialization=True)
            else:
                docs = self.get_docs(data_type, df)
                db = FAISS.from_documents(docs, self.embedder)
                db.save_local(db_dir)
            embed_retriever = db.as_retriever(search_kwargs={'k': self.top_k})
        if self.mode == 'bm25' or self.mode == 'hybrid':
            if docs is None:
                docs = self.get_docs(data_type, df)
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = self.top_k
        if self.mode == 'hybrid':
            # return EnsembleRetriever(retrievers=[embed_retriever, bm25_retriever], weights=[0.9, 0.1])
            return EnsembleRetriever(retrievers=[embed_retriever, bm25_retriever], weights=[0.5, 0.5])
        elif self.mode == 'embed':
            return embed_retriever
        elif self.mode == 'bm25':
            return bm25_retriever

    def get_docs(self, data_type, df):
        if data_type == 'schema':
            return self.build_schema_corpus(df)
        elif data_type == 'cell':
            return self.build_cell_corpus(df)
        elif data_type == 'row':
            return self.build_row_corpus(df)
        elif data_type == 'column':
            return self.build_column_corpus(df)

    def build_schema_corpus(self, df):
        docs = []
        for col_name, col in df.items():
            if col.dtype != 'object' and col.dtype != str:
                result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "min": {col.min()}, "max": {col.max()}}}'
            else:
                most_freq_vals = col.value_counts().index.tolist()
                example_cells = most_freq_vals[:min(3, len(most_freq_vals))]
                result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "cell_examples": {example_cells}}}'
            docs.append(Document(page_content=col_name, metadata={'result_text': result_text}))
        return docs

    def build_cell_corpus(self, df):
        docs = []
        categorical_columns = df.columns[(df.dtypes == 'object') | (df.dtypes == str)]
        other_columns = df.columns[~(df.dtypes == 'object') | (df.dtypes == str)]
        if len(other_columns) > 0:
            for col_name in other_columns:
                col = df[col_name]
                docs.append(f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "min": {col.min()}, "max": {col.max()}}}')
        if len(categorical_columns) > 0:
            cell_cnt = Counter(df[categorical_columns].apply(lambda x: '{"column_name": "' + x.name + '", "cell_value": "' + x.astype(str) + '"}').values.flatten())
            docs += [cell for cell, _ in cell_cnt.most_common(self.max_encode_cell - len(docs))]
        docs = [Document(page_content=doc) for doc in docs]
        return docs

    def build_row_corpus(self, df):
        row_docs = []
        for row_id, (_, row) in enumerate(df.iterrows()):
            row_text = '|'.join(str(cell) for cell in row)
            row_doc = Document(page_content=row_text, metadata={'row_id': row_id})
            row_docs.append(row_doc)
        return row_docs

    def build_column_corpus(self, df):
        col_docs = []
        for col_id, (_, column) in enumerate(df.items()):
            col_text = '|'.join(str(cell) for cell in column)
            col_doc = Document(page_content=col_text, metadata={'col_id': col_id})
            col_docs.append(col_doc)
        return col_docs

    def retrieve_schema(self, query):
        results = self.schema_retriever.invoke(query)
        observations = [doc.metadata['result_text'] for doc in results]
        return observations

    def retrieve_cell(self, query):
        results = self.cell_retriever.invoke(query)
        observations = [doc.page_content for doc in results]
        return observations

    def sample_rows_and_columns(self, query):
        # Apply row sampling
        row_results = self.row_retriever.invoke(query)
        row_ids = sorted([doc.metadata['row_id'] for doc in row_results])
        # Apply column sampling
        col_results = self.column_retriever.invoke(query)
        col_ids = sorted([doc.metadata['col_id'] for doc in col_results])
        # Return sampled rows and columns
        return self.df.iloc[row_ids, col_ids]
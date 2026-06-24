# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""This code implements a KNN service and metadata provider.

Given a query image, text, or embeddings, we find nearest neighbors and return
their indices, distances, embeddings, and other metadata.

The code below is based on https://github.com/rom1504/clip-retrieval/
"""

import base64
import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
import io
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import ssl
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import urllib.request
import urllib.error

import faiss
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("agile_modeling")

# Constants
DEFAULT_TIMEOUT = 10
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
)
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


class KNNServiceError(Exception):
    """Base exception for KNN service errors."""
    pass


class InvalidInputError(KNNServiceError):
    """Raised when invalid input is provided."""
    pass


class ResourceLoadError(KNNServiceError):
    """Raised when resources fail to load."""
    pass


def convert_metadata_to_base64(meta: Optional[Dict[str, Any]]) -> None:
    """Converts the image at a path to the Base64 string and sets it as the image.

    If there is no `image_path` key present in the metadata dictionary, the
    function will have no effect.

    Args:
        meta: metadata dictionary.
    """
    if meta is None or "image_path" not in meta:
        return
    
    path = meta["image_path"]
    if not os.path.exists(path):
        LOGGER.warning(f"Image path does not exist: {path}")
        return
    
    try:
        with Image.open(path) as img:
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            meta["image"] = img_str
    except Exception as e:
        LOGGER.error(f"Failed to convert image to base64: {path}, error: {e}")


def download_image(url: str, timeout: int = DEFAULT_TIMEOUT) -> BytesIO:
    """Download an image from a url and return a byte stream.
    
    Args:
        url: The URL to download from.
        timeout: Request timeout in seconds.
        
    Returns:
        BytesIO stream containing the image data.
        
    Raises:
        KNNServiceError: If download fails.
    """
    if not url or not isinstance(url, str):
        raise InvalidInputError("Invalid URL provided")
    
    try:
        urllib_request = urllib.request.Request(
            url,
            data=None,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        urllib_context = ssl.create_default_context()
        urllib_context.set_alpn_protocols(["http/1.1"])

        with urllib.request.urlopen(
            urllib_request, timeout=timeout, context=urllib_context
        ) as response:
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > MAX_IMAGE_SIZE:
                raise KNNServiceError(f"Image too large: {content_length} bytes")
            
            data = response.read()
            if len(data) > MAX_IMAGE_SIZE:
                raise KNNServiceError(f"Image too large: {len(data)} bytes")
            
            img_stream = BytesIO(data)
        return img_stream
    except urllib.error.URLError as e:
        raise KNNServiceError(f"Failed to download image from {url}: {e}")
    except Exception as e:
        raise KNNServiceError(f"Unexpected error downloading image: {e}")


def normalized(a: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:
    """Normalize array along specified axis.
    
    Args:
        a: Input array.
        axis: Axis along which to normalize.
        order: Order of the norm.
        
    Returns:
        Normalized array.
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def validate_embedding(embedding: Union[List, np.ndarray]) -> np.ndarray:
    """Validate and convert embedding to proper format.
    
    Args:
        embedding: Input embedding.
        
    Returns:
        Validated numpy array.
        
    Raises:
        InvalidInputError: If embedding is invalid.
    """
    try:
        arr = np.array(embedding, dtype=np.float32)
        if arr.ndim != 1:
            raise InvalidInputError(f"Embedding must be 1D, got {arr.ndim}D")
        if arr.size == 0:
            raise InvalidInputError("Embedding cannot be empty")
        if not np.isfinite(arr).all():
            raise InvalidInputError("Embedding contains non-finite values")
        return arr
    except (ValueError, TypeError) as e:
        raise InvalidInputError(f"Invalid embedding format: {e}")


class ThreadSafeCounter:
    """Thread-safe counter for generating unique IDs."""
    
    def __init__(self, start: int = 0):
        self._value = start
        self._lock = threading.Lock()
    
    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value


class KnnService:
    """The KNN service provides nearest neighbors given text or image."""

    def __init__(self, clip_resources: Dict[str, 'ClipResource'], **kwargs):
        """Initialize KNN service.
        
        Args:
            clip_resources: Dictionary of clip resources.
            **kwargs: Additional configuration options.
        """
        if not clip_resources:
            raise InvalidInputError("clip_resources cannot be empty")
        
        self.clip_resources = clip_resources
        self._lock = threading.RLock()
        self._request_counter = ThreadSafeCounter()
        
        # Configuration
        self.default_threshold = kwargs.get('dedup_threshold', 0.94)
        self.max_results = kwargs.get('max_results', 10000)

    def compute_query(
        self,
        clip_resource: 'ClipResource',
        text_input: Optional[str] = None,
        image_input: Optional[str] = None,
        image_url_input: Optional[str] = None,
        embedding_input: Optional[Union[List, np.ndarray]] = None,
        use_mclip: bool = False,
    ) -> np.ndarray:
        """Computes the query embedding.
        
        Args:
            clip_resource: The CLIP resource to use.
            text_input: Text query.
            image_input: Base64 encoded image.
            image_url_input: Image URL.
            embedding_input: Direct embedding input.
            use_mclip: Whether to use multilingual CLIP.
            
        Returns:
            Query embedding as numpy array.
            
        Raises:
            InvalidInputError: If input is invalid.
            KNNServiceError: If computation fails.
        """
        try:
            import torch
        except ImportError:
            raise ResourceLoadError("PyTorch is required but not installed")

        request_id = self._request_counter.next()
        LOGGER.debug(f"Computing query embedding (request {request_id})")

        if text_input is not None and text_input.strip():
            return self._compute_text_embedding(
                clip_resource, text_input.strip(), use_mclip
            )
        elif image_input is not None or image_url_input is not None:
            return self._compute_image_embedding(
                clip_resource, image_input, image_url_input
            )
        elif embedding_input is not None:
            validated_embedding = validate_embedding(embedding_input)
            return np.expand_dims(validated_embedding, 0)
        else:
            raise InvalidInputError(
                "One of text_input, image_input, image_url_input, or embedding_input must be provided"
            )

    def _compute_text_embedding(
        self, 
        clip_resource: 'ClipResource', 
        text: str, 
        use_mclip: bool
    ) -> np.ndarray:
        """Compute embedding for text input."""
        try:
            import torch
            
            if use_mclip:
                if clip_resource.model_txt_mclip is None:
                    raise KNNServiceError("MCLIP model not available")
                return normalized(clip_resource.model_txt_mclip(text))
            else:
                if clip_resource.tokenizer is None or clip_resource.model is None:
                    raise KNNServiceError("CLIP model or tokenizer not available")
                
                tokens = clip_resource.tokenizer([text]).to(clip_resource.device)
                with torch.no_grad():
                    text_features = clip_resource.model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu().to(torch.float32).detach().numpy()
        except Exception as e:
            raise KNNServiceError(f"Failed to compute text embedding: {e}")

    def _compute_image_embedding(
        self, 
        clip_resource: 'ClipResource', 
        image_input: Optional[str], 
        image_url_input: Optional[str]
    ) -> np.ndarray:
        """Compute embedding for image input."""
        try:
            import torch
            
            if clip_resource.model is None or clip_resource.preprocess is None:
                raise KNNServiceError("CLIP model or preprocessor not available")

            # Get image data
            if image_input is not None:
                try:
                    binary_data = base64.b64decode(image_input)
                    img_data = BytesIO(binary_data)
                except Exception as e:
                    raise InvalidInputError(f"Invalid base64 image data: {e}")
            elif image_url_input is not None:
                img_data = download_image(image_url_input)
            else:
                raise InvalidInputError("No image input provided")

            # Process image
            try:
                with Image.open(img_data) as img:
                    # Convert to RGB if necessary
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')
                    
                    prepro = (
                        clip_resource.preprocess(img)
                        .unsqueeze(0)
                        .to(clip_resource.device)
                    )
            except Exception as e:
                raise InvalidInputError(f"Invalid image format: {e}")

            with torch.no_grad():
                image_features = clip_resource.model.encode_image(prepro)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().to(torch.float32).detach().numpy()
        except Exception as e:
            if isinstance(e, (InvalidInputError, KNNServiceError)):
                raise
            raise KNNServiceError(f"Failed to compute image embedding: {e}")

    def connected_components_dedup(
        self, 
        embeddings: np.ndarray, 
        threshold: float = None
    ) -> List[int]:
        """Find duplicates using connected components approach.
        
        Args:
            embeddings: Array of embeddings to deduplicate.
            threshold: Similarity threshold for considering duplicates.
            
        Returns:
            List of indices to remove.
        """
        if threshold is None:
            threshold = self.default_threshold
            
        if embeddings.shape[0] <= 1:
            return []

        try:
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            
            # Use range search to find similar embeddings
            lims, _, indices = index.range_search(embeddings, threshold)
            
            same_mapping = defaultdict(list)
            for i in range(embeddings.shape[0]):
                for j in indices[lims[i]:lims[i + 1]]:
                    if i != j:  # Don't include self-matches
                        same_mapping[int(i)].append(int(j))

            # Find connected components
            groups = self._connected_components(same_mapping)
            
            # Keep first item from each group, mark others for removal
            non_uniques = set()
            for group in groups:
                if len(group) > 1:
                    for idx in group[1:]:
                        non_uniques.add(idx)

            return list(non_uniques)
        except Exception as e:
            LOGGER.error(f"Deduplication failed: {e}")
            return []

    def _connected_components(self, neighbors: Dict[int, List[int]]) -> List[List[int]]:
        """Find connected components in the graph."""
        seen = set()

        def component(node: int) -> List[int]:
            result = []
            nodes = {node}
            while nodes:
                current = nodes.pop()
                if current in seen:
                    continue
                seen.add(current)
                nodes.update(set(neighbors.get(current, [])) - seen)
                result.append(current)
            return result

        components = []
        for node in neighbors:
            if node not in seen:
                comp = component(node)
                if comp:
                    components.append(comp)
        return components

    def post_filter(
        self, 
        embeddings: np.ndarray, 
        deduplicate: bool
    ) -> Set[int]:
        """Post filter results by deduplicating.
        
        Args:
            embeddings: Result embeddings.
            deduplicate: Whether to perform deduplication.
            
        Returns:
            Set of indices to remove.
        """
        to_remove = set()
        if deduplicate and embeddings.shape[0] > 1:
            to_remove = set(self.connected_components_dedup(embeddings))
        return to_remove

    def knn_search(
        self,
        query: np.ndarray,
        modality: str,
        num_result_ids: int,
        clip_resource: 'ClipResource',
        deduplicate: bool
    ) -> Tuple[List[float], List[int], List[np.ndarray]]:
        """Compute the KNN search.
        
        Args:
            query: Query embedding.
            modality: Search modality ('image' or 'text').
            num_result_ids: Number of results to retrieve.
            clip_resource: CLIP resource to use.
            deduplicate: Whether to deduplicate results.
            
        Returns:
            Tuple of (distances, indices, embeddings).
            
        Raises:
            KNNServiceError: If search fails.
        """
        if modality not in ('image', 'text'):
            raise InvalidInputError(f"Invalid modality: {modality}")
        
        # Select appropriate index
        index = (
            clip_resource.image_index 
            if modality == "image" 
            else clip_resource.text_index
        )
        
        if index is None:
            raise KNNServiceError(f"No {modality} index available")

        try:
            # Clamp num_result_ids to reasonable bounds
            num_result_ids = min(max(1, num_result_ids), self.max_results)
            
            distances, indices, embeddings = index.search_and_reconstruct(
                query, num_result_ids
            )
            
            # Process results
            results = indices[0]
            valid_mask = results != -1
            
            if not valid_mask.any():
                return [], [], []
            
            result_indices = results[valid_mask]
            result_distances = distances[0][valid_mask]
            result_embeddings = embeddings[0][valid_mask]
            result_embeddings = normalized(result_embeddings)
            
            # Apply post-filtering
            if deduplicate:
                local_indices_to_remove = self.post_filter(result_embeddings, deduplicate)
                indices_to_remove = {result_indices[i] for i in local_indices_to_remove}
                
                # Filter results
                final_indices = []
                final_distances = []
                final_embeddings = []
                seen_indices = set()
                
                for idx, dist, emb in zip(result_indices, result_distances, result_embeddings):
                    if idx not in indices_to_remove and idx not in seen_indices:
                        seen_indices.add(idx)
                        final_indices.append(idx)
                        final_distances.append(dist)
                        final_embeddings.append(emb)
                
                return final_distances, final_indices, final_embeddings
            else:
                return (
                    result_distances.tolist(),
                    result_indices.tolist(),
                    result_embeddings.tolist()
                )
                
        except Exception as e:
            raise KNNServiceError(f"KNN search failed: {e}")

    def map_to_metadata(
        self,
        indices: List[int],
        distances: List[float],
        embeddings: List[np.ndarray],
        num_images: int,
        metadata_provider: 'BaseMetadataProvider'
    ) -> List[Dict[str, Any]]:
        """Map the indices to the metadata.
        
        Args:
            indices: Result indices.
            distances: Result distances.
            embeddings: Result embeddings.
            num_images: Number of images to return.
            metadata_provider: Metadata provider instance.
            
        Returns:
            List of result dictionaries with metadata.
        """
        if not indices:
            return []
        
        results = []
        try:
            # Limit to requested number of images
            limited_indices = indices[:num_images]
            metas = metadata_provider.get(limited_indices)
            
            for i, (dist, idx, emb) in enumerate(zip(distances, indices, embeddings)):
                if i >= num_images:
                    break
                    
                output = {}
                meta = metas[i] if i < len(metas) else None
                
                # Convert metadata to base64 if needed
                if meta is not None:
                    convert_metadata_to_base64(meta)
                    output.update(meta_to_dict(meta))
                
                output["id"] = int(idx)
                output["similarity"] = float(dist)
                output["embedding"] = emb.tolist() if isinstance(emb, np.ndarray) else emb
                results.append(output)
                
        except Exception as e:
            LOGGER.error(f"Failed to map metadata: {e}")
            # Return basic results without metadata
            for i, (dist, idx, emb) in enumerate(zip(distances, indices, embeddings)):
                if i >= num_images:
                    break
                results.append({
                    "id": int(idx),
                    "similarity": float(dist),
                    "embedding": emb.tolist() if isinstance(emb, np.ndarray) else emb
                })
        
        return results

    def query(
        self,
        text_input: Optional[str] = None,
        image_input: Optional[str] = None,
        image_url_input: Optional[str] = None,
        embedding_input: Optional[Union[List, np.ndarray]] = None,
        modality: str = "image",
        num_images: int = 100,
        num_result_ids: int = 100,
        indice_name: Optional[str] = None,
        use_mclip: bool = False,
        deduplicate: bool = True,
    ) -> List[Dict[str, Any]]:
        """Implements the querying functionality of the knn service.

        Args:
            text_input: The text input. Only one input must be set.
            image_input: The image input. Only one input must be set.
            image_url_input: The image input by url. Only one input must be set.
            embedding_input: An embedding input.
            modality: The modality ('image' or 'text').
            num_images: The number of nearest neighbors to return.
            num_result_ids: The number of result ids to retrieve initially.
            indice_name: The index name to use.
            use_mclip: Whether to use mclip.
            deduplicate: Whether to deduplicate results.

        Returns:
            List of dictionaries containing metadata of the nearest neighbors.
            
        Raises:
            InvalidInputError: If input parameters are invalid.
            KNNServiceError: If query processing fails.
        """
        with self._lock:
            # Validate inputs
            input_count = sum(
                x is not None 
                for x in [text_input, image_input, image_url_input, embedding_input]
            )
            if input_count == 0:
                raise InvalidInputError(
                    "Must provide one of: text_input, image_input, image_url_input, or embedding_input"
                )
            if input_count > 1:
                raise InvalidInputError(
                    "Only one input type should be provided"
                )
            
            # Validate parameters
            if num_images <= 0:
                raise InvalidInputError("num_images must be positive")
            if num_result_ids <= 0:
                raise InvalidInputError("num_result_ids must be positive")
            
            # Select index
            if indice_name is None:
                indice_name = next(iter(self.clip_resources.keys()))
            elif indice_name not in self.clip_resources:
                raise InvalidInputError(f"Unknown index: {indice_name}")

            clip_resource = self.clip_resources[indice_name]

            try:
                # Compute query embedding
                query = self.compute_query(
                    clip_resource=clip_resource,
                    text_input=text_input,
                    image_input=image_input,
                    image_url_input=image_url_input,
                    embedding_input=embedding_input,
                    use_mclip=use_mclip,
                )
                
                # Perform KNN search
                distances, indices, embeddings = self.knn_search(
                    query=query,
                    modality=modality,
                    num_result_ids=num_result_ids,
                    clip_resource=clip_resource,
                    deduplicate=deduplicate,
                )
                
                if not distances:
                    return []
                
                # Map to metadata
                results = self.map_to_metadata(
                    indices=indices,
                    distances=distances,
                    embeddings=embeddings,
                    num_images=num_images,
                    metadata_provider=clip_resource.metadata_provider,
                )
                
                return results
                
            except Exception as e:
                if isinstance(e, (InvalidInputError, KNNServiceError)):
                    raise
                raise KNNServiceError(f"Query processing failed: {e}")


def meta_to_dict(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metadata to dictionary with proper type handling.
    
    Args:
        meta: Input metadata dictionary.
        
    Returns:
        Processed metadata dictionary.
    """
    output = {}
    for k, v in meta.items():
        try:
            if isinstance(v, bytes):
                v = v.decode('utf-8', errors='replace')
            elif hasattr(v, 'item') and hasattr(v, 'dtype'):  # numpy types
                v = v.item()
            elif isinstance(v, (np.integer, np.floating)):
                v = v.item()
            output[k] = v
        except Exception as e:
            LOGGER.warning(f"Failed to process metadata key {k}: {e}")
            output[k] = str(v)
    return output


def load_index(
    path: str, 
    enable_faiss_memory_mapping: bool = True
) -> faiss.Index:
    """Loads the FAISS index.

    Args:
        path: The path to the FAISS index.
        enable_faiss_memory_mapping: Whether to enable memory mapping.

    Returns:
        The loaded FAISS index.
        
    Raises:
        ResourceLoadError: If index loading fails.
    """
    if not os.path.exists(path):
        raise ResourceLoadError(f"Index file not found: {path}")
    
    try:
        if enable_faiss_memory_mapping:
            if os.path.isdir(path):
                return faiss.read_index(
                    os.path.join(path, "populated.index"), 
                    faiss.IO_FLAG_ONDISK_SAME_DIR
                )
            else:
                return faiss.read_index(
                    path, 
                    faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
                )
        else:
            return faiss.read_index(path)
    except Exception as e:
        raise ResourceLoadError(f"Failed to load index from {path}: {e}")


class BaseMetadataProvider:
    """Base class for metadata providers."""
    
    def get(self, ids: List[int], cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get metadata for given IDs.
        
        Args:
            ids: List of IDs to retrieve metadata for.
            cols: Optional list of columns to retrieve.
            
        Returns:
            List of metadata dictionaries.
        """
        raise NotImplementedError


class ParquetMetadataProvider(BaseMetadataProvider):
    """Provides metadata from contiguous ids using parquet."""

    def __init__(self, parquet_folder: str):
        """Initialize parquet metadata provider.
        
        Args:
            parquet_folder: Path to folder containing parquet files.
            
        Raises:
            ResourceLoadError: If parquet files cannot be loaded.
        """
        if not os.path.exists(parquet_folder):
            raise ResourceLoadError(f"Parquet folder not found: {parquet_folder}")
        
        try:
            data_dir = Path(parquet_folder)
            parquet_files = list(data_dir.glob("*.parquet"))
            
            if not parquet_files:
                raise ResourceLoadError(f"No parquet files found in {parquet_folder}")
            
            LOGGER.info(f"Loading {len(parquet_files)} parquet files...")
            dataframes = []
            for parquet_file in sorted(parquet_files):
                df = pd.read_parquet(parquet_file)
                dataframes.append(df)
            
            self.metadata_df = pd.concat(dataframes, ignore_index=True)
            LOGGER.info(f"Loaded metadata for {len(self.metadata_df)} items")
            
        except Exception as e:
            raise ResourceLoadError(f"Failed to load parquet metadata: {e}")

    def get(self, ids: List[int], cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get metadata for given IDs."""
        try:
            if cols is None:
                cols = self.metadata_df.columns.tolist()
            else:
                available_cols = set(self.metadata_df.columns.tolist())
                cols = list(set(cols) & available_cols)
            
            results = []
            for idx in ids:
                if 0 <= idx < len(self.metadata_df):
                    row_data = self.metadata_df.iloc[idx][cols].to_dict()
                    results.append(row_data)
                else:
                    LOGGER.warning(f"Index {idx} out of range")
                    results.append({})
            
            return results
        except Exception as e:
            LOGGER.error(f"Failed to retrieve metadata: {e}")
            return [{} for _ in ids]


class Hdf5MetadataProvider(BaseMetadataProvider):
    """Provides metadata from contiguous ids using hdf5."""

    def __init__(self, hdf5_file: str):
        """Initialize HDF5 metadata provider.
        
        Args:
            hdf5_file: Path to HDF5 file.
            
        Raises:
            ResourceLoadError: If HDF5 file cannot be loaded.
        """
        if not os.path.exists(hdf5_file):
            raise ResourceLoadError(f"HDF5 file not found: {hdf5_file}")
        
        try:
            self.hdf5_file = hdf5_file
            self._file = None
            self._dataset = None
            self._lock = threading.Lock()
            self._ensure_open()
        except Exception as e:
            raise ResourceLoadError(f"Failed to initialize HDF5 provider: {e}")

    def _ensure_open(self):
        """Ensure HDF5 file is open."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_file, "r")
            self._dataset = self._file["dataset"]

    def get(self, ids: List[int], cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get metadata for given IDs."""
        with self._lock:
            try:
                self._ensure_open()
                
                items = [{} for _ in range(len(ids))]
                available_cols = list(self._dataset.keys())
                
                if cols is None:
                    cols = available_cols
                else:
                    cols = list(set(cols) & set(available_cols))
                
                for col in cols:
                    try:
                        for i, idx in enumerate(ids):
                            if 0 <= idx < len(self._dataset[col]):
                                items[i][col] = self._dataset[col][idx]
                            else:
                                LOGGER.warning(f"Index {idx} out of range for column {col}")
                    except Exception as e:
                        LOGGER.error(f"Failed to read column {col}: {e}")
                
                return items
            except Exception as e:
                LOGGER.error(f"Failed to retrieve HDF5 metadata: {e}")
                return [{} for _ in ids]

    def __del__(self):
        """Cleanup HDF5 resources."""
        if hasattr(self, '_file') and self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


def parquet_to_hdf5(
    parquet_folder: str,
    output_hdf5_file: str,
    columns_to_return: List[str] = None,
    compression: str = "gzip"
) -> None:
    """Convert a collection of parquet files to an hdf5 file.
    
    Args:
        parquet_folder: Path to folder containing parquet files.
        output_hdf5_file: Path for output HDF5 file.
        columns_to_return: List of columns to include.
        compression: Compression method for HDF5.
        
    Raises:
        ResourceLoadError: If conversion fails.
    """
    if columns_to_return is None:
        columns_to_return = ["url", "image_path", "caption", "NSFW"]
    
    data_dir = Path(parquet_folder)
    parquet_files = list(sorted(data_dir.glob("*.parquet")))
    
    if not parquet_files:
        raise ResourceLoadError(f"No parquet files found in {parquet_folder}")
    
    try:
        with h5py.File(output_hdf5_file, "w") as f:
            ds = f.create_group("dataset")
            
            for parquet_file in tqdm(parquet_files, desc="Converting parquet to HDF5"):
                try:
                    df = pd.read_parquet(parquet_file)
                    
                    for col in df.columns:
                        if col not in columns_to_return:
                            continue
                        
                        series = df[col]
                        
                        # Handle different data types
                        if series.dtype in ("float64", "float32"):
                            series = series.fillna(0.0)
                        elif series.dtype in ("int64", "int32"):
                            series = series.fillna(0)
                        elif series.dtype == "object":
                            series = series.fillna("")
                            # Remove null bytes that can cause issues
                            series = series.str.replace("\x00", "", regex=False)
                        
                        data = series.to_numpy()
                        
                        if col not in ds:
                            # Create dataset with chunking for better performance
                            ds.create_dataset(
                                col, 
                                data=data, 
                                maxshape=(None,), 
                                compression=compression,
                                chunks=True
                            )
                        else:
                            # Extend existing dataset
                            prev_len = len(ds[col])
                            ds[col].resize((prev_len + len(data),))
                            ds[col][prev_len:] = data
                            
                except Exception as e:
                    LOGGER.error(f"Failed to process {parquet_file}: {e}")
                    continue
                    
    except Exception as e:
        raise ResourceLoadError(f"Failed to convert parquet to HDF5: {e}")


def load_metadata_provider(
    indice_folder: str, 
    enable_hdf5: bool = False, 
    image_index: Optional[faiss.Index] = None
) -> BaseMetadataProvider:
    """Load the metadata provider.
    
    Args:
        indice_folder: Path to index folder.
        enable_hdf5: Whether to use HDF5 format.
        image_index: Optional image index (for compatibility).
        
    Returns:
        Metadata provider instance.
        
    Raises:
        ResourceLoadError: If metadata provider cannot be loaded.
    """
    parquet_folder = os.path.join(indice_folder, "metadata")
    
    if enable_hdf5:
        hdf5_path = os.path.join(indice_folder, "metadata.hdf5")
        if not os.path.exists(hdf5_path):
            LOGGER.info("Converting parquet to HDF5...")
            parquet_to_hdf5(parquet_folder, hdf5_path)
        return Hdf5MetadataProvider(hdf5_path)
    else:
        return ParquetMetadataProvider(parquet_folder)


@dataclass
class ClipResource:
    """The resource for clip: model, index, options."""
    
    device: str
    model: Optional[Any] = None
    preprocess: Optional[Callable] = None
    tokenizer: Optional[Callable] = None
    model_txt_mclip: Optional[Any] = None
    metadata_provider: Optional[BaseMetadataProvider] = None
    image_index: Optional[faiss.Index] = None
    text_index: Optional[faiss.Index] = None


@dataclass
class ClipOptions:
    """The options for clip."""
    
    indice_folder: str
    clip_model: str = "ViT-B/32"
    enable_hdf5: bool = False
    enable_faiss_memory_mapping: bool = True
    enable_mclip_option: bool = False
    use_jit: bool = True


def dict_to_clip_options(config_dict: Dict[str, Any], base_options: ClipOptions) -> ClipOptions:
    """Convert dictionary to ClipOptions, using base_options as defaults.
    
    Args:
        config_dict: Configuration dictionary.
        base_options: Base options to use as defaults.
        
    Returns:
        ClipOptions instance.
    """
    return ClipOptions(
        indice_folder=config_dict.get("indice_folder", base_options.indice_folder),
        clip_model=config_dict.get("clip_model", base_options.clip_model),
        enable_hdf5=config_dict.get("enable_hdf5", base_options.enable_hdf5),
        enable_faiss_memory_mapping=config_dict.get(
            "enable_faiss_memory_mapping", base_options.enable_faiss_memory_mapping
        ),
        enable_mclip_option=config_dict.get(
            "enable_mclip_option", base_options.enable_mclip_option
        ),
        use_jit=config_dict.get("use_jit", base_options.use_jit),
    )


def load_clip_index(clip_options: ClipOptions) -> ClipResource:
    """Load the clip index and create ClipResource.
    
    Args:
        clip_options: Configuration options.
        
    Returns:
        ClipResource instance.
        
    Raises:
        ResourceLoadError: If resources cannot be loaded.
    """
    try:
        import torch
    except ImportError:
        raise ResourceLoadError("PyTorch is required but not installed")
    
    try:
        # Try to import CLIP loading functions
        try:
            from load_clip import load_clip, get_tokenizer
        except ImportError:
            LOGGER.warning("load_clip module not found. CLIP functionality will be limited.")
            # Create dummy functions for testing
            def load_clip(model_name, use_jit=True, device="cpu"):
                return None, None
            def get_tokenizer(model_name):
                return None
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info(f"Using device: {device}")
        
        # Load CLIP model
        model, preprocess = load_clip(
            clip_options.clip_model, 
            use_jit=clip_options.use_jit, 
            device=device
        )
        
        tokenizer = get_tokenizer(clip_options.clip_model)
        
        # Load MCLIP if enabled
        model_txt_mclip = None
        if clip_options.enable_mclip_option:
            try:
                model_txt_mclip = load_mclip(clip_options.clip_model)
            except Exception as e:
                LOGGER.warning(f"Failed to load MCLIP: {e}")
        
        # Check for index files
        image_index_path = os.path.join(clip_options.indice_folder, "image.index")
        text_index_path = os.path.join(clip_options.indice_folder, "text.index")
        
        image_present = os.path.exists(image_index_path)
        text_present = os.path.exists(text_index_path)
        
        if not image_present and not text_present:
            raise ResourceLoadError("No index files found")
        
        # Load indices
        LOGGER.info("Loading indices...")
        image_index = None
        text_index = None
        
        if image_present:
            try:
                image_index = load_index(
                    image_index_path, 
                    clip_options.enable_faiss_memory_mapping
                )
                LOGGER.info(f"Loaded image index with {image_index.ntotal} vectors")
            except Exception as e:
                LOGGER.error(f"Failed to load image index: {e}")
        
        if text_present:
            try:
                text_index = load_index(
                    text_index_path, 
                    clip_options.enable_faiss_memory_mapping
                )
                LOGGER.info(f"Loaded text index with {text_index.ntotal} vectors")
            except Exception as e:
                LOGGER.error(f"Failed to load text index: {e}")
        
        # Load metadata provider
        LOGGER.info("Loading metadata provider...")
        metadata_provider = load_metadata_provider(
            clip_options.indice_folder,
            clip_options.enable_hdf5,
            image_index,
        )
        
        return ClipResource(
            device=device,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            model_txt_mclip=model_txt_mclip,
            metadata_provider=metadata_provider,
            image_index=image_index,
            text_index=text_index,
        )
        
    except Exception as e:
        raise ResourceLoadError(f"Failed to load CLIP resources: {e}")


def load_mclip(model_name: str):
    """Load multilingual CLIP model.
    
    Args:
        model_name: Model name.
        
    Returns:
        MCLIP model instance.
        
    Note:
        This is a placeholder function. Actual implementation depends on
        the specific MCLIP library being used.
    """
    # This is a placeholder - actual implementation would depend on the
    # specific multilingual CLIP library being used
    LOGGER.warning("MCLIP loading not implemented")
    return None


def load_clip_indices(
    indices_paths: str,
    clip_options: ClipOptions,
) -> Dict[str, ClipResource]:
    """Load CLIP indices from disk.
    
    Args:
        indices_paths: Path to JSON file containing index configurations.
        clip_options: Base CLIP options.
        
    Returns:
        Dictionary mapping index names to ClipResource instances.
        
    Raises:
        ResourceLoadError: If indices cannot be loaded.
    """
    LOGGER.info("Loading CLIP indices...")
    
    if not os.path.exists(indices_paths):
        raise ResourceLoadError(f"Indices file not found: {indices_paths}")
    
    try:
        with open(indices_paths, "r", encoding="utf-8") as f:
            indices_config = json.load(f)
    except Exception as e:
        raise ResourceLoadError(f"Failed to read indices configuration: {e}")
    
    if not isinstance(indices_config, dict):
        raise ResourceLoadError("Indices configuration must be a dictionary")
    
    clip_resources = {}
    
    for name, config in indices_config.items():
        try:
            LOGGER.info(f"Loading index: {name}")
            
            # Handle different configuration formats
            if isinstance(config, str):
                # Simple string path
                current_options = dict_to_clip_options(
                    {"indice_folder": config}, clip_options
                )
            elif isinstance(config, dict):
                # Dictionary configuration
                current_options = dict_to_clip_options(config, clip_options)
            else:
                raise ValueError(f"Invalid configuration type for {name}: {type(config)}")
            
            clip_resources[name] = load_clip_index(current_options)
            LOGGER.info(f"Successfully loaded index: {name}")
            
        except Exception as e:
            LOGGER.error(f"Failed to load index {name}: {e}")
            # Continue loading other indices
            continue
    
    if not clip_resources:
        raise ResourceLoadError("No indices could be loaded")
    
    return clip_resources


def create(
    indices_paths: str = "indices.json",
    enable_hdf5: bool = False,
    enable_faiss_memory_mapping: bool = True,
    enable_mclip_option: bool = False,
    clip_model: str = "ViT-B/32",
    use_jit: bool = True,
    **kwargs
) -> KnnService:
    """Create a KNN service instance.
    
    Args:
        indices_paths: Path to indices configuration file.
        enable_hdf5: Whether to use HDF5 for metadata storage.
        enable_faiss_memory_mapping: Whether to enable FAISS memory mapping.
        enable_mclip_option: Whether to enable multilingual CLIP.
        clip_model: CLIP model name to use.
        use_jit: Whether to use JIT compilation.
        **kwargs: Additional arguments passed to KnnService.
        
    Returns:
        KnnService instance.
        
    Raises:
        ResourceLoadError: If service cannot be created.
    """
    try:
        clip_options = ClipOptions(
            indice_folder="./index",
            clip_model=clip_model,
            enable_hdf5=enable_hdf5,
            enable_faiss_memory_mapping=enable_faiss_memory_mapping,
            enable_mclip_option=enable_mclip_option,
            use_jit=use_jit,
        )
        
        clip_resources = load_clip_indices(
            indices_paths=indices_paths,
            clip_options=clip_options,
        )
        
        knn_service = KnnService(clip_resources=clip_resources, **kwargs)
        LOGGER.info("KNN service created successfully")
        return knn_service
        
    except Exception as e:
        raise ResourceLoadError(f"Failed to create KNN service: {e}")


# Utility functions for testing and debugging
def validate_service_setup(service: KnnService) -> Dict[str, Any]:
    """Validate that a KNN service is properly set up.
    
    Args:
        service: KNN service instance.
        
    Returns:
        Dictionary with validation results.
    """
    results = {
        "valid": True,
        "issues": [],
        "resources": {},
    }
    
    try:
        if not service.clip_resources:
            results["valid"] = False
            results["issues"].append("No CLIP resources loaded")
            return results
        
        for name, resource in service.clip_resources.items():
            resource_info = {
                "device": resource.device,
                "has_model": resource.model is not None,
                "has_preprocess": resource.preprocess is not None,
                "has_tokenizer": resource.tokenizer is not None,
                "has_image_index": resource.image_index is not None,
                "has_text_index": resource.text_index is not None,
                "has_metadata_provider": resource.metadata_provider is not None,
            }
            
            # Check for at least one index
            if not resource_info["has_image_index"] and not resource_info["has_text_index"]:
                results["valid"] = False
                results["issues"].append(f"Resource {name} has no indices")
            
            # Check for model components
            if not resource_info["has_model"]:
                results["issues"].append(f"Resource {name} has no model")
            
            results["resources"][name] = resource_info
    
    except Exception as e:
        results["valid"] = False
        results["issues"].append(f"Validation error: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage
    try:
        service = create()
        validation = validate_service_setup(service)
        
        if validation["valid"]:
            print("KNN service setup successfully!")
        else:
            print("KNN service setup issues:")
            for issue in validation["issues"]:
                print(f"  - {issue}")
                
    except Exception as e:
        print(f"Failed to create service: {e}")

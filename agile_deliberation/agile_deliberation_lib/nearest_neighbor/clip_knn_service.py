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
# pylint: disable-all
import base64
from collections import defaultdict
from dataclasses import dataclass
import io
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import ssl
from typing import Any, Callable, Dict
import urllib

from clip_retrieval.clip_back import load_mclip
import torch
from load_clip import load_clip, get_tokenizer

import faiss
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger("agile_deliberation")


def convert_metadata_to_base64(meta):
  """Converts the image at a path to the Base64 string and sets it as the image.

  If there is no `image_path` key present in the metadata dictionary, the
  function will have no effect.

  Args:
    meta: The metadata dictionary.
  """
  if meta is not None and "image_path" in meta:
    path = meta["image_path"]
    if os.path.exists(path):
      img = Image.open(path)
      buffered = BytesIO()
      img.save(buffered, format="JPEG")
      img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
      meta["image"] = img_str


def download_image(url):
  """Download an image from a url and return a byte stream.

  Args:
    url: The url to downlaod from.

  Returns:
    A BytesIO stream of the image.
  """
  urllib_request = urllib.request.Request(
      url,
      data=None,
      headers={
          "User-Agent": (
              "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101"
              " Firefox/72.0"
          )
      },
  )
  urllib_context = ssl.create_default_context()
  urllib_context.set_alpn_protocols(["http/1.1"])

  with urllib.request.urlopen(
      urllib_request, timeout=10, context=urllib_context
  ) as r:
    img_stream = io.BytesIO(r.read())
  return img_stream


def normalized(a, axis=-1, order=2):
  """Normalize an array.

  Args:
    a: The input array.
    axis: The axis to normalize along.
    order: The order of the norm.

  Returns:
    The normalized array.
  """
  l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
  l2[l2 == 0] = 1
  return a / np.expand_dims(l2, axis)


class KnnService:
  """The KNN service provides nearest neighbors given text or image.

  Attributes:
    clip_resources: The resources for the KNN service.
  """

  def __init__(self, **kwargs):
    """Initializes the KnnService.

    Args:
      **kwargs: Keyword arguments containing 'clip_resources'.
    """
    super().__init__()
    self.clip_resources = kwargs["clip_resources"]

  def compute_query(
      self,
      clip_resource,
      text_input,
      image_input,
      image_url_input,
      embedding_input,
      use_mclip,
  ):
    """Computes the query embedding.

    Args:
      clip_resource: The CLIP resource.
      text_input: The text input.
      image_input: The base64 encoded image input.
      image_url_input: The URL of the image input.
      embedding_input: The precomputed embedding input.
      use_mclip: Whether to use mclip.

    Returns:
      The unified query embedding.
    """
    import torch  # pylint:disable=g-import-not-at-top

    if text_input is not None and text_input != "":
      if use_mclip:
        query = normalized(clip_resource.model_txt_mclip(text_input))
      else:
        text = clip_resource.tokenizer([text_input]).to(clip_resource.device)
        with torch.no_grad():
          text_features = clip_resource.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        query = text_features.cpu().to(torch.float32).detach().numpy()
    elif image_input is not None or image_url_input is not None:
      if image_input is not None:
        binary_data = base64.b64decode(image_input)
        img_data = BytesIO(binary_data)
      else:
        assert image_url_input is not None
        img_data = download_image(image_url_input)
      img = Image.open(img_data)
      prepro = (
          clip_resource.preprocess(img).unsqueeze(0).to(clip_resource.device)
      )
      with torch.no_grad():
        image_features = clip_resource.model.encode_image(prepro)
      image_features /= image_features.norm(dim=-1, keepdim=True)
      query = image_features.cpu().to(torch.float32).detach().numpy()
    elif embedding_input is not None:
      query = np.expand_dims(np.array(embedding_input).astype("float32"), 0)
    else:
      raise ValueError("must fill one of text, image, image url, or embedding input")

    return query

  def hash_based_dedup(self, embeddings):
    """Deduplicate embeddings based on their hash.

    Args:
      embeddings: The list of embedding vectors.

    Returns:
      A list of indices to remove due to duplication.
    """
    seen_hashes = set()
    to_remove = []
    for i, embedding in enumerate(embeddings):
      h = hash(np.round(embedding, 2).tobytes())
      if h in seen_hashes:
        to_remove.append(i)
        continue
      seen_hashes.add(h)

    return to_remove

  def connected_components(self, neighbors):
    """Find connected components in the graph.

    Args:
      neighbors: A mapping from nodes to their neighboring nodes.

    Returns:
      A list of connected components.
    """
    seen = set()

    def component(node):
      r = []
      nodes = set([node])
      while nodes:
        node = nodes.pop()
        seen.add(node)
        nodes |= set(neighbors[node]) - seen
        r.append(node)
      return r

    u = []
    for node in neighbors:
      if node not in seen:
        u.append(component(node))
    return u

  def get_non_uniques(self, embeddings, threshold=0.94):
    """Find non-unique embeddings.

    Args:
      embeddings: A numpy array of embeddings.
      threshold: The similarity threshold for equivalence.

    Returns:
      A list of indices that are not unique.
    """
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)  # pylint: disable=no-value-for-parameter
    l, _, I = index.range_search(embeddings, threshold)  # pylint: disable=no-value-for-parameter,invalid-name

    same_mapping = defaultdict(list)

    # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#range-search
    for i in range(embeddings.shape[0]):
      for j in I[l[i] : l[i + 1]]:
        same_mapping[int(i)].append(int(j))

    groups = self.connected_components(same_mapping)
    non_uniques = set()
    for g in groups:
      for e in g[1:]:
        non_uniques.add(e)

    return list(non_uniques)

  def connected_components_dedup(self, embeddings):
    """Deduplicate embeddings using connected components.

    Args:
      embeddings: A numpy array of embeddings.

    Returns:
      A list of indices that are not unique.
    """
    non_uniques = self.get_non_uniques(embeddings)
    return non_uniques

  def get_violent_items(self, safety_prompts, embeddings):
    """Get violent items from embeddings.

    Args:
      safety_prompts: The safety prompt embeddings.
      embeddings: The item embeddings.

    Returns:
      An array of indices of violent items.
    """
    safety_predictions = np.einsum("ij,kj->ik", embeddings, safety_prompts)
    safety_results = np.argmax(safety_predictions, axis=1)
    return np.where(safety_results == 1)[0]

  def post_filter(self, embeddings, deduplicate):
    """Post filter results by deduping.

    Args:
      embeddings: The result embeddings.
      deduplicate: Whether to deduplicate.

    Returns:
      A set of local indices to remove.
    """
    to_remove = set()
    if deduplicate:
      to_remove = set(self.connected_components_dedup(embeddings))

    return to_remove

  def knn_search(
      self, query, modality, num_result_ids, clip_resource, deduplicate
  ):
    """Compute the knn search.

    Args:
      query: The query embedding.
      modality: The search modality (e.g., 'image' or 'text').
      num_result_ids: The number of results to fetch.
      clip_resource: The CLIP resource context.
      deduplicate: Whether to deduplicate results.

    Returns:
      A tuple of lists: distances, indices, and embeddings.
    """

    image_index = clip_resource.image_index
    text_index = clip_resource.text_index

    index = image_index if modality == "image" else text_index

    distances, indices, embeddings = index.search_and_reconstruct(
        query, num_result_ids
    )
    results = indices[0]

    nb_results = np.where(results == -1)[0]

    if len(nb_results) > 0:
      nb_results = nb_results[0]
    else:
      nb_results = len(results)
    result_indices = results[:nb_results]
    result_distances = distances[0][:nb_results]
    result_embeddings = embeddings[0][:nb_results]
    result_embeddings = normalized(result_embeddings)
    local_indices_to_remove = self.post_filter(result_embeddings, deduplicate)
    indices_to_remove = set()
    for local_index in local_indices_to_remove:
      indices_to_remove.add(result_indices[local_index])
    indices = []
    distances = []
    embeddings = []
    for ind, distance, emb in zip(
        result_indices, result_distances, result_embeddings
    ):
      if ind not in indices_to_remove:
        indices_to_remove.add(ind)
        indices.append(ind)
        distances.append(distance)
        embeddings.append(emb)

    return distances, indices, embeddings

  def map_to_metadata(
      self, indices, distances, embeddings, num_images, metadata_provider
  ):
    """Map the indices to the metadata.

    Args:
      indices: The list of result indices.
      distances: The list of calculated distances.
      embeddings: The list of embeddings.
      num_images: The number of maximum images to return.
      metadata_provider: The metadata provider instance.

    Returns:
      A list of result dictionaries containing metadata and metadata outputs.
    """

    results = []

    metas = metadata_provider.get(indices[:num_images])

    for key, (d, i, e) in enumerate(zip(distances, indices, embeddings)):
      output = {}
      meta = None if key + 1 > len(metas) else metas[key]
      convert_metadata_to_base64(meta)
      if meta is not None:
        output.update(meta_to_dict(meta))
      output["id"] = i.item()
      output["similarity"] = d.item()
      output["embedding"] = e
      results.append(output)

    return results

  def query(
      self,
      text_input=None,
      image_input=None,
      image_url_input=None,
      embedding_input=None,
      modality="image",
      num_images=100,
      num_result_ids=100,
      indice_name=None,
      use_mclip=False,
      deduplicate=True,
  ):
    """Implements the querying functionality of the knn service.

    Args:
      text_input: The text input. Only one input must be set.
      image_input: The image input. Only one input must be set.
      image_url_input: The image input by url. Only one input must be set.
      embedding_input: An embedding input.
      modality: The modality.
      num_images: The number of nearest neighbors.
      num_result_ids: The number of result ids.
      indice_name: The index name.
      use_mclip: Whether to use mclip.
      deduplicate: Dedupe results.

    Returns:
      The metadata of the nearest neighbors.

    Raises:
      ValueError: If none of the input modalities are specified.
    """

    if (
        text_input is None
        and image_input is None
        and image_url_input is None
        and embedding_input is None
    ):
      raise ValueError("must fill one of text, image and image url input")
    if indice_name is None:
      indice_name = next(iter(self.clip_resources.keys()))

    clip_resource = self.clip_resources[indice_name]

    query = self.compute_query(
        clip_resource=clip_resource,
        text_input=text_input,
        image_input=image_input,
        image_url_input=image_url_input,
        embedding_input=embedding_input,
        use_mclip=use_mclip,
    )
    distances, indices, embeddings = self.knn_search(
        query,
        modality=modality,
        num_result_ids=num_result_ids,
        clip_resource=clip_resource,
        deduplicate=deduplicate,
    )
    if len(distances) == 0:
      return []
    results = self.map_to_metadata(
        indices,
        distances,
        embeddings,
        num_images,
        clip_resource.metadata_provider,
    )

    return results


def meta_to_dict(meta):
  """Convert meta object to dictionary.

  Args:
    meta: The metadata structure to convert.

  Returns:
    A normalized dictionary representation.
  """
  output = {}
  for k, v in meta.items():
    if isinstance(v, bytes):
      v = v.decode()
    elif type(v).__module__ == np.__name__:
      v = v.item()
    output[k] = v
  return output


def load_index(path, enable_faiss_memory_mapping):
  """Loads the index.

  Args:
    path: The path to the FAISS index.
    enable_faiss_memory_mapping: Whether to enable memory mapping.

  Returns:
    The FAISS index.
  """
  if enable_faiss_memory_mapping:
    if os.path.isdir(path):
      return faiss.read_index(
          path + "/populated.index", faiss.IO_FLAG_ONDISK_SAME_DIR
      )
    else:
      return faiss.read_index(
          path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
      )
  else:
    return faiss.read_index(path)


class ParquetMetadataProvider:
  """Provides metadata from contiguous ids using parquet.

  Attributes:
    metadata_df: A pandas DataFrame containing the metadata.
  """

  def __init__(self, parquet_folder):
    """Initializes the ParquetMetadataProvider.

    Args:
      parquet_folder: The folder containing parquet files.
    """
    data_dir = Path(parquet_folder)
    self.metadata_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in sorted(data_dir.glob("*.parquet"))
    )

  def get(self, ids, cols=None):
    """Gets metadata for specific ids.

    Args:
      ids: The sequence of ids to retrieve metadata for.
      cols: A list of optional columns to retrieve.

    Returns:
      A list of metadata dictionaries.
    """
    if cols is None:
      cols = self.metadata_df.columns.tolist()
    else:
      cols = list(set(self.metadata_df.columns.tolist()) & set(cols))

    return [
        self.metadata_df[i : (i + 1)][cols].to_dict(orient="records")[0]
        for i in ids
    ]


def parquet_to_hdf5(
    parquet_folder,
    output_hdf5_file,
    columns_to_return=["url", "image_path", "caption", "NSFW"],
):
  """This converts a collection of parquet files to an hdf5 file.

  Args:
    parquet_folder: The directory of input parquet files.
    output_hdf5_file: The path of the target HDF5 file.
    columns_to_return: The list of schema columns to extract.
  """
  f = h5py.File(output_hdf5_file, "w")
  data_dir = Path(parquet_folder)
  ds = f.create_group("dataset")
  for parquet_files in tqdm(sorted(data_dir.glob("*.parquet"))):
    df = pd.read_parquet(parquet_files)
    for k in df.keys():
      if k not in columns_to_return:
        continue
      col = df[k]
      if col.dtype in ("float64", "float32"):
        col = col.fillna(0.0)
      if col.dtype in ("int64", "int32"):
        col = col.fillna(0)
      if col.dtype == "object":
        col = col.fillna("")
        col = col.str.replace("\x00", "", regex=False)
      z = col.to_numpy()
      if k not in ds:
        ds.create_dataset(k, data=z, maxshape=(None,), compression="gzip")
      else:
        dataset = ds[k]
        assert isinstance(dataset, h5py.Dataset)
        prevlen = len(dataset)
        dataset.resize((prevlen + len(z),))
        dataset[prevlen:] = z

  del ds
  f.close()


class Hdf5MetadataProvider:
  """Provides metadata from contiguous ids using hdf5.

  Attributes:
    ds: The active HDF5 dataset group.
  """

  def __init__(self, hdf5_file):
    """Initializes the Hdf5MetadataProvider.

    Args:
      hdf5_file: The path to the HDF5 file.
    """
    f = h5py.File(hdf5_file, "r")
    ds = f["dataset"]
    assert isinstance(ds, h5py.Group)
    self.ds = ds

  def get(self, ids, cols=None):
    """Gets metadata from ids.

    Args:
      ids: The sequence of ids to retrieve metadata for.
      cols: A list of optional columns to retrieve.

    Returns:
      A list of metadata dictionaries.
    """
    items = [{} for _ in range(len(ids))]
    if cols is None:
      cols = self.ds.keys()
    else:
      cols = list(self.ds.keys() & set(cols))
    for k in cols:
      for i, e in enumerate(ids):
        items[i][k] = self.ds[k][e]
    return items


def load_metadata_provider(indice_folder, enable_hdf5, image_index):
  """Load the metadata provider.

  Args:
    indice_folder: The folder holding indices and metadata data.
    enable_hdf5: Whether to prefer HDF5 loading or fallback to Parquet.
    image_index: The image index argument.

  Returns:
    The loaded metadata provider instance.
  """
  parquet_folder = indice_folder + "/metadata"
  if enable_hdf5:
    hdf5_path = indice_folder + "/metadata.hdf5"
    if not os.path.exists(hdf5_path):
      parquet_to_hdf5(parquet_folder, hdf5_path)
    metadata_provider = Hdf5MetadataProvider(hdf5_path)
  else:
    metadata_provider = ParquetMetadataProvider(parquet_folder)

  return metadata_provider


@dataclass
class ClipResource:
  """The resource for clip : model, index, options."""

  device: str
  model: Any
  preprocess: Callable
  tokenizer: Callable
  model_txt_mclip: Any
  metadata_provider: Any
  image_index: Any
  text_index: Any


@dataclass
class ClipOptions:
  """The options for clip."""

  indice_folder: str
  clip_model: str
  enable_hdf5: bool
  enable_faiss_memory_mapping: bool
  enable_mclip_option: bool
  use_jit: bool


def dict_to_clip_options(d, clip_options):
  """Convert dictionary to clip options.

  Args:
    d: The dictionary overriding clip options.
    clip_options: The default ClipOptions instance.

  Returns:
    A newly configured ClipOptions object.
  """
  return ClipOptions(
      indice_folder=d["indice_folder"]
      if "indice_folder" in d
      else clip_options.indice_folder,
      clip_model=d["clip_model"]
      if "clip_model" in d
      else clip_options.clip_model,
      enable_hdf5=d["enable_hdf5"]
      if "enable_hdf5" in d
      else clip_options.enable_hdf5,
      enable_faiss_memory_mapping=d["enable_faiss_memory_mapping"]
      if "enable_faiss_memory_mapping" in d
      else clip_options.enable_faiss_memory_mapping,
      enable_mclip_option=d["enable_mclip_option"]
      if "enable_mclip_option" in d
      else clip_options.enable_mclip_option,
      use_jit=d["use_jit"] if "use_jit" in d else clip_options.use_jit,
  )


def load_clip_index(clip_options):
  """Load the clip index.

  Args:
    clip_options: The ClipOptions parameters.

  Returns:
    A hydrated ClipResource object.
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = load_clip(
      clip_options.clip_model, use_jit=clip_options.use_jit, device=device
  )

  tokenizer = get_tokenizer(clip_options.clip_model)

  if clip_options.enable_mclip_option:
    model_txt_mclip = load_mclip(clip_options.clip_model)
  else:
    model_txt_mclip = None

  image_present = os.path.exists(clip_options.indice_folder + "/image.index")
  text_present = os.path.exists(clip_options.indice_folder + "/text.index")

  LOGGER.info("loading indices...")
  image_index = (
      load_index(
          clip_options.indice_folder + "/image.index",
          clip_options.enable_faiss_memory_mapping,
      )
      if image_present
      else None
  )
  text_index = (
      load_index(
          clip_options.indice_folder + "/text.index",
          clip_options.enable_faiss_memory_mapping,
      )
      if text_present
      else None
  )

  LOGGER.info("loading metadata...")

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


def load_clip_indices(
    indices_paths,
    clip_options,
):
  """This loads CLIP indices from disk.

  Args:
    indices_paths: The JSON path mapping logical names to folders.
    clip_options: the fallback base ClipOptions.

  Returns:
    A map of strings (namespaces) to ClipResource definitions.

  Raises:
    ValueError: If an unknown indice_folder type is encountered.
  """
  LOGGER.info("loading clip...")

  with open(indices_paths, "r", encoding="utf-8") as f:
    indices = json.load(f)

  clip_resources = {}

  for name, indice_value in indices.items():
    # if indice_folder is a string
    if isinstance(indice_value, str):
      clip_options = dict_to_clip_options(
          {"indice_folder": indice_value}, clip_options
      )
    elif isinstance(indice_value, dict):
      clip_options = dict_to_clip_options(indice_value, clip_options)
    else:
      raise ValueError("Unknown type for indice_folder")
    clip_resources[name] = load_clip_index(clip_options)
  return clip_resources


def create(
    indices_paths="indices.json",
    enable_hdf5=False,
    enable_faiss_memory_mapping=True,
    enable_mclip_option=True,
    clip_model="ViT-B/32",
    use_jit=True,
):
  """Creates the KNN service.

  Args:
    indices_paths: Path to the JSON index registry.
    enable_hdf5: If True, uses HDF5 for metadata caching.
    enable_faiss_memory_mapping: If True, uses mmap to optimize memory usages.
    enable_mclip_option: Uses multilingual capability, if applicable.
    clip_model: The architecture to instantiate (e.g. ViT-B/32).
    use_jit: If True, leverages torchscript.

  Returns:
    An initialized KnnService.
  """
  clip_resources = load_clip_indices(
      indices_paths=indices_paths,
      clip_options=ClipOptions(
          indice_folder="./index",
          clip_model=clip_model,
          enable_hdf5=enable_hdf5,
          enable_faiss_memory_mapping=enable_faiss_memory_mapping,
          enable_mclip_option=enable_mclip_option,
          use_jit=use_jit,
      ),
  )

  knn_service = KnnService(clip_resources=clip_resources)
  return knn_service
